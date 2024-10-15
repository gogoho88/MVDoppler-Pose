"""
- refer to "https://github.com/chinhsuanwu/mobilevit-pytorch/blob/master/mobilevit.py"
"""
import math
import torch
import torch.nn as nn

from einops import rearrange

from xformers.components.attention.core import scaled_dot_product_attention


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernal_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernal_size, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn, mode='self'):
        super().__init__()
        self.mode = mode
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        B, C, N, D_3 = x.shape
        qkv = self.to_qkv(x)
        qkv = rearrange(qkv, 'b c n (n_qkv h d) -> n_qkv b h n (c d)', n_qkv = 3, h = self.heads)
        qkv = qkv.flatten(1, 2)
        q, k, v = qkv.unbind()

        mask = (torch.rand((k.shape[1], k.shape[1])) <= 1).to(k.device)
        out = scaled_dot_product_attention(q, k, v, att_mask=mask)

        out = rearrange(out, '(b h) n (c d) -> b c n (h d)', b = B, c = C)
        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
    def forward(self, x):
        B, C, N_2, D = x.shape
        q = self.to_q(x[:,:,0:N_2//2,:])
        q = rearrange(q, 'b c n (h d) -> (b h c) n d', h = self.heads)
        k = self.to_k(x[:,:,N_2//2:,:])
        k = rearrange(k, 'b c n (h d) -> (b h c) n d', h = self.heads)
        v = self.to_v(x[:,:,N_2//2:,:])
        v = rearrange(v, 'b c n (h d) -> (b h c) n d', h = self.heads)
        

        mask = (torch.rand((k.shape[1], k.shape[1])) <= 1).to(k.device)
        out = scaled_dot_product_attention(q, k, v, att_mask=mask)

        out = rearrange(out, '(b h c) n d -> b c n (h d)', b = B, c = C)
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads, dim_head, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    def __init__(self, inp, oup, stride=1, expansion=4):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileViTBlock(nn.Module):
    def __init__(self, dim, depth, channel, kernel_size, patch_size, mlp_dim, dropout=0.):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    
    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)
        
        # Global representations
        _, _, h, w = x.shape
        x = rearrange(x, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x = self.transformer(x)
        x = rearrange(x, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

class MobileViTBlock_Cross(nn.Module):
    """
    V0: Indap: cross_attention + followed Cross_FF
    """
    def __init__(self, dim, depth, cross_attn_depth, channel, kernel_size, patch_size_single, patch_size_cross, mlp_dim, dropout=0.):
        super().__init__()
        self.ph_single, self.pw_single = patch_size_single
        self.ph, self.pw = patch_size_cross

        self.conv1 = conv_nxn_bn(channel, channel, kernel_size)
        self.conv2 = conv_1x1_bn(channel, dim)

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)
        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(cross_attn_depth):
            self.cross_attn_layers.append(nn.ModuleList([
                PreNorm(dim, CrossAttention(dim, 4, 8, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout)),
            ]))

        self.conv3 = conv_1x1_bn(dim, channel)
        self.conv4 = conv_nxn_bn(2 * channel, channel, kernel_size)
    
    def forward(self, x1, x2):
        y1 = x1.clone()
        y2 = x2.clone()

        # Local representations
        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x2 = self.conv1(x2)
        x2 = self.conv2(x2)
        
        # Global representations
        _, _, h, w = x1.shape
        x1 = rearrange(x1, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph_single, pw=self.pw_single)
        x2 = rearrange(x2, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph_single, pw=self.pw_single)

        x1 = self.transformer(x1)   # (TODO): test bet. weight share / non-share
        x2 = self.transformer(x2)

        x1 = rearrange(x1, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph_single, w=w//self.pw_single, ph=self.ph_single, pw=self.pw_single)
        x2 = rearrange(x2, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph_single, w=w//self.pw_single, ph=self.ph_single, pw=self.pw_single)
        
        x1 = rearrange(x1, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)
        x2 = rearrange(x2, 'b d (h ph) (w pw) -> b (ph pw) (h w) d', ph=self.ph, pw=self.pw)

        for cross_attn_1, f_1, f_2 in self.cross_attn_layers:
            cal_qkv = torch.cat((x1,x2), dim=2)
            cal_out = x1 + cross_attn_1(cal_qkv)
            x1_out = f_1(cal_out)  # (TODO): test cal_out = f_1(cal_out)+cal_out or cal_out = f_1(norm(cal_out))+cal_out
            
            cal_qkv = torch.cat((x2,x1), dim=2) 
            cal_out =  x2 + cross_attn_1(cal_qkv)
            x2_out = f_2(cal_out)

        x1 = rearrange(x1_out, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)
        x2 = rearrange(x2_out, 'b (ph pw) (h w) d -> b d (h ph) (w pw)', h=h//self.ph, w=w//self.pw, ph=self.ph, pw=self.pw)

        # Fusion
        x1= self.conv3(x1)
        x1 = torch.cat((x1, y1), 1)
        x1 = self.conv4(x1)
        x2 = self.conv3(x2)
        x2 = torch.cat((x2, y2), 1)
        x2 = self.conv4(x2)
        return (x1, x2)

class MobileViT(nn.Module):
    def __init__(self, args, image_size, dims, channels, expansion=4, kernel_size=3, patch_size=(2, 2), fusion_path_embed='time_centric'):
        super().__init__()
        self.fusion_level = args.fusion.fusion_level
        self.fusion_mode = args.fusion.fusion_mode
        self.project = args.project
        ih, iw = image_size
        if fusion_path_embed=='time_centric':
            patch_single = [(ih//8,1),(ih//16,1),(ih//32,1)]
            patch_cross = [(ih//8,1),(ih//16,1),(ih//32,1)]

        L = [3, 3, 3]
        L_C = [3, 3, 3]

        self.conv1 = conv_nxn_bn(1, channels[0], stride=2)

        self.mv2 = nn.ModuleList([])
        self.mv2.append(MV2Block(channels[0], channels[1], 1, expansion))
        self.mv2.append(MV2Block(channels[1], channels[2], 2, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))
        self.mv2.append(MV2Block(channels[2], channels[3], 1, expansion))   # Repeat
        self.mv2.append(MV2Block(channels[3], channels[4], 2, expansion))
        self.mv2.append(MV2Block(channels[5], channels[6], 2, expansion))
        self.mv2.append(MV2Block(channels[7], channels[8], 2, expansion))
        
        if self.fusion_mode=='cross':
            self.mvit_cross = nn.ModuleList([])
            self.mvit_cross.append(MobileViTBlock_Cross(dims[0], L[0], L_C[0], channels[5], kernel_size, patch_single[0], patch_cross[0], int(dims[0]*2)))
            self.mvit_cross.append(MobileViTBlock_Cross(dims[1], L[1], L_C[1], channels[7], kernel_size, patch_single[1], patch_cross[1], int(dims[1]*4)))
            self.mvit_cross.append(MobileViTBlock_Cross(dims[2], L[2], L_C[2], channels[9], kernel_size, patch_single[2], patch_cross[2], int(dims[2]*4)))
        else:
            self.mvit = nn.ModuleList([])
            self.mvit.append(MobileViTBlock(dims[0], L[0], channels[5], kernel_size, (16, 1), int(dims[0]*2)))
            self.mvit.append(MobileViTBlock(dims[1], L[1], channels[7], kernel_size, (8, 1), int(dims[1]*4)))
            self.mvit.append(MobileViTBlock(dims[2], L[2], channels[9], kernel_size, (4, 1), int(dims[2]*4)))

        self.conv2 = conv_1x1_bn(channels[-2], channels[-1])

        self.pool = nn.AdaptiveAvgPool2d((1,None))

    def encoding_single(self, x):
        x = self.conv1(x)
        x = self.mv2[0](x)

        x = self.mv2[1](x)
        x = self.mv2[2](x)
        x = self.mv2[3](x)      # Repeat

        x = self.mv2[4](x)
        x = self.mvit[0](x)

        x = self.mv2[5](x)
        x = self.mvit[1](x)

        x = self.mv2[6](x)
        x = self.mvit[2](x)
        x = self.conv2(x)

        return x

    def encoding_cross(self, x1, x2):
        # x1 processing before transformer
        x1 = self.conv1(x1)
        x1 = self.mv2[0](x1)

        x1 = self.mv2[1](x1)
        x1 = self.mv2[2](x1)
        x1 = self.mv2[3](x1)      # Repeat
        # x2 processing before transformer
        x2 = self.conv1(x2)
        x2 = self.mv2[0](x2)

        x2 = self.mv2[1](x2)
        x2 = self.mv2[2](x2)
        x2 = self.mv2[3](x2)      # Repeat

        # cross-attention transformer for multi-view fusion
        x1 = self.mv2[4](x1)
        x2 = self.mv2[4](x2)
        (x1, x2) = self.mvit_cross[0](x1, x2)

        x1 = self.mv2[5](x1)
        x2 = self.mv2[5](x2)
        (x1, x2) = self.mvit_cross[1](x1, x2)

        x1 = self.mv2[6](x1)
        x2 = self.mv2[6](x2)
        (x1, x2) = self.mvit_cross[2](x1, x2)
                
        x1 = self.conv2(x1)
        x2 = self.conv2(x2)

        return (x1, x2)

    def forward(self, x):
        if 'single' in self.fusion_level:
            if '1' in self.fusion_level:
                x = x[:,0,:,:].unsqueeze(dim=1)
            elif '2' in self.fusion_level:
                x = x[:,1,:,:].unsqueeze(dim=1)
            x = self.encoding_single(x)
            x = self.pool(x).squeeze(dim=(2))
        else:
            x1 = x[:,0,:,:].unsqueeze(dim=1)
            x2 = x[:,1,:,:].unsqueeze(dim=1)
            if 'cross' in self.fusion_mode:
                x1, x2 = self.encoding_cross(x1,x2)
                x1 = self.pool(x1).squeeze(dim=(2))
                x2 = self.pool(x2).squeeze(dim=(2))
                x = (x1+x2)/2
            elif 'late' in self.fusion_level:
                if 'average' in self.fusion_mode:
                    x1 = self.encoding_single(x1)
                    x2 = self.encoding_single(x2)
                    x1 = self.pool(x1).squeeze(dim=(2))
                    x2 = self.pool(x2).squeeze(dim=(2))
                    x = (x1+x2)/2
        return x

class main_Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        model_type = args.train.model
        self.decoder_input = args.model.decoder_input
        if 'mobileVit' in model_type:
            fusion_patch_embedding = 'time_centric'
            if 'xxs' in model_type:
                dims = [64, 80, 96]
                channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
                expansion = 2
            elif 'xs' in model_type:
                dims = [96, 120, 144]
                channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
                expansion = 4
            elif 's' in model_type:
                dims = [144, 192, 240]
                channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
                expansion = 4
            self.radar_mD_encoder = MobileViT(args, image_size=(args.transforms.Dop_size, args.transforms.win_size), 
                                                    dims=dims, 
                                                    channels=channels, 
                                                    expansion=expansion, 
                                                    fusion_path_embed=fusion_patch_embedding)
            self.radar_Rng_encoder = MobileViT(args, image_size=(args.transforms.R_size_rng, args.transforms.win_size_rng), 
                                                    dims=dims, 
                                                    channels=channels, 
                                                    expansion=expansion, 
                                                    fusion_path_embed=fusion_patch_embedding)
        self.regress_mD = nn.Sequential(
                        nn.Conv1d(channels[-1], 3*17, kernel_size=1),
                        nn.BatchNorm1d(3*17),
                        nn.Tanh()
                        )
        self.regress_Rng = nn.Sequential(
                        nn.Conv1d(channels[-1], 3*17, kernel_size=1),
                        nn.BatchNorm1d(3*17),
                        nn.Tanh()
                        )
        if self.decoder_input=='all':
            decoder_dim = 20
        elif self.decoder_input=='vel':
            decoder_dim = 16
        elif self.decoder_input=='rng':
            decoder_dim = 4
        self.fc = nn.Sequential(
                        nn.Linear(decoder_dim*3*17, 16*3*17),
                        nn.Tanh(),
                        nn.Dropout(p=0.5),
                        nn.Linear(16*3*17, 16*3*17)
                        )
        # self.fc_init = nn.Sequential(
        #                 nn.Linear(20*3*17, 1*3*17),
        #                 nn.Tanh(),
        #                 nn.Dropout(p=0.5),
        #                 nn.Linear(1*3*17, 1*3*17)
        #                 )
        # self.fc_vel = nn.Sequential(
        #                 nn.Linear(20*3*17, 16*3*17),
        #                 nn.Tanh(),
        #                 nn.Dropout(p=0.5),
        #                 nn.Linear(16*3*17, 16*3*17)
        #                 )
        #initialize
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x_mD, x_R):
        # Encoder
        x_mD = self.radar_mD_encoder(x_mD)
        x_R = self.radar_Rng_encoder(x_R)
        # Decoder
        _,_,T_mD = x_mD.size()
        _,_,T_R = x_R.size()

        x_mD = self.regress_mD(x_mD).view(-1,T_mD*17*3)     # 17x3xT_mD
        x_R = self.regress_Rng(x_R).view(-1,T_R*17*3)       # 17x3xT_R

        if self.decoder_input=='all':
            x = self.fc(torch.cat((x_mD,x_R),dim=-1))
        elif self.decoder_input=='vel':
            x = self.fc(x_mD)
        elif self.decoder_input=='rng':
            x = self.fc(x_R)
        x = rearrange(x, 'b (t j c) -> b t j c', j=17, t=16, c=3).contiguous()
        return x
        




def mobilevit_xxs(args):
    dims = [64, 80, 96]
    channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320]
    return MobileViT(args, (args.transforms.Dop_size, args.transforms.win_size), dims, channels, expansion=2)


def mobilevit_xs(args):
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384]
    return MobileViT(args, (args.transforms.Dop_size, args.transforms.win_size), dims, channels)


def mobilevit_s(args):
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640]
    return MobileViT(args, (args.transforms.Dop_size, args.transforms.win_size), dims, channels)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    img = torch.randn(5, 1, 256, 256)
    
    vit = mobilevit_xxs()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))

    vit = mobilevit_xs()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))

    vit = mobilevit_s()
    out = vit(img)
    print(out.shape)
    print(count_parameters(vit))