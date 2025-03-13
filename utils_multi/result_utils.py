import os
import torch
import copy
import numpy as np
from scipy import signal
from numpy import linalg as LA
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from .camera import *


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def mae(output, target, flag_temporal=False):
    distance = torch.abs((output-target))
    if flag_temporal:
        return distance.to('cpu').numpy()
    else:
        return distance.mean(dim=(1,2)).to('cpu').numpy()

def compute_similarity_transform(S1: torch.Tensor, S2: torch.Tensor) -> torch.Tensor:
    """
    Computes a similarity transform (sR, t) in a batched way that takes
    a set of 3D points S1 (B, N, 3) closest to a set of 3D points S2 (B, N, 3),
    where R is a 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    Args:
        S1 (torch.Tensor): First set of points of shape (B, N, 3).
        S2 (torch.Tensor): Second set of points of shape (B, N, 3).
    Returns:
        (torch.Tensor): The first set of points after applying the similarity transformation.
    """

    batch_size = S1.shape[0]
    S1 = S1.permute(0, 2, 1)
    S2 = S2.permute(0, 2, 1)
    # 1. Remove mean.
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = (X1**2).sum(dim=(1,2))

    # 3. The outer product of X1 and X2.
    K = torch.matmul(X1, X2.permute(0, 2, 1))

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are singular vectors of K.
    U, s, V = torch.svd(K)
    Vh = V.permute(0, 2, 1)

    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = torch.eye(U.shape[1], device=U.device).unsqueeze(0).repeat(batch_size, 1, 1)
    Z[:, -1, -1] *= torch.sign(torch.linalg.det(torch.matmul(U, Vh)))

    # Construct R.
    R = torch.matmul(torch.matmul(V, Z), U.permute(0, 2, 1))

    # 5. Recover scale.
    trace = torch.matmul(R, K).diagonal(offset=0, dim1=-1, dim2=-2).sum(dim=-1)
    scale = (trace / var1).unsqueeze(dim=-1).unsqueeze(dim=-1)

    # 6. Recover translation.
    t = mu2 - scale*torch.matmul(R, mu1)

    # 7. Error:
    S1_hat = scale*torch.matmul(R, S1) + t

    return S1_hat.permute(0, 2, 1)

def pa_mpjpe(output, target, flag_temporal=False):
    B, N, T, _, _ = output.size()
    output_B = output.view(-1,17,3)
    target_B = target.view(-1,17,3)
    output_hat = compute_similarity_transform(output_B, target_B)
    distance = torch.norm(output_hat-target_B, dim=len(target_B.shape)-1).view(B,N,T,17)
    if flag_temporal:
        return distance.to('cpu').numpy()
    else:
        return distance.mean(dim=(2)).to('cpu').numpy()

def mpjpe(output, target, flag_temporal=False):
    distance = torch.norm(output-target, dim=len(target.shape)-1)
    if flag_temporal:
        return distance.to('cpu').numpy()
    else:
        return distance.mean(dim=(2)).to('cpu').numpy()

def PCC(output, target, flag_temporal=False):
    """
    output: B, N_frame,T, 17, 3
    target: B, N_frame,T, 17, 3
    """
    eps=1e-6
    target_mean = target.mean(dim=2).unsqueeze(dim=2)
    output_mean = output.mean(dim=2).unsqueeze(dim=2)
    Pcc = torch.sum((target-target_mean)*(output-output_mean),dim=2) / \
            ((torch.sqrt(torch.sum((target-target_mean)**2,dim=2)+eps))*(torch.sqrt(torch.sum((output-output_mean)**2,dim=2)+eps)))
    if flag_temporal:
        temporal_dim = target.shape[2]
        Pcc = Pcc.mean(dim=(3)).to('cpu').numpy()
        Pcc_tile = np.tile(Pcc,(1, temporal_dim)).reshape((Pcc.shape[0],Pcc.shape[1],temporal_dim,Pcc.shape[2]))
        return Pcc_tile
    else:
        return Pcc.mean(dim=(3)).to('cpu').numpy()

def PCK(output, target, threshold=0.15, flag_temporal=False):
    """
    in
        output: B, N_frame,T, 17, 3
        target: B, N_frame,T, 17, 3
    out
        PCK: B, 17
    """
    # threshold_sel = (target[:,:,:,10,:]-target[:,:,:,8,:]).norm(dim=3)  # L2 distance
    # threshold_sel = threshold*0.5
    # threshold_sel = threshold_sel.unsqueeze(dim=3).repeat(1,1,1,17)
    threshold_sel = threshold

    distance = (target-output).norm(dim=4)
    detection = (distance <= threshold_sel)
    PCK = detection.sum(dim=(2))/(detection.shape[2])

    if flag_temporal:
        return detection.to('cpu').numpy()
    else:
        return PCK.to('cpu').numpy()

def heading_direction_xy(output, target, device):
    """
    Calculate human heading direction (theta) in xy domain
    in
        output: B, N_frame,T, 17, 3
        target: B, N_frame,T, 17, 3
    out
        PCK: B, 17
    """
    target = torch.from_numpy(target).to(device)
    output = torch.from_numpy(output).to(device)
    target_direction_Rpelvis_xy = (target[:,:,:,1,(0,2)]/(target[:,:,:,1,(0,2)].norm(dim=-1).unsqueeze(dim=-1))) # xy vector from centor to R pelvis (target)
    # output_direction_Rpelvis_xy = output[:,:,:,1,(0,2)]/(output[:,:,:,1,(0,2)].norm(dim=-1).unsqueeze(dim=-1))  # xy vector from centor to R pelvis (pred)
    output_direction_Rpelvis_xy = ((output[:,:,:,1,(0,2)]-output[:,:,:,0,(0,2)])/((output[:,:,:,1,(0,2)]-output[:,:,:,0,(0,2)]).norm(dim=-1).unsqueeze(dim=-1)))
                                    # xy vector from centor to R pelvis (pred)
    # cal angle: cos(theta) = dotproduct(x,y)/|x||y|
    dotproduct_xy = torch.matmul(target_direction_Rpelvis_xy.unsqueeze(dim=-1).permute(0,1,2,4,3), output_direction_Rpelvis_xy.unsqueeze(dim=-1)).squeeze()
    dotproduct_xy[dotproduct_xy>=1]=1.
    dotproduct_xy[dotproduct_xy<=-1]=-1.
    angle_xy = (torch.acos(dotproduct_xy)*180./torch.pi).to('cpu').numpy()
    target_angle_xy = np.unwrap(torch.atan2(target_direction_Rpelvis_xy[:,:,:,1],
                            target_direction_Rpelvis_xy[:,:,:,0]).to('cpu').numpy(),axis=-1)*180./torch.pi + 90. + 90.
    output_angle_xy = np.unwrap(torch.atan2(output_direction_Rpelvis_xy[:,:,:,1],
                            output_direction_Rpelvis_xy[:,:,:,0]).to('cpu').numpy(),axis=-1)*180./torch.pi + 90. + 90.

    return angle_xy, output_angle_xy, target_angle_xy

def estimate_freq_peak(data_input, vec_s):
    # estimate frequency from sinosoidal signal using periodogram
    # data_input: 1D vector
    vec_len = len(data_input)
    sos = signal.butter(2, 1.6, 'lowpass', fs=vec_len/vec_s, output='sos') # band pass filter
    data_norm = (data_input - np.mean(data_input))/np.std(data_input)
    data_filter = signal.sosfilt(sos,data_norm)
    t = np.linspace(0, 3, len(data_filter))
    tnew = np.linspace(0, 3, 90)
    data_filter_resize = np.interp(tnew,t,data_filter)
    peaks, properties = signal.find_peaks(data_filter_resize, prominence=0.4, width=4)
    peaks_neg, properties_neg = signal.find_peaks(-1*data_filter_resize, prominence=0.4, width=4)
    freq = (len(peaks)+len(peaks_neg))/2/vec_s
    return freq

def estimate_freq_psd(data_input, vec_s):
    # estimate frequency from sinosoidal signal using periodogram
    # data_input: 1D vector
    vec_len = len(data_input)
    sos = signal.butter(2, 1.6, 'lowpass', fs=vec_len/vec_s, output='sos') # band pass filter
    data_norm = (data_input - np.mean(data_input))/np.std(data_input)
    data_filter = signal.sosfilt(sos,data_norm)
    [f,Sf] = signal.periodogram(data_filter,fs=vec_len/vec_s,nfft=128)
    freq = f[np.argmax(Sf)]
    return freq

def freq_filter(data_input, vec_s):
    # estimate frequency from sinosoidal signal using periodogram
    # data_input: 1D vector
    vec_len = len(data_input)
    sos = signal.butter(2, 1.6, 'lowpass', fs=vec_len/vec_s, output='sos') # band pass filter
    data_norm = (data_input - np.mean(data_input))/np.std(data_input)
    data_filter = signal.sosfilt(sos,data_norm)
    data_out = data_filter
    return data_out

def limb_freq(output, target, joint_pelvis, joint_limb, win_sec, des, device):
    """
    Calculate freq. of each limb joint
    in
        output: B, N_frame,T, 17, 3
        target: B, N_frame,T, 17, 3
    out
        PCK: B, 17
    """
    target = torch.from_numpy(target).to(device)
    output = torch.from_numpy(output).to(device)
    output = output-output[:,:,:,0,:].unsqueeze(dim=-2) # center align
    
    if joint_limb==16:
        joint_pelvis = 11

    target_uvec_pelvis_xy = target[:,:,:,joint_pelvis,(0,2)]/(target[:,:,:,joint_pelvis,(0,2)].norm(dim=-1).unsqueeze(dim=-1))  
    target_uvec_limb_xy = target[:,:,:,joint_limb,(0,2)]/(target[:,:,:,joint_limb,(0,2)].norm(dim=-1).unsqueeze(dim=-1))
    target_ang_pelvis_xy = torch.atan2(target_uvec_pelvis_xy[:,:,:,1],target_uvec_pelvis_xy[:,:,:,0]).to('cpu').numpy()
    target_ang_limb_xy = torch.atan2(target_uvec_limb_xy[:,:,:,1],target_uvec_limb_xy[:,:,:,0]).to('cpu').numpy()

    output_uvec_pelvis_xy = output[:,:,:,joint_pelvis,(0,2)]/(output[:,:,:,joint_pelvis,(0,2)].norm(dim=-1).unsqueeze(dim=-1))
    output_uvec_limb_xy = output[:,:,:,joint_limb,(0,2)]/(output[:,:,:,joint_limb,(0,2)].norm(dim=-1).unsqueeze(dim=-1))
    output_ang_pelvis_xy = torch.atan2(output_uvec_pelvis_xy[:,:,:,1],output_uvec_pelvis_xy[:,:,:,0]).to('cpu').numpy()
    output_ang_limb_xy = torch.atan2(output_uvec_limb_xy[:,:,:,1],output_uvec_limb_xy[:,:,:,0]).to('cpu').numpy()

    if joint_limb==16:
        target_ang_pelvis_xy = target_ang_pelvis_xy + np.pi
        output_ang_pelvis_xy = output_ang_pelvis_xy + np.pi
        target_ang_pelvis_xy = np.where(target_ang_pelvis_xy<=np.pi,target_ang_pelvis_xy,target_ang_pelvis_xy-2*np.pi)
        output_ang_pelvis_xy = np.where(output_ang_pelvis_xy<=np.pi,output_ang_pelvis_xy,output_ang_pelvis_xy-2*np.pi)

    if (('pockets' in des[0]['pattern']) or ('texting' in des[0]['pattern'])):    # hand
        if ((joint_limb==13) or (joint_limb==16)):
            target_ang_vel = np.unwrap((target_ang_limb_xy-target_ang_pelvis_xy), axis=-1)
            output_ang_vel = np.unwrap((output_ang_limb_xy-output_ang_pelvis_xy), axis=-1)
        else:
            target_ang_vel = np.diff(np.unwrap((target_ang_limb_xy-target_ang_pelvis_xy), axis=-1),axis=-1)
            output_ang_vel = np.diff(np.unwrap((output_ang_limb_xy-output_ang_pelvis_xy), axis=-1),axis=-1)
    elif ('phone_call') in des[0]['pattern']:
        if (joint_limb==16):
            target_ang_vel = np.unwrap((target_ang_limb_xy-target_ang_pelvis_xy), axis=-1)
            output_ang_vel = np.unwrap((output_ang_limb_xy-output_ang_pelvis_xy), axis=-1)
        else:
            target_ang_vel = np.diff(np.unwrap((target_ang_limb_xy-target_ang_pelvis_xy), axis=-1),axis=-1)
            output_ang_vel = np.diff(np.unwrap((output_ang_limb_xy-output_ang_pelvis_xy), axis=-1),axis=-1)
    else:
        target_ang_vel = np.diff(np.unwrap((target_ang_limb_xy-target_ang_pelvis_xy), axis=-1),axis=-1)
        output_ang_vel = np.diff(np.unwrap((output_ang_limb_xy-output_ang_pelvis_xy), axis=-1),axis=-1)

    B, N, T = target_ang_vel.shape
    target_f_all = np.zeros((B,N))
    output_f_all = np.zeros((B,N))
    for b_sel in range(B):
        for n_sel in range(N):
            target_f = estimate_freq_peak(target_ang_vel[b_sel,n_sel,:], win_sec)
            output_f = estimate_freq_peak(output_ang_vel[b_sel,n_sel,:], win_sec)
            target_f_all[b_sel,n_sel] = target_f
            output_f_all[b_sel,n_sel] = output_f

    return output_f_all, target_f_all, output_ang_vel, target_ang_vel

def nPCC_loss(output, target, eps=1e-6):
    target_mean = target.mean(dim=1).unsqueeze(dim=1)
    output_mean = output.mean(dim=1).unsqueeze(dim=1)
    Pcc = torch.sum((target-target_mean)*(output-output_mean),dim=1) / \
        ((torch.sqrt(torch.sum((target-target_mean)**2,dim=1)+eps))*(torch.sqrt(torch.sum((output-output_mean)**2,dim=1)+eps)))
    ccc = 1-Pcc
    return ccc

def motion_cal(predicted, target, intervals=[2, 4, 6, 8], operator=torch.cross):
    assert predicted.shape == target.shape
    loss = 0
    for itv in intervals:
        pred_encode = operator(predicted[:, :-itv], predicted[:, itv:], dim=3)
        target_encode = operator(target[:, :-itv], target[:, itv:], dim=3)
        # loss += torch.mean(torch.abs(pred_encode - target_encode)) / len(intervals)
        loss += torch.mean(nPCC_loss(pred_encode, target_encode)) / len(intervals)
    return loss

def test_keypoint(data_test, device, model, output_pred=False, output_temporal=False):
    """Test the loss and accuracy of the model on a data set 

    Args:
        data_test: data set
        device: device which the model runs on
        model: training model
        detail: perform detailed analysis (direction, limb freq.)
        output_pred: True - output the GT and predicted keypoints together
        output_temporal: True - output the temporal dimension / False - temporal dimension is collapsed from output

    Returns:
        accuracy and loss
    """
    test_loss = 0.0
    total = 0
    des_test = []
    MPJPE_all_list = []
    PAMPJPE_all_list = []
    PCC_all_list = []
    PCK_all_list = []
    PCK25_all_list = []
    PCK35_all_list = []
    MAE_all_list = []
    if output_pred:
        y_test = []
        y_test_pred = []
    for idx, (x_batch, x_R_batch, y_batch, des) in enumerate(data_test):
        with torch.no_grad():
            x_batch = x_batch.to(device, dtype=torch.float)
            x_R_batch = x_R_batch.to(device, dtype=torch.float)
            y_batch = y_batch.to(device, dtype=torch.float) 

            B,N_div,N_R,F,T = x_batch.size()
            _,_,_,R_Rng,T_Rng = x_R_batch.size()

            x_batch = x_batch.view(-1,N_R,F,T)
            x_R_batch = x_R_batch.view(-1,N_R,R_Rng,T_Rng)
            y_batch = y_batch.view(-1,16,17,3)

            model.eval()
            output = model(x_batch, x_R_batch)

            total += len(y_batch)

            y_batch = y_batch.view(B,N_div,16,17,3)
            output = output.view(B,N_div,16,17,3)

            MPJPE_all = mpjpe(output, y_batch, flag_temporal=output_temporal)
            PAMPJPE_all = pa_mpjpe(output, y_batch, flag_temporal=output_temporal)
            Pcc_all = PCC(output, y_batch, flag_temporal=output_temporal)
            PCK_all = PCK(output, y_batch, flag_temporal=output_temporal)
            PCK_all_25 = PCK(output, y_batch, threshold=0.20, flag_temporal=output_temporal)
            PCK_all_35 = PCK(output, y_batch, threshold=0.25, flag_temporal=output_temporal)
            MAE_all = mae(output, y_batch, flag_temporal=output_temporal)
            if output_pred:
                y_test.append(y_batch.to('cpu').numpy())
                y_test_pred.append(output.to('cpu').numpy())
            
            # append list
            MPJPE_all_list.append(MPJPE_all), PAMPJPE_all_list.append(PAMPJPE_all), PCC_all_list.append(Pcc_all), PCK_all_list.append(PCK_all), MAE_all_list.append(MAE_all)
            PCK25_all_list.append(PCK_all_25), PCK35_all_list.append(PCK_all_35)
            # append des
            for j in range(len(des['ID'])):
                des_sample = {}
                for key in des.keys():
                    des_sample[key] = des[key][j]
                des_test.append(des_sample)

    # flatten
    if output_temporal:
        temporal_dim = y_batch.shape[2]
    else:
        temporal_dim = 1
    MPJPE_all_array = np.stack(MPJPE_all_list[:-1]).reshape(-1,N_div,temporal_dim,17)
    MPJPE_all_array = np.append(MPJPE_all_array, MPJPE_all_list[-1], axis=0)
    PAMPJPE_all_array = np.stack(PAMPJPE_all_list[:-1]).reshape(-1,N_div,temporal_dim,17)
    PAMPJPE_all_array = np.append(PAMPJPE_all_array, PAMPJPE_all_list[-1], axis=0)
    PCC_all_array = np.stack(PCC_all_list[:-1]).reshape(-1,N_div,temporal_dim,17)
    PCC_all_array = np.append(PCC_all_array, PCC_all_list[-1], axis=0)
    PCK_all_array = np.stack(PCK_all_list[:-1]).reshape(-1,N_div,temporal_dim,17)
    PCK_all_array = np.append(PCK_all_array, PCK_all_list[-1], axis=0)
    MAE_all_array = np.stack(MAE_all_list[:-1]).reshape(-1,N_div,temporal_dim,17,3)
    MAE_all_array = np.append(MAE_all_array, MAE_all_list[-1], axis=0)
    PCK25_all_array = np.stack(PCK25_all_list[:-1]).reshape(-1,N_div,temporal_dim,17)
    PCK25_all_array = np.append(PCK25_all_array, PCK25_all_list[-1], axis=0)
    PCK35_all_array = np.stack(PCK35_all_list[:-1]).reshape(-1,N_div,temporal_dim,17)
    PCK35_all_array = np.append(PCK35_all_array, PCK35_all_list[-1], axis=0)
    test_loss = {}
    test_loss['MPJPE'] = MPJPE_all_array
    test_loss['PAMPJPE'] = PAMPJPE_all_array
    test_loss['PCC'] = PCC_all_array
    test_loss['PCK'] = PCK_all_array
    test_loss['PCK20'] = PCK25_all_array
    test_loss['PCK25'] = PCK35_all_array
    test_loss['MAE'] = MAE_all_array
    if output_pred:
        y_test_array = np.stack(y_test[:-1]).reshape(-1,N_div,16,17,3)
        y_test_array = np.append(y_test_array,y_test[-1], axis=0)
        y_test_pred_array = np.stack(y_test_pred[:-1]).reshape(-1,N_div,16,17,3)
        y_test_pred_array = np.append(y_test_pred_array,y_test_pred[-1], axis=0)
    if output_pred:
        return test_loss, des_test, (y_test_array, y_test_pred_array)
    else:
        return test_loss, des_test
    
def save_keypoint_plot(y, des, idx, path, args, epoch='gt'):
    """
    Save the estimated 3D keypoint Plot
    """
    # y_sel = y[idx].view(args.train.batch_size//8,-1,19,17,3)[:,:,9,:,:]
    y_sel = y[idx]
    for i in range(len(y_sel)):
        fig = plt.figure()
        ax = plt.axes(projection = '3d')
        if epoch=='gt':
            show3Dpose(y_sel[i], ax, path=os.path.join(path,f'{i}_gt.png'), save=True)
        else:
            show3Dpose(y_sel[i], ax, path=os.path.join(path,f'{i}_epoch{epoch}.png'), save=True)
        plt.close()

def save_confusion_des(y_ls, y_pred_ls, des_ls, labels, name, path_des):
    """Save the confusion matrix and des information
    """
    font = {'size': 10}
    plt.rc('font', **font)
    plt.rcParams['figure.figsize'] = [8, 6]
    cm_display =  metrics.ConfusionMatrixDisplay.from_predictions(
        y_ls, 
        y_pred_ls, 
        values_format = '.3f',
        display_labels= labels,
        normalize='true', 
        include_values=True, 
        cmap=plt.cm.Blues, 
        colorbar=False
        )

    plt.title(name) 
    plt.savefig(path_des+'fig/cm-'+name+'.png')
    plt.close()
    df = pd.concat([
        pd.DataFrame(y_ls, columns=['label']), 
        pd.DataFrame(y_pred_ls, columns=['label_pred']), 
        (des_ls.drop(des_ls.columns[[0]],axis=1)).reset_index(drop=True)
        ], 
        axis = 1
        ).reset_index()
    df.to_csv(
        path_des + 'des/des-'+name+'.csv',
        index=False,
        )  

def save_result_keypoint(model, path):
    """
    Save the model
    """
    model.eval()
    torch.save(model.state_dict(), path)
    return

def print_dict(dict_result, condition_coord):
    for key in dict_result.keys():
        if key=='diff':
            dict_result_sel = dict_result[key][:,:,8]
            if condition_coord=='all':
                dict_result_mask = dict_result_sel
            else:
                condition_coord = np.where(condition_coord!=0., condition_coord, np.nan)
                dict_result_mask = dict_result_sel * condition_coord
            print(np.nanmean(dict_result_mask), np.nanstd(dict_result_mask))  
            print('----------------------')

def analyze_result_2D(summary, test_coord, condition_coord, metric_list, limb_list,
                x_vec = np.linspace(-5,5,25),
                y_vec = np.linspace(5,15,25),
                vx_vec = np.linspace(-1.6,1.6,25),
                vy_vec = np.linspace(-1.6,1.6,25)):
    summary_2D = {}
    for metric in metric_list:
        summary_2D[metric] = {}
        limb_list = summary[metric].keys()
        for key_joint in limb_list:
            summary_2D[metric][key_joint] = {}
            summary_sel = summary[metric][key_joint]['diff']          

            xy_grid = np.zeros((len(y_vec),len(x_vec)),dtype=np.float64)
            numxy_grid = np.zeros((len(y_vec),len(x_vec)),dtype=np.float64) + 1e-8
            vxvy_grid = np.zeros((len(vy_vec),len(vx_vec)),dtype=np.float64)
            numvxvy_grid = np.zeros((len(vy_vec),len(vx_vec)),dtype=np.float64) + 1e-8
            for batch in range(test_coord.shape[0]):
                for n in range(test_coord.shape[1]):
                    x_idx = abs(x_vec-test_coord[batch,n,8,0].mean()).argmin()
                    y_idx = abs(y_vec-test_coord[batch,n,8,1].mean()).argmin()
                    vx_idx = abs(vx_vec-test_coord[batch,n,8,2].mean()).argmin()
                    vy_idx = abs(vy_vec-test_coord[batch,n,8,3].mean()).argmin()
                    if condition_coord[batch,n]==True:
                        xy_grid[y_idx,x_idx] += summary_sel[batch,n,8].mean() + 1e-9
                        vxvy_grid[vy_idx,vx_idx] += summary_sel[batch,n,8].mean() + 1e-9
                        numxy_grid[y_idx,x_idx] += 1.
                        numvxvy_grid[vy_idx,vx_idx] += 1.
            xy_grid = np.where(xy_grid!=0., xy_grid, np.nan)
            vxvy_grid = np.where(vxvy_grid!=0., vxvy_grid, np.nan)
            xy_grid = xy_grid/numxy_grid
            vxvy_grid = vxvy_grid/numvxvy_grid
            summary_2D[metric][key_joint]['xy'] = xy_grid
            summary_2D[metric][key_joint]['vxvy'] = vxvy_grid
    return summary_2D

def analyze_result_1D(summary, coord, condition_coord, metric_list, limb_list, summary_1D_vec, des, xvec=None):
    summary_1D = {}
    summary_1D_xvec = {}
    sample_subject = [sample['subject'] for sample in des]
    subject_list = list(dict.fromkeys(sample_subject))
    subject_list.append('all')
    for key in summary_1D_vec:
        summary_1D[key] = {}
        if key=='ori':
            # key_var = summary['direction']['all']['target'].std(axis=-1)
            key_var = np.abs(np.diff(summary['direction']['all']['target'], axis=-1))[:,:,7]
            # key_var = np.abs(np.diff(summary['direction']['all']['target'], axis=-1)).mean(-1)
        elif key=='rng':
            key_var = (((coord[:,:,:,0]+10) + (coord[:,:,:,1]))/2)[:,:,8]
            # key_var = (((coord[:,:,:,0]+10) + (coord[:,:,:,1]))/2).mean(-1)
        elif key=='x':
            key_var = (coord[:,:,:,0])[:,:,8]
        elif key=='y':
            key_var = (coord[:,:,:,1])[:,:,8]
        elif key=='vel':
            key_var = np.sqrt(np.abs(coord[:,:,:,2])**2 + np.abs(coord[:,:,:,3])**2)[:,:,8]
            # key_var = np.sqrt(np.abs(coord[:,:,:,2])**2 + np.abs(coord[:,:,:,3])**2).mean(-1)
        elif key=='ang':
            theta = np.arctan2(coord[:,:,:,2],coord[:,:,:,3])*180/np.pi
            # theta = np.where(theta<0, theta+360, theta) # 0 ~ 360
            key_var = theta[:,:,8]
        if xvec is not None:
            key_vec = xvec[key]
        else:
            key_vec_min = np.sort(key_var.ravel())[int(len(key_var.ravel())*0.1)]
            key_vec_max = np.sort(key_var.ravel())[int(len(key_var.ravel())*0.9)]
            key_vec = np.linspace(key_vec_min,key_vec_max,9)
        for metric in metric_list:
            summary_1D[key][metric] = {}
            limb_list = summary[metric].keys()
            for key_joint in limb_list:
                summary_1D[key][metric][key_joint] = {}
                summary_sel = summary[metric][key_joint]['diff']
                for subject in subject_list:
                    summary_1D[key][metric][key_joint][subject] = {}
                    # define grid
                    grid_1D = []
                    for idx_keyval in range(len(key_vec)):
                        grid_1D.append([]) 
                    for batch, batch_subject in zip(range(key_var.shape[0]), sample_subject):
                        for n in range(key_var.shape[1]):
                            if subject=='all':
                                idx_1D = abs(key_vec-key_var[batch,n].mean()).argmin()
                                if condition_coord[batch,n]==True:
                                    grid_1D[idx_1D].append(summary_sel[batch,n,8].mean())
                            else:
                                if batch_subject==subject:
                                    idx_1D = abs(key_vec-key_var[batch,n].mean()).argmin()
                                    if condition_coord[batch,n]==True:
                                        grid_1D[idx_1D].append(summary_sel[batch,n,8].mean())
                    grid_1D_mean = np.zeros((len(grid_1D)), dtype=float)
                    grid_1D_std = np.zeros((len(grid_1D)), dtype=float)
                    for idx_keyval in range(len(key_vec)):
                        if len(grid_1D[idx_keyval])==0:
                            grid_1D_mean[idx_keyval] = np.nan
                            grid_1D_std[idx_keyval] = np.nan
                        else:
                            grid_1D_mean[idx_keyval] = np.mean(grid_1D[idx_keyval])
                            grid_1D_std[idx_keyval] = np.std(grid_1D[idx_keyval])
                    summary_1D[key][metric][key_joint][subject]['mean'] = grid_1D_mean
                    summary_1D[key][metric][key_joint][subject]['std'] = grid_1D_std
        summary_1D_xvec[key] = key_vec
    return summary_1D, summary_1D_xvec

def summarize(summary, condition_coord='all'):
  print('----------------------')
  for key in summary.keys():
    dict_sel = summary[key]
    print(key)
    if 'diff' in dict_sel.keys():
      print_dict(dict_sel, condition_coord)
    else:
      for idx_key,key2 in enumerate(dict_sel.keys()):
        if idx_key>0:
          print(key)
        print(key2)
        print_dict(dict_sel[key2], condition_coord)
  return

def summarize_result_2D(summary, save_path=None, visdom=False, save_result=False,
                        x_vec = np.linspace(-6,6,25),
                        y_vec = np.linspace(4,16,25),
                        vx_vec = np.linspace(-1.6,1.6,25),
                        vy_vec = np.linspace(-1.6,1.6,25)):
    if visdom:
        import visdom
        vis = visdom.Visdom()
        for key in summary.keys():
            dict_sel = summary[key]
            for key_joint in dict_sel.keys():
                vis.heatmap(summary[key][key_joint]['xy'],
                        opts=dict(xlabel="x [m]",ylabel="y [m]",title=f'{key}-{key_joint}-xy',
                        ))
                vis.heatmap(summary[key][key_joint]['vxvy'],
                        opts=dict(xlabel="vx [m/s]",ylabel="vy [m/s]",title=f'{key}-{key_joint}-vxvy',
                        ))
    if save_result:
        SMALL_SIZE = 14
        MEDIUM_SIZE = 16
        BIGGER_SIZE = 16

        plt.rc('font', size=SMALL_SIZE) # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE) # fontsize of the axes title
        plt.rc('axes', labelsize=BIGGER_SIZE) # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE) # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE) # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE-2) # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE) # fontsize of the figure title

        X, Y = np.meshgrid(x_vec,y_vec)
        VX, VY = np.meshgrid(vx_vec,vy_vec)
        for key in summary.keys():
            dict_sel = summary[key]
            for key_joint in dict_sel.keys():
                if (key == 'MPJPE') or (key == 'PAMPJPE'):
                    # bar_minmax = (40,180)       # subject_normal
                    summary[key][key_joint]['xy'] = copy.deepcopy(summary[key][key_joint]['xy']*1000)
                    summary[key][key_joint]['vxvy'] = copy.deepcopy(summary[key][key_joint]['vxvy']*1000)
                # X-Y            
                plt.pcolormesh(X,Y,summary[key][key_joint]['xy'], cmap='jet')
                if key == 'MPJPE':
                    mean_val = np.nanmean(summary[key][key_joint]['xy'])
                    plt.gca().collections[0].set_clim(mean_val-30,mean_val+80)
                    plt.gca().collections[0].set_clim(50,160)
                if key == 'PCC':
                    plt.gca().collections[0].set_clim(0.35,0.65)
                clb = plt.colorbar()
                if key == 'MPJPE':
                    clb.set_label('MPJPE [mm]')
                # plt.title(f'{key}')
                plt.xlabel('x [m]')
                plt.ylabel('y [m]')
                plt.xticks(np.linspace(-5,5,9), rotation=45)
                plt.yticks(np.linspace(5,15,9))
                plt.show()
                plt.savefig(f'{save_path}'+f'{key}-{key_joint}-xy.jpg', dpi=1200, bbox_inches='tight')
                plt.close()
                # VX-VY
                plt.pcolormesh(VX,VY,summary[key][key_joint]['vxvy'], cmap='jet')
                if key == 'MPJPE':
                    mean_val = np.nanmean(summary[key][key_joint]['vxvy'])
                    plt.gca().collections[0].set_clim(mean_val-30,mean_val+80)
                    plt.gca().collections[0].set_clim(50,160)
                if key == 'PCC':
                    plt.gca().collections[0].set_clim(0.4,0.7)
                clb = plt.colorbar()
                if key == 'MPJPE':
                    clb.set_label('MPJPE [mm]')
                # plt.title(f'{key}')
                plt.xlabel('vx [m/s]')
                plt.ylabel('vy [m/s]')
                plt.xticks(np.linspace(-1.6,1.6,9), rotation=45)
                plt.yticks(np.linspace(-1.6,1.6,9))
                plt.show()
                plt.savefig(f'{save_path}'+f'{key}-{key_joint}-vxvy.jpg', dpi=1200, bbox_inches='tight')
                plt.close()
    return

def summarize_result_1D(summary, summary_vec, save_path=None, visdom=False, save_result=False):
    if visdom:
        import visdom
        vis = visdom.Visdom()
        for key in summary.keys():
            dict_sel = summary[key]
            for key_joint in dict_sel.keys():
                vis.line(X=summary_vec[key],Y=summary[key][key_joint],
                        opts=dict(xlabel="Average Absolute Angular Vel. of Body [Deg/s]",title=f'{key}-{key_joint}-ori',
                        ))
    if save_result:
        SMALL_SIZE = 20
        MEDIUM_SIZE = 24
        BIGGER_SIZE = 24

        plt.rc('font', size=SMALL_SIZE) # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE) # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE) # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE) # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE) # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE-1) # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE) # fontsize of the figure title
        sns.set_palette('muted')
        x_label_list = {
            'ori': 'Average Absolute Angular Vel. of Body [Deg/s]',
            'rng': 'Average Distance [m]',
            'vel': 'Average Absolute Velocity [m/s]',
            'ang': 'Orientation [Deg]',
            'x': 'x [m]',
            'y': 'y [m]'
        }
        y_label_list = {
            'MPJPE': 'MPJPE [mm]',
            'PAMPJPE': 'PA-MPJPE [mm]',
            'PCK': 'Acc [%]',
            'PCK20': 'Acc [%]',
            'PCK25': 'Acc [%]',
            'PCC': r'$\rho$',
            'direction': 'Error [Deg]',
            'freq': 'Error [Hz]',
        }
        for key_1D in summary.keys():
            dict_sel = summary[key_1D]
            vec_1D = summary_vec[key_1D]
            x_label = x_label_list[key_1D]
            for metric in dict_sel.keys():
                y_label = y_label_list[metric]
                dict_sel_metric = summary[key_1D][metric]
                for key_joint in dict_sel_metric.keys():
                    ###### merge right and left joint #######
                    if key_joint=='r_hand' or key_joint=='r_foot':
                        val_1D_save = copy.deepcopy(summary[key_1D][metric][key_joint])
                        continue
                    if key_joint=='l_hand' or key_joint=='l_foot':
                        for subject in val_1D_save.keys():
                            val_1D_save[subject]['mean'] = (val_1D_save[subject]['mean']+summary[key_1D][metric][key_joint][subject]['mean'])/2
                            val_1D_save[subject]['std'] = (val_1D_save[subject]['std']+summary[key_1D][metric][key_joint][subject]['std'])/2
                        val_1D = val_1D_save
                    #####################################
                    else:
                        val_1D = summary[key_1D][metric][key_joint]
                    ### plot for multi_subject + avg
                    plt.figure(figsize=(10,4))
                    for subject in val_1D.keys():
                        if subject=='all':
                            plt.plot(vec_1D,val_1D[subject]['mean'],
                                        color='black',linestyle='--',linewidth=3)
                        else:
                            plt.plot(vec_1D,val_1D[subject]['mean'])
                    # plt.title(f'Metric: {metric}')
                    plt.legend(list(val_1D.keys()), loc='upper right')
                    plt.grid(True)
                    plt.xlabel(x_label)
                    plt.ylabel(y_label)
                    plt.show()
                    plt.savefig(f'{save_path}'+f'{key_1D}-{metric}-{key_joint}-1D.jpg', dpi=600, bbox_inches='tight')
                    plt.close()
                    ### plot for avg + std(error bound)
                    if key_1D=='ang':
                        x = np.linspace(-180,180,len(vec_1D))
                        y = copy.deepcopy(val_1D['all']['mean'])
                        error = copy.deepcopy(val_1D['all']['std'])
                        # y = np.concatenate((val_1D['all']['mean'][3*(len(vec_1D)//4)+1:],val_1D['all']['mean'][:3*(len(vec_1D)//4)+1]))     
                        # error = np.concatenate((val_1D['all']['std'][3*(len(vec_1D)//4)+1:],val_1D['all']['std'][:3*(len(vec_1D)//4)+1]))
                        # if metric =='MPJPE' and key_joint == 'legs':
                        #     y_label = 'Error [Hz]'
                    else:
                        x = copy.deepcopy(vec_1D)
                        y = copy.deepcopy(val_1D['all']['mean'])
                        error = copy.deepcopy(val_1D['all']['std'])
                    plt.figure(figsize=(10,4))
                    plt.plot(x,y, color='black',linewidth=3)
                    error = val_1D['all']['std']
                    plt.fill_between(x, y-error, y+error, alpha=0.3)
                    # plt.title(f'Metric: {metric}')
                    plt.grid(True)
                    plt.xlabel(x_label)
                    plt.ylabel(y_label)
                    if 'PCK' in metric:
                        plt.ylim([70,100])
                    plt.show()
                    plt.savefig(f'{save_path}'+f'{key_1D}-{metric}-{key_joint}-1D_error.jpg', dpi=600, bbox_inches='tight')
                    plt.close()

    return

def summarize_result_1D_all(summary, summary_vec, plot_list, save_path=None, save_result=False):
    if save_result:
        SMALL_SIZE = 20
        MEDIUM_SIZE = 24
        BIGGER_SIZE = 24

        plt.rc('font', size=SMALL_SIZE) # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE) # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE) # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE) # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE) # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE-2) # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE) # fontsize of the figure title
        sns.set_palette('muted')
        x_label_list = {
            'ori': 'Average Absolute Angular Vel. of Body [Deg/s]',
            'rng': 'Average Distance [m]',
            'vel': 'Average Absolute Velocity [m/s]',
            'ang': 'Walking Angle Relative to Sensor [Deg]',
            'x': 'x [m]',
            'y': 'Distance [m]'
        }
        y_label_list = {
            'MPJPE': 'MPJPE [mm]',
            'PAMPJPE': 'PA-MPJPE [mm]',
            'PCK': 'Acc [%]',
            'PCK20': 'Acc [%]',
            'PCK25': 'Acc [%]',
            'PCC': r'$\rho$',
            'direction': 'Error [Deg]',
            'freq': 'Error [Hz]',
        }
        
        summary_list = list(summary.keys())
        for key_1D in summary[summary_list[0]].keys():
            dict_sel = summary[summary_list[0]][key_1D]
            vec_1D = summary_vec[summary_list[0]][key_1D]
            x_label = x_label_list[key_1D]
            for metric in dict_sel.keys():
                y_label = y_label_list[metric]
                dict_sel_metric = dict_sel[metric]
                for key_joint in dict_sel_metric.keys():
                    val_1D_dict = {}
                    if key_joint=='r_hand' or key_joint=='r_foot' or key_joint=='l_foot':
                        continue
                    for class_name in summary_list:
                        ###### merge right and left joint #######
                        if key_joint=='l_hand':
                            val_1D = {}
                            for subject in summary[class_name][key_1D][metric][key_joint].keys():
                                val_1D[subject] = {}
                                val_1D[subject]['mean'] = (summary[class_name][key_1D][metric]['l_hand'][subject]['mean']+
                                                            summary[class_name][key_1D][metric]['r_hand'][subject]['mean']+
                                                            summary[class_name][key_1D][metric]['l_foot'][subject]['mean']+
                                                            summary[class_name][key_1D][metric]['r_foot'][subject]['mean'])/4
                                val_1D[subject]['std'] = (summary[class_name][key_1D][metric]['l_hand'][subject]['std']+
                                                            summary[class_name][key_1D][metric]['r_hand'][subject]['std']+
                                                            summary[class_name][key_1D][metric]['l_foot'][subject]['std']+
                                                            summary[class_name][key_1D][metric]['r_foot'][subject]['std'])/4
                        #####################################
                        else:
                            val_1D = summary[class_name][key_1D][metric][key_joint]
                        val_1D_dict[class_name] = val_1D
                    ### 99 ######
                    if metric=='MPJPE':
                        if key_joint=='all':
                            val_temp = copy.deepcopy(val_1D_dict)
                    ##############
                    fig, ax = plt.subplots(figsize=(10,4))
                    # plt.figure(figsize=(10,4))
                    list_legend = []
                    for model_name, plot_style in zip(summary_list, plot_list):
                        train_model, test_class, standard = model_name.split(':')
                        if 'single' in train_model:
                            label = 'mmWave-Single'
                        elif 'video' in train_model:
                            label = 'Vision'
                        else:
                            label = 'mmWave-Multi'
                        val_1D_model = val_1D_dict[model_name]['all']['mean']
                        val_1D_model_std = val_1D_dict[model_name]['all']['mean']
                        if (metric=='MPJPE') or (metric=='PAMPJPE'):
                            val_1D_model = copy.deepcopy(val_1D_model*1000)
                            val_1D_model_std = copy.deepcopy(val_1D_model_std*1000)
                        if key_1D=='y':
                            vec_1D = copy.deepcopy(np.linspace(5,15,len(vec_1D)))
                        if key_1D=='ang':
                            vec_1D = copy.deepcopy(np.linspace(-180,180,len(vec_1D)))
                        ax.plot(vec_1D,val_1D_model, color=plot_style[0], linestyle=plot_style[1], linewidth=3, label=label)
                        # ax.fill_between(vec_1D, val_1D_model-val_1D_model_std, val_1D_model+val_1D_model_std, color=plot_style[0], alpha=0.1, label=label)
                        list_legend.append(label)
                    # plt.title(f'Metric: {metric}')
                    handler, labeler = ax.get_legend_handles_labels()
                    hd = [(handler[0],handler[0]), (handler[1],handler[1])]
                    # hd = [(handler[0],handler[0]), (handler[1],handler[1]), (handler[2],handler[2])]
                    # hd = [(handler[0],handler[1]), (handler[2],handler[3]), (handler[4],handler[5])]
                    ax.legend(hd, list_legend, loc="upper left")
                    # plt.legend(list_legend, loc='upper center')
                    plt.grid(True)
                    plt.xlabel(x_label)
                    plt.ylabel(y_label)
                    # if metric=='MPJPE':
                    #     plt.ylim([0, 180])
                    if key_1D=='y':
                        plt.xlim([5, 15])
                    elif key_1D=='ang':
                        plt.xlim([-180, 180])
                    if metric=='MPJPE':
                        plt.ylim([40,160])
                    elif metric=='PCC':
                        plt.ylim([0.1,0.6])
                    plt.show()
                    plt.savefig(f'{save_path}'+f'{key_1D}-{metric}-{key_joint}-1D.jpg', dpi=600, bbox_inches='tight')
                    plt.close()   
    return

def summarize_result_bar_all(summary, save_path=None, save_result=False):
    if save_result:
        SMALL_SIZE = 20
        MEDIUM_SIZE = 22
        BIGGER_SIZE = 24

        plt.rc('font', size=SMALL_SIZE) # controls default text sizes
        plt.rc('axes', titlesize=SMALL_SIZE) # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE) # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE) # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE) # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE-2) # legend fontsize
        plt.rc('figure', titlesize=BIGGER_SIZE) # fontsize of the figure title
        sns.set_palette('muted')
        
        class_list = list(summary.keys())
        for metric in summary[class_list[0]].keys():
            dict_sel = summary[class_list[0]][metric]
            for key_joint in dict_sel.keys():
                val_1D_dict = {}
                if key_joint=='r_hand' or key_joint=='r_foot':
                    continue
                for class_name in class_list:
                    ###### merge right and left joint #######
                    if key_joint=='l_hand' or key_joint=='l_foot':
                        val_1D = (summary[class_name][metric][('_').join(['r',key_joint.split('_')[-1]])]['diff']+
                                            summary[class_name][metric][key_joint]['diff'])/2
                    #####################################
                    else:
                        val_1D = summary[class_name][metric][key_joint]['diff']
                    val_1D_dict[class_name] = val_1D
                    
                # Bar plot
                data_df = pd.DataFrame(index={'two_hand','one_hand','no_hand'},columns={'train with all', 'train with each class'}).sort_index()
                data_df_err = pd.DataFrame(index={'two_hand','one_hand','no_hand'},columns={'train with all', 'train with each class'}).sort_index()
                for class_name in class_list:
                    train_model, test_class = class_name.split(':')
                    if 'baseline' in train_model:
                        legend = 'train with all'
                    else:
                        legend = 'train with each class'
                    if 'normal' in test_class:
                        x_key = 'two_hand'
                    elif 'phone' in test_class:
                        x_key = 'one_hand'
                    elif ('texting' in test_class) or ('pocket' in test_class):
                        x_key = 'no_hand'
                    data_df.loc[x_key,legend] = val_1D_dict[class_name].mean()
                    data_df_err.loc[x_key,legend] = val_1D_dict[class_name].std()
                plt.rcParams['figure.figsize'] = [10, 6]
                # ax = data_df.plot(kind='bar',yerr=data_df_err, cmap=plt.cm.get_cmap('magma',2), alpha=0.85, edgecolor='black', linewidth=0.7, zorder=2)
                ax = data_df.plot(kind='bar', cmap=plt.cm.get_cmap('magma',2), alpha=0.85, edgecolor='black', linewidth=1.5, zorder=2)
                plt.xticks(rotation=0)
                plt.grid(True, axis='y', zorder=1)
                plt.legend(loc='upper right')
                plt.xlabel('Class')
                if metric == 'freq':
                    # plt.ylim([0, 0.15])     # normal
                    plt.ylim([0, 0.12])     # texting
                    plt.ylabel('Error [Hz]')
                elif metric == 'direction':
                    # plt.ylim([0, 30])       # normal
                    plt.ylim([0, 25])       # texting
                    plt.ylabel('Error [Deg]')
                elif metric == 'MPJPE':
                    plt.ylabel('Error [mm]')
                plt.imshow
                plt.savefig(f'{save_path}'+f'bar-{metric}-{key_joint}.jpg', dpi=600, bbox_inches='tight')
                plt.close()
    return