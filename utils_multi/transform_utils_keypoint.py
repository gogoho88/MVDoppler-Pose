import numpy as np
import pandas as pd
import copy
import cv2
import torch
import torch.nn.functional as F
import torchvision
from .preprocessing_utils import OneEuroFilter
from .camera import *

### Radar data Transforms

class ToCHWTensor(object):
    """Convert numpy array to CHW tensor format.
    """
    def __init__(self, apply=['radar', 'keypoint']):
        self.radar = True if 'radar' in apply else False
        self.keypoint = True if 'keypoint' in apply else False

    def __call__(self, dat):
        if self.radar:
            dat['radar'] = torch.from_numpy(dat['radar'].transpose(2,0,1))
            dat['radar_rng'] = torch.from_numpy(dat['radar_rng'].transpose(2,1,0))
        if self.keypoint:
            dat['keypoint'] = torch.from_numpy(dat['keypoint'])
        return dat

class RandomizeCrop_Time(object):
    """
    Randomly select starting time idx of radar&keypoint and crop it to N-sec. size
    """
    def __init__(self, win_sec, total_sec=10., start_sec=0.5, end_sec=0.5, radar_len=1985, radar_rng_len=250, keypoint_len=300, apply=['radar', 'keypoint']):
        self.win_sec = win_sec
        self.total_sec = total_sec
        self.radar = True if 'radar' in apply else False
        self.radar_pcl = True if 'radar_pcl' in apply else False
        self.keypoint = True if 'keypoint' in apply else False
        self.start_sec = start_sec
        self.end_sec = total_sec-(end_sec+win_sec)
        self.radar_len = radar_len
        self.radar_rng_len = radar_rng_len
        self.keypoint_len = keypoint_len
        self.win_radar_len = round(radar_len*(win_sec/total_sec))
        self.win_radar_rng_len = round(radar_rng_len*(win_sec/total_sec))
        self.win_keypoint_len = round(keypoint_len*(win_sec/total_sec))
    
    def __call__(self, dat):
        start_sec_select = torch.rand(1).item()*(self.end_sec-self.start_sec) + self.start_sec     # rand in [self.start.sec, self.end_sec)
        start_radar = round((start_sec_select/self.total_sec)*self.radar_len)
        start_radar_rng = round((start_sec_select/self.total_sec)*self.radar_rng_len)
        start_keypoint = round((start_sec_select/self.total_sec)*self.keypoint_len)
        start_keypoint = start_keypoint + torch.randint((dat['des']['len_keypoint_3D']-self.keypoint_len)+1,(1,)).item()    # adjust since len_keypoint is not exactly 300
        
        if self.radar:
            dat['radar'] = dat['radar'][:,:,start_radar:start_radar+self.win_radar_len]
            dat['radar_rng'] = dat['radar_rng'][:,:,start_radar_rng:start_radar_rng+self.win_radar_rng_len]
        if self.radar_pcl:
            dat['radar_pcl'] = torch.tensor(dat['radar_pcl'].iloc[start_keypoint:start_keypoint+self.win_keypoint_len][['x','y','vx','vy']].values)
        if self.keypoint:
            dat['keypoint'] = dat['keypoint'][start_keypoint:start_keypoint+self.win_keypoint_len,:,:]
        return dat
    
class UniformCrop_Time(object):
    """
    Uniformly select starting time idx of radar&keypoint and crop it to N-sec. size, Divide, and Merge Them
    """
    def __init__(self, win_sec, total_sec=10., start_sec=0.5, end_sec=0.5, radar_len=1985, radar_rng_len=250, keypoint_len=300, n_div='all', apply=['radar', 'keypoint']):
        self.win_sec = win_sec
        self.total_sec = total_sec
        self.radar = True if 'radar' in apply else False
        self.radar_pcl = True if 'radar_pcl' in apply else False
        self.keypoint = True if 'keypoint' in apply else False
        self.start_sec = start_sec
        self.end_sec = total_sec-(end_sec+win_sec)
        self.radar_len = radar_len
        self.radar_rng_len = radar_rng_len
        self.keypoint_len = keypoint_len
        self.win_radar_len = round(radar_len*(win_sec/total_sec))
        self.win_radar_rng_len = round(radar_rng_len*(win_sec/total_sec))
        self.win_keypoint_len = round(keypoint_len*(win_sec/total_sec))
        if n_div=='all':
            self.n_div = int((self.end_sec-self.start_sec)*(self.keypoint_len/self.total_sec))
        else:
            self.n_div=n_div
    
    def __call__(self, dat):
        start_sec_list = torch.linspace(self.start_sec, self.end_sec, self.n_div)
        dat_radar_merge = []
        dat_radar_rng_merge = []
        dat_radarpcl_merge = []
        dat_keypoint_merge = []
        list_keypoint_start = []
        for start_sec_select in start_sec_list.tolist():
            # select start point
            start_radar = round((start_sec_select/self.total_sec)*self.radar_len)
            start_radar_rng = round((start_sec_select/self.total_sec)*self.radar_rng_len)
            start_keypoint = round((start_sec_select/self.total_sec)*self.keypoint_len)
            start_keypoint = start_keypoint + (dat['des']['len_keypoint_3D']-self.keypoint_len)
            # windowing
            dat_radar = dat['radar'][:,:,start_radar:start_radar+self.win_radar_len]
            dat_radar_rng = dat['radar_rng'][:,:,start_radar_rng:start_radar_rng+self.win_radar_rng_len]
            dat_radarpcl = torch.tensor(dat['radar_pcl'].iloc[start_keypoint:start_keypoint+self.win_keypoint_len][['x','y','vx','vy']].values)
            dat_keypoint = dat['keypoint'][start_keypoint:start_keypoint+self.win_keypoint_len,:,:]
            dat_radar_merge.append(dat_radar)
            dat_radar_rng_merge.append(dat_radar_rng)
            dat_radarpcl_merge.append(dat_radarpcl)        
            dat_keypoint_merge.append(dat_keypoint)
            list_keypoint_start.append(start_keypoint)
        dat_radar_merge = torch.stack(dat_radar_merge)
        dat_radar_rng_merge = torch.stack(dat_radar_rng_merge)
        dat_radarpcl_merge = torch.stack(dat_radarpcl_merge)
        dat_keypoint_merge = torch.stack(dat_keypoint_merge)
        if self.radar:
            dat['radar'] = dat_radar_merge
            dat['radar_rng'] = dat_radar_rng_merge
        if self.radar_pcl:
            dat['radar_pcl'] = dat_radarpcl_merge
        if self.keypoint:
            dat['keypoint'] = dat_keypoint_merge
            dat['keypoint_startpoint'] = torch.tensor(list_keypoint_start)
        return dat

class RandFlip(object):
    """
    Randomly flip the mD image in time or frequency dim.
    """
    def __init__(self, p=0.5, apply=['radar', 'keypoint']):
        self.p = p
        self.radar = True if 'radar' in apply else False
        self.radar_pcl = True if 'radar_pcl' in apply else False
        self.keypoint = True if 'keypoint' in apply else False
    def __call__(self, dat): 
        dat_radar = dat['radar']
        dat_radar_rng = dat['radar_rng']
        dat_radar_pcl = dat['radar_pcl']
        keypoint = dat['keypoint']
        p_freq, p_time = torch.rand(2)
        if p_time.item() <self.p:
            if self.radar:
                dat_radar = dat_radar.flip(dims=(2,))
                dat_radar_rng = dat_radar_rng.flip(dims=(2,))
            if self.radar_pcl:
                dat_radar_pcl = dat_radar_pcl.flip(dims=(0,))
            if self.keypoint:
                keypoint = keypoint.flip(dims=(0,))
        dat['radar'] = dat_radar
        dat['radar_rng'] = dat_radar_rng
        dat['radar_pcl'] = dat_radar_pcl
        dat['keypoint'] = keypoint     
        return dat
        

class NormalizeKeypoint(object):
    """
    - Normlize all the keypoint such that z-length of each keypoint becomes 1.
    """
    def __init__(self):
        pass
    def __call__(self, dat):
        keypoint = dat['keypoint']
        y_feet1 = keypoint[:,3,1]
        y_feet2 = keypoint[:,6,1]
        y_feet = (y_feet1+y_feet2)/2
        y_head = keypoint[:,10,1]
        normalize_factor = (y_feet-y_head).repeat((17,3,1)).permute(2,0,1)
        keypoint_normalize = keypoint/normalize_factor

        dat['keypoint'] = keypoint_normalize
        return dat
    
class NormalizeRadarPcl(object):
    """
    - Normlize radar pcl data to have the magnitude of 1 (only for vx, vy).
    """
    def __init__(self):
        pass
    def __call__(self, dat):
        radar_pcl = dat['radar_pcl']
        x = radar_pcl['x'].to_numpy()
        y = radar_pcl['y'].to_numpy()
        vx = radar_pcl['vx'].to_numpy()
        vy = radar_pcl['vy'].to_numpy()
        v = np.sqrt(vx**2 + vy**2)
        # Normalize
        vx = vx/v   
        vy = vy/v

        radar_pcl['vx'] = vx
        radar_pcl['vy'] = vy

        dat['radar_pcl'] = radar_pcl
        return dat

class ResizeRadarPcl(object):
    """
    Resize the length of Radar Pointcloud data to be matched with the length of Keypoint data
    """
    def __init__(self):
        pass
    def __call__(self, dat):
        radar_pcl = dat['radar_pcl']
        keypoint = dat['keypoint']
        ori_len = radar_pcl.shape[0]
        resize_len = len(keypoint)
        x_ori = np.linspace(0, ori_len-1, ori_len)
        x_new = np.linspace(0, ori_len-1, resize_len)
        radar_pcl_new = pd.DataFrame(0, index=np.arange(resize_len), columns=radar_pcl.columns)
        for col in radar_pcl.columns:
            radar_pcl_sel = radar_pcl[col].to_numpy()
            radar_pcl_interp = np.interp(x_new, x_ori, radar_pcl_sel)
            radar_pcl_new[col] = radar_pcl_interp
        dat['radar_pcl'] = radar_pcl_new
        return dat

class ResizeTheta(object):
    def __init__(self, len=19, flag_train=True):
        self.flag_train=flag_train
        self.len = len
    def __call__(self, dat):
        radar_pcl = dat['radar_pcl']
        if self.flag_train:
            ori_len = radar_pcl.shape[0]
            theta_new = np.zeros(self.len)
            x_ori = np.linspace(0, ori_len-1, ori_len)
            x_new = np.linspace(0, ori_len-1, self.len)
            theta_new = np.interp(x_new, x_ori, radar_pcl)
        else:
            theta_new = np.zeros((radar_pcl.shape[0],self.len))
            ori_len = radar_pcl.shape[1]
            x_ori = np.linspace(0, ori_len-1, ori_len)
            x_new = np.linspace(0, ori_len-1, self.len)
            for frame in range(radar_pcl.shape[0]):
                theta_sel = radar_pcl[frame,:]
                theta_interp = np.interp(x_new, x_ori, theta_sel)
                theta_new[frame,:] = theta_interp
        dat['radar_pcl'] = torch.tensor(theta_new)
        return dat

class ResizeKeypoint(object):
    def __init__(self, len=19, flag_train=True):
        self.flag_train=flag_train
        self.len = len
    def __call__(self, dat):
        keypoint = np.array(dat['keypoint'])
        if self.flag_train:
            keypoint_new = np.zeros((self.len, 17, 3))
            t_ori = np.linspace(0,len(keypoint)-1,len(keypoint))
            t_new = np.linspace(0,len(keypoint)-1,self.len)
            for body in range(17):
                for coord in range(3):
                    keypoint_sel = keypoint[:,body,coord]
                    keypoint_interp = np.interp(t_new, t_ori, keypoint_sel)
                    keypoint_new[:,body,coord] = keypoint_interp
        else:
            keypoint_new = np.zeros((keypoint.shape[0],self.len, 17, 3))
            t_ori = np.linspace(0,keypoint.shape[1]-1,keypoint.shape[1])
            t_new = np.linspace(0,keypoint.shape[1]-1,self.len)
            for frame in range(keypoint.shape[0]):
                for body in range(17):
                    for coord in range(3):
                        keypoint_sel = keypoint[frame,:,body,coord]
                        keypoint_interp = np.interp(t_new, t_ori, keypoint_sel)
                        keypoint_new[frame,:,body,coord] = keypoint_interp
        dat['keypoint'] = torch.tensor(keypoint_new)
        return dat

class ResizeRadar(object):
    def __init__(self, size_mD, size_rng, flag_train=True):
        self.size_mD = size_mD
        self.size_rng = size_rng
        self.flag_train = flag_train
    def __call__(self, dat):
        radar_dat = dat['radar']
        radar_dat_rng = dat['radar_rng']
        if self.flag_train:
            radar_dat = F.interpolate(radar_dat.unsqueeze(dim=0), self.size_mD).squeeze(dim=0)
            radar_dat_rng = F.interpolate(radar_dat_rng.unsqueeze(dim=0), self.size_rng).squeeze(dim=0)
        else:
            radar_dat = F.interpolate(radar_dat, self.size_mD)
            radar_dat_rng = F.interpolate(radar_dat_rng, self.size_rng)
        dat['radar'] = radar_dat
        dat['radar_rng'] = radar_dat_rng
        return dat

class LPFpoint(object):
    """
    Apply temporal LPF(1eurofilter to keypoint)

    Args:
    - beta: lower beta leads to more smoothing
    """
    def __init__(self, min_cutoff=0.006, beta=0.6, apply=['radar_pcl', 'keypoint']):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.radar_pcl = True if 'radar_pcl' in apply else False
        self.keypoint = True if 'keypoint' in apply else False
    def __call__(self, dat):
        if self.radar_pcl:
            radar_pcl = dat['radar_pcl']
            x = radar_pcl['x'].to_numpy()
            y = radar_pcl['y'].to_numpy()
            vx = radar_pcl['vx'].to_numpy()
            vy = radar_pcl['vy'].to_numpy()

            if np.isnan(x).sum()+np.isnan(y).sum()+np.isnan(vx).sum()+np.isnan(vy).sum()>0:
                idx_nan = np.where(np.isnan(vx)==True)[0]
                x[idx_nan] = x[idx_nan[0]-1]
                y[idx_nan] = y[idx_nan[0]-1]
                vx[idx_nan] = vx[idx_nan[0]-1]
                vy[idx_nan] = vy[idx_nan[0]-1]
            
            theta = np.arctan2(vy, vx)

            x_lpf = torch.zeros(x.shape, dtype=torch.float64)
            y_lpf = torch.zeros(y.shape, dtype=torch.float64)
            vx_lpf = torch.zeros(vx.shape, dtype=torch.float64)
            vy_lpf = torch.zeros(vy.shape, dtype=torch.float64)
            theta_lpf = torch.zeros(theta.shape, dtype=torch.float64)
            x_ref = copy.deepcopy(x)
            y_ref = copy.deepcopy(y)
            vx_ref = copy.deepcopy(vx)
            vy_ref = copy.deepcopy(vy)
            theta_ref = copy.deepcopy(theta)

            for j in range(len(x)):
                curr_x = x_ref[j]
                curr_y = y_ref[j]
                curr_vx = vx_ref[j]
                curr_vy = vy_ref[j]
                curr_theta = theta_ref[j]
                if j==0:
                    x_track = OneEuroFilter(j, curr_x, min_cutoff=self.min_cutoff, beta=self.beta)
                    y_track = OneEuroFilter(j, curr_y, min_cutoff=self.min_cutoff, beta=self.beta)
                    vx_track = OneEuroFilter(j, curr_vx, min_cutoff=self.min_cutoff, beta=self.beta)
                    vy_track = OneEuroFilter(j, curr_vy, min_cutoff=self.min_cutoff, beta=self.beta)
                    theta_track = OneEuroFilter(j, curr_theta, min_cutoff=self.min_cutoff, beta=self.beta*0.05)
                if j > 1:
                    curr_x = x_track(j, curr_x)
                    curr_y = y_track(j, curr_y)
                    curr_vx = vx_track(j, curr_vx)
                    curr_vy = vy_track(j, curr_vy)
                    curr_theta = theta_track(j, curr_theta)
                x_lpf[j] = curr_x
                y_lpf[j] = curr_y
                vx_lpf[j] = curr_vx
                vy_lpf[j] = curr_vy
                theta_lpf[j] = curr_theta
            radar_pcl['x'] = x_lpf
            radar_pcl['y'] = y_lpf
            radar_pcl['vx'] = vx_lpf
            radar_pcl['vy'] = vy_lpf
            radar_pcl['theta'] = theta
            dat['radar_pcl'] = radar_pcl

        if self.keypoint:
            keypoint = dat['keypoint']
            keypoint_lpf = torch.zeros(keypoint.size(), dtype=torch.float64)
            keypoint_ref = copy.deepcopy(keypoint)
            for j in range(len(keypoint)):
                curr_kp = keypoint_ref[j]
                if j == 0:
                    x_track = [OneEuroFilter(j, curr_kp[k][0], min_cutoff=self.min_cutoff, beta=self.beta) for k in range(17)]  # track for all keypoints
                    y_track = [OneEuroFilter(j, curr_kp[k][1], min_cutoff=self.min_cutoff, beta=self.beta) for k in range(17)]
                    z_track = [OneEuroFilter(j, curr_kp[k][2], min_cutoff=self.min_cutoff, beta=self.beta) for k in range(17)]
                if j > 1:
                    for i in range(17):
                        curr_kp[i][0] = x_track[i](j, curr_kp[i][0])
                        curr_kp[i][1] = y_track[i](j, curr_kp[i][1])
                        curr_kp[i][2] = z_track[i](j, curr_kp[i][2])
                keypoint_lpf[j] = curr_kp
            dat['keypoint'] = keypoint_lpf
        return dat

class Keypoint_to_Global(object):
    """
    Map keypoint to global coordinate

    Args:
    - rot: rotation matirx
    """
    def __init__(self, rot = [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]):
        self.rot = rot
        self.rot = np.array(self.rot, dtype='float64')
    def __call__(self, dat):
        keypoint = dat['keypoint']
        vals_pre = np.array(keypoint, dtype='float64')
        vals = camera_to_world(vals_pre, R=self.rot, t=0)
        dat['keypoint'] = torch.from_numpy(vals)
        return dat

class RotateKeypoint(object):
    """
    Rotate keypoint to have no direction changes

    Args:
    """
    def __init__(self):
        pass
    def __call__(self, dat):
        keypoint = dat['keypoint']
        radar_pcl = dat['radar_pcl']
        keypoint = np.array(keypoint)
        keypoint_new = torch.zeros(keypoint.shape, dtype=torch.float64)

        theta = radar_pcl['theta'].to_numpy()
        theta = theta + np.pi/2
        theta_deg = theta * 180/np.pi

        R_inv = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])    # inverse rotation matrix

        keypoint_rotate_all = R_inv.transpose(2,0,1) @ np.array([keypoint[:,:,0],keypoint[:,:,2]]).transpose(1,0,2)
        keypoint_rotate_all = torch.tensor(keypoint_rotate_all.transpose(0,2,1))

        keypoint_new[:,:,(0,2)] = keypoint_rotate_all
        keypoint_new[:,:,1] = torch.tensor(keypoint[:,:,1])

        dat['keypoint_rot'] = keypoint_new
        return dat
        

class NormalizeRadar(object):
    """
    Apply z-normalization
    """
    def __init__(self, mean_mD, std_mD, mean_rng, std_rng, flag_train=True):
        self.mean_mD = mean_mD
        self.std_mD = std_mD
        self.mean_rng = mean_rng
        self.std_rng = std_rng
        self.flag_train = flag_train
    def __call__(self, dat):
        radar_dat = dat['radar']
        radar_dat_rng = dat['radar_rng']
        if self.flag_train:
            radar_dat[0] = (radar_dat[0]-self.mean_mD[0])/self.std_mD[0]
            radar_dat[1] = (radar_dat[1]-self.mean_mD[1])/self.std_mD[1]
            radar_dat_rng[0] = (radar_dat_rng[0]-self.mean_rng[0])/self.std_rng[0]
            radar_dat_rng[1] = (radar_dat_rng[1]-self.mean_rng[1])/self.std_rng[1]
        else:
            radar_dat[:,0,:,:] = (radar_dat[:,0,:,:]-self.mean_mD[0])/self.std_mD[0]
            radar_dat[:,1,:,:] = (radar_dat[:,1,:,:]-self.mean_mD[1])/self.std_mD[1]
            radar_dat_rng[:,0,:,:] = (radar_dat_rng[:,0,:,:]-self.mean_rng[0])/self.std_rng[0]
            radar_dat_rng[:,1,:,:] = (radar_dat_rng[:,1,:,:]-self.mean_rng[1])/self.std_rng[1]
        dat['radar'] = radar_dat
        dat['radar_rng'] = radar_dat_rng
        return dat