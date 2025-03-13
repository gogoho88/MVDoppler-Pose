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
            dat_keypoint = dat['keypoint'][start_keypoint:start_keypoint+self.win_keypoint_len,:,:]
            dat_radar_merge.append(dat_radar)
            dat_radar_rng_merge.append(dat_radar_rng)
            dat_keypoint_merge.append(dat_keypoint)
            list_keypoint_start.append(start_keypoint)
        dat_radar_merge = torch.stack(dat_radar_merge)
        dat_radar_rng_merge = torch.stack(dat_radar_rng_merge)
        dat_keypoint_merge = torch.stack(dat_keypoint_merge)
        if self.radar:
            dat['radar'] = dat_radar_merge
            dat['radar_rng'] = dat_radar_rng_merge
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
        self.keypoint = True if 'keypoint' in apply else False
    def __call__(self, dat): 
        dat_radar = dat['radar']
        dat_radar_rng = dat['radar_rng']
        keypoint = dat['keypoint']
        p_freq, p_time = torch.rand(2)
        if p_time.item() <self.p:
            if self.radar:
                dat_radar = dat_radar.flip(dims=(2,))
                dat_radar_rng = dat_radar_rng.flip(dims=(2,))
            if self.keypoint:
                keypoint = keypoint.flip(dims=(0,))
        dat['radar'] = dat_radar
        dat['radar_rng'] = dat_radar_rng
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