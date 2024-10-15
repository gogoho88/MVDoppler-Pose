"""
Datasets for gaits, identity, location, and velocities
"""

import torch
from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import h5py
import pickle
import joblib
from scipy import signal

def medfilt_keypoint(keypoint, kernel_size):
    keypoint_new = np.zeros((len(keypoint), 17, 3))
    for body in range(17):
        for coord in range(3):
            keypoint_new[:,body,coord] = signal.medfilt(keypoint[:,body,coord], kernel_size)
    return keypoint_new

def read_h5_basic(path):
    """Read HDF5 files

    Args:
        path (string): a path of a HDF5 file

    Returns:
        radar_dat: micro-Doppler data with shape (256, 128, 2) as (1, time, micro-Doppler, 2 radar channel)
        des: information for radar data
    """
    hf = h5py.File(path, 'r')
    radar_dat = np.array(hf.get('radar_dat'))
    radar_rng = np.array(hf.get('radar_rng'))
    des = dict(hf.attrs)
    hf.close()
    return radar_dat, radar_rng, des

class RadarDataset(Dataset):
    """
    Dataset of different classifications

    The input-output pairs (radar_dat, label) of the RadarDataset are of the following form:
    radar_dat: radar data with shape (micro-Doppler range, time range, 3). The last dimension is three-channels 
                for RGB image. The one-channel data is repeated three times to generate three channels.
    label: depends on the argument `label_type`

    Args:
        csv_file (string): path of the csv file containing labels and information
        data_dir: path of all radar data
        transform: transform function on `radar_dat` contained in `transform_utils.py`
        target_transform: transform function on `label` contained in `transform_utils.py` 
        label_type: wanted label type. Look function `get_label()` for detail.       
        return_des: if True, returns information of the radar data in addition to the input-output pair
        data_format: wanted data format
    """
    def __init__(self,
                 file_list,
                 data_dir,
                 transform=None,
                 target_transform=None,
                 return_des=False,
                 ):
#        self.csv_file = pd.read_csv(csv_file)
        self.file_list = file_list
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.label_type = label_type
        self.return_des = return_des

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # fname = self.csv_file.iloc[idx].item()
        fname = self.file_list[idx]
        data_path = os.path.join(self.data_dir, fname + '.h5')

        radar_dat, des = read_h5_basic(data_path) 

        if self.transform:
            radar_dat = self.transform(radar_dat)
            
        if self.target_transform:
            label = self.target_transform(des)
        
            if self.return_des:
                return radar_dat.type(torch.FloatTensor), label.type(torch.FloatTensor), des
            else:
                return radar_dat.type(torch.FloatTensor), label.type(torch.FloatTensor)
        else:
            return radar_dat.type(torch.FloatTensor), des

class RadarDataset_Keypoint(Dataset):
    """
    """
    def __init__(self,
                 des,
                 data_dir,
                 flag,
                 args,
                 transform=None,
                 target_transform=None,
                 ):
        self.des = des
        self.data_dir = data_dir
        self.flag = flag
        self.load = args.preprocess.load
        self.transform = transform
        self.target_transform = target_transform
        self.input_sensor = args.model.encoder_input

    def __len__(self):
        return len(self.des)

    def __getitem__(self, idx):
        fname = self.des[idx]
        folder = fname['Folder']
        episode = str(fname['Episode'])
        radar_path      = os.path.join(self.data_dir, folder, 'radar_v2', episode+'.h5')
        radar_pcl_path  = os.path.join(self.data_dir, folder, 'radar', episode + self.load.radarpcl + '.pkl')
        keypoint_path   = os.path.join(self.data_dir, folder, episode)

        radar_dat, radar_rng, radar_des = read_h5_basic(radar_path)
        radar_pcl = pd.DataFrame(pickle.load( open( radar_pcl_path, "rb" ) ))
        keypoint_3D = np.load(keypoint_path + '/output_3D/keypoint3D_adjusted.npz', allow_pickle=True)['reconstruction']    # 3D keypoint

        data = {}
        data['radar'] = radar_dat
        data['radar_rng'] = radar_rng
        data['radar_pcl'] = radar_pcl
        data['keypoint'] = keypoint_3D
        data['des'] = fname

        if self.transform:
            data = self.transform(data)

        ##### 99
        if self.input_sensor=='single':
            if self.flag=='train':
                data['radar'] = torch.cat((data['radar'][0].unsqueeze(dim=0),data['radar'][0].unsqueeze(dim=0)),dim=0)
                data['radar_rng'] = torch.cat((data['radar_rng'][0].unsqueeze(dim=0),data['radar_rng'][0].unsqueeze(dim=0)),dim=0)
            else:
                data['radar'] = torch.cat((data['radar'][:,0,:,:].unsqueeze(dim=1),data['radar'][:,0,:,:].unsqueeze(dim=1)),dim=1)
                data['radar_rng'] = torch.cat((data['radar_rng'][:,0,:,:].unsqueeze(dim=1),data['radar_rng'][:,0,:,:].unsqueeze(dim=1)),dim=1)
        #####

        if self.flag=='train':
            return data['radar'], data['radar_rng'], data['keypoint']
        elif self.flag=='test':
            return data['radar'], data['radar_rng'], data['keypoint'], data['radar_pcl'], radar_des
        elif self.flag=='test_video':
            return data['radar'], data['radar_rng'], (data['keypoint'], data['keypoint_startpoint']), data['radar_pcl'], radar_des, (folder, episode)
        