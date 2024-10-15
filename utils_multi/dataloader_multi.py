import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from .transform_utils_keypoint import *
from .dataset import RadarDataset, RadarDataset_Keypoint, read_h5_basic
from torchvision import transforms
import pandas as pd
import numpy as np
import random
import pickle
import joblib
from scipy import signal
# random.seed(144000)

import copy

def medfilt_keypoint(keypoint, kernel_size):
    keypoint_new = np.zeros((len(keypoint), 17, 3))
    for body in range(17):
        for coord in range(3):
            keypoint_new[:,body,coord] = signal.medfilt(keypoint[:,body,coord], kernel_size)
    return keypoint_new

def Preprocess_Keypoint(args):
    """
    Preprocess & Save the pre-processed data
    """
    # load & clean des
    des_all = pd.read_csv(args.result.csv_file, index_col=0)
    val_idx=((des_all['len_radar']==1985)&
            (des_all['len_raw_vid']>=300)&
            (des_all['len_raw_vid']==des_all['len_keypoint_3D'])&
            (des_all['n_detect_0']<15)
            )
    des_clean = des_all[val_idx]

    composed_all = transforms.Compose([
        ToCHWTensor(apply=['keypoint']),
        NormalizeKeypoint(),
        LPFpoint(apply=['radar_pcl', 'keypoint']),
        ResizeRadarPcl(),
    ])
    composed_keypoint = transforms.Compose([
        ToCHWTensor(apply=['keypoint']),
        LPFpoint(apply=['keypoint']),
    ])
    composed_radarpcl = transforms.Compose([
        LPFpoint(apply=['radar_pcl']),
        ResizeRadarPcl(),
    ])

    data_dir=args.result.data_dir
    for idx in tqdm(range(len(des_clean))):
        fname = des_clean.iloc[idx]
        folder = fname['Folder']
        episode = str(fname['Episode'])
        radar_path = os.path.join(data_dir, folder, 'radar_v2', episode+'.h5')
        radar_pcl_path = os.path.join(data_dir, folder, 'radar', episode+'.pkl')
        keypoint_path = os.path.join(data_dir, folder, episode)

        radar_dat, radar_rng, radar_des = read_h5_basic(radar_path)
        radar_pcl = pd.DataFrame(pickle.load( open( radar_pcl_path, "rb" ) ))

        ###### load adjusted keypoint
        keypoint_path_4dhuman = '/workspace/0_temp/4dhuman/results/'
        keypoint_info = joblib.load(keypoint_path_4dhuman + f'/demo_{episode}.pkl')
        KEYPOINT_LIST_17 = [39, 28, 29, 30, 27, 26, 25, 41, 37, 42, 43, 33, 32, 31, 34, 35, 36]
        frame_list = list(keypoint_info.keys())
        keypoint_3D_temp = [keypoint_info[frame]['3d_joints'] for frame in frame_list]
        nsubject_keypoint_3D = np.array([len(keypoint) for keypoint in keypoint_3D_temp])

        T           = len(keypoint_3D_temp)
        output_sel = np.zeros((T,17,3))
        n_idx       = 0                 # idx of person
        for t_idx in range(T):
            if len(keypoint_3D_temp[t_idx])<1:
                continue
            keypoint_3d_t = keypoint_3D_temp[t_idx][n_idx][KEYPOINT_LIST_17,:]
            keypoint_3d_t -= keypoint_3d_t[[0], :]         # align origin
            output_sel[t_idx,:,:] = keypoint_3d_t
        output_filt = medfilt_keypoint(output_sel, kernel_size = 7)
        keypoint_3D = output_filt
        #######

        data = {}
        data['radar'] = radar_dat
        data['radar_rng'] = radar_rng
        data['radar_pcl'] = radar_pcl
        data['keypoint'] = keypoint_3D
        data['des'] = fname

        
        # pre-processing
        data = composed_all(data) 

        # Save
        ## save preprocessed keypoint
        if args.preprocess.save_keypoint:
            np.savez_compressed(keypoint_path + '/output_3D/keypoint3D_adjusted.npz', 
                                reconstruction=np.array(data['keypoint']))
        ## save preprocessed radar_pcl
        if args.preprocess.save_radarPCL:
            with open(os.path.join(data_dir, folder, 'radar', episode+'_preprocessed_LPF-0.05.pkl'),"wb") as f:
                pickle.dump(data['radar_pcl'].to_dict('records'), f)

def Analyze_statistics(args):
    """
    Preprocess & Save the pre-processed data
    """
    # load & clean des
    des_all = pd.read_csv(args.result.csv_file, index_col=0)
    val_idx=((des_all['len_radar']==1985)&
            (des_all['len_raw_vid']>=300)&
            (des_all['len_raw_vid']==des_all['len_keypoint_3D'])&
            (des_all['n_detect_0']<15)
            )
    des_clean = des_all[val_idx]

    size_radar = (args.transforms.Dop_size, args.transforms.win_size)
    size_radar_rng = (args.transforms.R_size_rng, args.transforms.win_size_rng)
    composed_test = transforms.Compose([
        ToCHWTensor(apply=['radar', 'keypoint']),
        # RotateKeypoint(),
        UniformCrop_Time(win_sec=args.transforms.win_sec, n_div=args.transforms.test_ndiv, apply=['radar', 'radar_pcl', 'keypoint']),
        ResizeRadar(size_mD=size_radar, size_rng=size_radar_rng, flag_train=False),
        ResizeKeypoint(len=16,flag_train=False),
    ])

    data_dir=args.result.data_dir
    radar_list = []
    radar_rng_list = []
    keypoint_list = []
    for idx in tqdm(range(len(des_clean))):
        fname = des_clean.iloc[idx]
        folder = fname['Folder']
        episode = str(fname['Episode'])
        radar_path = os.path.join(data_dir, folder, 'radar_v2', episode+'.h5')
        radar_pcl_path = os.path.join(data_dir, folder, 'radar', episode+'_preprocessed_LPF-0.05.pkl')
        keypoint_path = os.path.join(data_dir, folder, episode)

        radar_dat, radar_rng, radar_des = read_h5_basic(radar_path)
        radar_pcl = pd.DataFrame(pickle.load( open( radar_pcl_path, "rb" ) ))
        keypoint_3D = np.load(keypoint_path + '/output_3D/keypoint3D_adjusted.npz', allow_pickle=True)['reconstruction']    # 3D keypoint

        data = {}
        data['radar'] = radar_dat
        data['radar_rng'] = radar_rng
        data['radar_pcl'] = radar_pcl
        data['keypoint'] = keypoint_3D
        data['des'] = fname
        
        # pre-processing
        data = composed_test(data) 
        
        radar_list.append(data['radar'])
        radar_rng_list.append(data['radar_rng'])
        keypoint_list.append(data['keypoint'])
    radar_all = torch.stack(radar_list)
    radar_rng_all = torch.stack(radar_rng_list)
    keypoint_all = torch.stack(keypoint_list)
    a = 1

def LoadDataset_Keypoint(args):
    """Do transforms on radar data and labels. Load the data from 2 radar sensors.

    Args:
        args: args configured in Hydra YAML file

    """
    # load & clean des
    des_all = pd.read_csv(args.result.csv_file, index_col=0)
    if 'episode' in args.preprocess.keys():     # select only episode
        val_idx=((des_all['len_radar']==1985)&
            (des_all['len_raw_vid']>=300)&
            (des_all['Folder']==args.preprocess.episode)
            )
    else:
        val_idx=((des_all['len_radar']==1985)&
            (des_all['len_raw_vid']>=300)&
            (des_all['len_raw_vid']==des_all['len_keypoint_3D'])&
            (des_all['n_detect_0']<15)
            )
    des_clean = des_all[val_idx]

    # train-test split
    ## select 'class' if not 'all'
    des_clean = des_clean.to_dict('records')
    if args.train.traintest_class!='all':
        class_list = np.array([des['class'] for des in des_clean])
        target_class = args.train.traintest_class
        idx_select = np.where(target_class==class_list)[0].tolist()
        des_clean = [des_clean[idx] for idx in idx_select]

    ## random split
    if args.train.traintest_split=='random':
        random.Random(22).shuffle(des_clean)
        des_train = des_clean[:round(len(des_clean)*0.8)]
        des_test = des_clean[round(len(des_clean)*0.8):]
    ## subject independent split
    elif args.train.traintest_split=='subject_independent':
        des_clean = pd.DataFrame(des_clean)
        val_train=(((des_clean['ID']!='soheil') & (des_clean['ID']!='shreya') & (des_clean['ID']!='sonny')))
        val_test=(((des_clean['ID']=='soheil') | (des_clean['ID']=='shreya') | (des_clean['ID']=='sonny')))
        des_train = des_clean[val_train].to_dict('records')
        des_test = des_clean[val_test].to_dict('records')
        assert len(des_clean)==(len(des_train)+len(des_test))
    ## class independent split
    elif 'class_independent' in args.train.traintest_split:
        assert args.train.traintest_class=='all'
        class_list = np.array([des['class'] for des in des_clean])
        target_class = args.train.traintest_split.split('_')[-1]
        idx_train = np.where(target_class!=class_list)[0].tolist()
        idx_test = np.where(target_class==class_list)[0].tolist()
        des_train = [des_clean[idx] for idx in idx_train]
        des_test = [des_clean[idx] for idx in idx_test]
    elif args.train.traintest_split=='all':
        des_train = des_clean
        des_test = des_clean
    

    (mean1,mean2) = (args.transforms.radar1_mean, args.transforms.radar2_mean)
    (std1,std2) = (args.transforms.radar1_std, args.transforms.radar2_std)
    (mean_rng1,mean_rng2) = (args.transforms.radar1_rng_mean, args.transforms.radar2_rng_mean)
    (std_rng1,std_rng2) = (args.transforms.radar1_rng_std, args.transforms.radar2_rng_std)
    size_radar = (args.transforms.Dop_size, args.transforms.win_size)
    size_radar_rng = (args.transforms.R_size_rng, args.transforms.win_size_rng)

    ### Compose the transforms on train set
    composed_train = transforms.Compose([
        ToCHWTensor(apply=['radar', 'keypoint']),
        RandomizeCrop_Time(win_sec=args.transforms.win_sec, apply=['radar', 'radar_pcl', 'keypoint']),
        ResizeRadar(size_mD=size_radar, size_rng=size_radar_rng, flag_train=True),
        RandFlip(p=0.5, apply=['radar', 'radar_pcl', 'keypoint']),
        ResizeKeypoint(len=16,flag_train=True),
        NormalizeRadar(mean_mD=(mean1,mean2), std_mD=(std1,std2), mean_rng=(mean_rng1,mean_rng2), std_rng=(std_rng1,std_rng2), flag_train=True),
    ])
    ### Compose the transforms on valid and test sets
    composed_test = transforms.Compose([
        ToCHWTensor(apply=['radar', 'keypoint']),
        UniformCrop_Time(win_sec=args.transforms.win_sec, n_div=args.transforms.test_ndiv, apply=['radar', 'radar_pcl', 'keypoint']),
        ResizeRadar(size_mD=size_radar, size_rng=size_radar_rng, flag_train=False),
        ResizeKeypoint(len=16,flag_train=False),
        NormalizeRadar(mean_mD=(mean1,mean2), std_mD=(std1,std2), mean_rng=(mean_rng1,mean_rng2), std_rng=(std_rng1,std_rng2), flag_train=False),
    ])

    ### test for video
    composed_test_video = transforms.Compose([
        ToCHWTensor(apply=['radar', 'keypoint']),
        UniformCrop_Time(win_sec=args.transforms.win_sec, n_div=args.transforms.test_ndiv, apply=['radar', 'radar_pcl', 'keypoint']),
        ResizeRadar(size_mD=size_radar, size_rng=size_radar_rng, flag_train=False),
        ResizeKeypoint(len=16,flag_train=False),
        NormalizeRadar(mean_mD=(mean1,mean2), std_mD=(std1,std2), mean_rng=(mean_rng1,mean_rng2), std_rng=(std_rng1,std_rng2), flag_train=False),
    ])

    radar_train = RadarDataset_Keypoint(
                                des=des_train,
                                data_dir=args.result.data_dir,
                                flag='train',
                                args=args,
                                transform=composed_train, 
                                )
    radar_test =  RadarDataset_Keypoint(
                                des=des_test, 
                                data_dir=args.result.data_dir,
                                flag='test',
                                args=args,
                                transform = composed_test,
                                )
    radar_test_video =  RadarDataset_Keypoint(
                            des=des_test, 
                            data_dir=args.result.data_dir,
                            flag='test_video',
                            args=args,
                            transform = composed_test_video,
                            )

    data_train = DataLoader(radar_train, batch_size=args.train.batch_size, shuffle=False, num_workers=args.train.num_workers)
    data_test = DataLoader(radar_test, batch_size=int(np.max((args.train.batch_size//8,1))), shuffle=False, num_workers=args.train.num_workers)

    return data_train, data_test, radar_test_video, (len(radar_train), len(radar_test))

def my_collate_fn(batch):
    collate_data = []
    for idx in range(len(batch[0])):
        collate_data.append([])
    for sample in batch:
        for idx in range(len(sample)):
            collate_data[idx].append(sample[idx])
            # if isinstance(data, list) or isinstance(data, dict):
    for idx in range(len(collate_data)):
        if isinstance(collate_data[idx][0], list) or isinstance(collate_data[idx][0], dict):
            collate_data[idx] = collate_data[idx]
        else:   # if data is torch
            collate_data[idx] = torch.stack(collate_data[idx])
    return collate_data