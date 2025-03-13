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
        UniformCrop_Time(win_sec=args.transforms.win_sec, n_div=args.transforms.test_ndiv, apply=['radar', 'keypoint']),
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
        keypoint_path = os.path.join(data_dir, folder, episode)

        radar_dat, radar_rng, radar_des = read_h5_basic(radar_path)
        keypoint_3D = np.load(keypoint_path + '/output_3D/keypoints.npz', allow_pickle=True)['reconstruction']    # 3D keypoint

        data = {}
        data['radar'] = radar_dat
        data['radar_rng'] = radar_rng
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
        RandomizeCrop_Time(win_sec=args.transforms.win_sec, apply=['radar', 'keypoint']),
        ResizeRadar(size_mD=size_radar, size_rng=size_radar_rng, flag_train=True),
        RandFlip(p=0.5, apply=['radar', 'keypoint']),
        ResizeKeypoint(len=16,flag_train=True),
        NormalizeRadar(mean_mD=(mean1,mean2), std_mD=(std1,std2), mean_rng=(mean_rng1,mean_rng2), std_rng=(std_rng1,std_rng2), flag_train=True),
    ])
    ### Compose the transforms on valid and test sets
    composed_test = transforms.Compose([
        ToCHWTensor(apply=['radar', 'keypoint']),
        UniformCrop_Time(win_sec=args.transforms.win_sec, n_div=args.transforms.test_ndiv, apply=['radar', 'keypoint']),
        ResizeRadar(size_mD=size_radar, size_rng=size_radar_rng, flag_train=False),
        ResizeKeypoint(len=16,flag_train=False),
        NormalizeRadar(mean_mD=(mean1,mean2), std_mD=(std1,std2), mean_rng=(mean_rng1,mean_rng2), std_rng=(std_rng1,std_rng2), flag_train=False),
    ])

    ### test for video
    composed_test_video = transforms.Compose([
        ToCHWTensor(apply=['radar', 'keypoint']),
        UniformCrop_Time(win_sec=args.transforms.win_sec, n_div=args.transforms.test_ndiv, apply=['radar', 'keypoint']),
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