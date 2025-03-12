import torch
import hydra
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import signal
from utils_multi.result_utils import *
from utils_multi.camera import *
from utils_multi import dataloader_multi

from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from model import mobileVit_test23_xformer

def result_load(device, path_model, path_args):
  args = OmegaConf.load(path_args)

  if 'encoder_input' not in args.model.keys():
    args.model.encoder_input = 'multi'
  args.train.traintest_split = 'subject_independent'
  data_train, data_test, data_test_video, len_data = dataloader_multi.LoadDataset_Keypoint(args)
  model = mobileVit_test23_xformer.main_Net(args).to(device)

  keypoint_startpoint = [data_test_video.__getitem__(i)[2][1] for i in range(len(data_test_video))]
  list_episode = [data_test_video.__getitem__(i)[5] for i in range(len(data_test_video))]

  # Load model
  model.load_state_dict(torch.load(path_model))
  model.eval()
  model_size = count_parameters(model)

  test_loss, des_test, (y_target, y_pred), coord = test_keypoint(data_test, device, model, output_pred=True, output_temporal=True)
  # Select only desginated class
  class_idx = [i for i in range(len(des_test))]
  test_loss_sel = {}
  for key in test_loss.keys():
    test_loss_sel[key] = test_loss[key][class_idx]
  des_test_sel = [des_test[idx] for idx in class_idx]
  # Resize coord: (B x Sample x T x 4) -> (B x Sample x T_new x 4)
  coord_resize = np.zeros((coord.shape[0], coord.shape[1], y_target.shape[2], coord.shape[3]))
  t_ori = np.linspace(0,coord.shape[2]-1,coord.shape[2])
  t_new = np.linspace(0,coord.shape[2]-1,y_target.shape[2])
  for batch in range(coord.shape[0]):
    for n in range(coord.shape[1]):
      for key_val in range(coord.shape[3]):
        coord_sel = coord[batch,n,:,key_val]
        coord_interp = np.interp(t_new, t_ori, coord_sel)
        coord_resize[batch,n,:,key_val] = coord_interp

  return test_loss_sel, des_test_sel, (y_target[class_idx], y_pred[class_idx]), coord_resize[class_idx], model_size, args, keypoint_startpoint, list_episode

@hydra.main(version_base=None, config_path="conf", config_name="config_inference")
def main(args: DictConfig) -> None:
  config = OmegaConf.to_container(args)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  path_save = config['path_save']
  FPS = 90
  

  list_keypoint_start = [16, 36, 56, 76, 96, 116, 136, 156, 176, 196]

  test_loss, des_test, (y_target, y_pred), coord_resize, model_size, args_result, startpoint_idx, episodes = result_load(device, config['path_model'], config['path_args'])
  num_sample, num_tstart   = y_pred.shape[0], y_pred.shape[1]
  for i_sample in tqdm(range(num_sample)):
    name_episode  = des_test[i_sample]['fname'].split('.')[0]
    name_subject  = des_test[i_sample]['subject']
    name_pattern  = des_test[i_sample]['pattern']
    if str(args.test_episode) in name_episode:
      show3Dpose_fromradar_video(device, y_target[i_sample], list_keypoint_start, select_frame=5, path=f'{path_save}/{name_episode}-{name_subject}-{name_pattern}-GT.mp4')
      show3Dpose_fromradar_video(device, y_pred[i_sample], list_keypoint_start, select_frame=5, path=f'{path_save}/{name_episode}-{name_subject}-{name_pattern}-Pred.mp4')   


if __name__ == '__main__':
  main()