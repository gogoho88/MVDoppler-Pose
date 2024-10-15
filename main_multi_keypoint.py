import os
import sys
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from utils_multi.trainer_keypoint import Trainer
from utils_multi.result_utils import count_parameters
from utils_multi.dataloader_multi import *

# sys.path.append(os.path.abspath('/workspace/'))
import hydra
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

import wandb

from model import mobileVit_test23_xformer

@hydra.main(version_base=None, config_path="conf", config_name="config_keypoint_adjust")
def main(args: DictConfig) -> None:
  config = OmegaConf.to_container(args)
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # Preprocessing & Save
  if args.preprocess.flag_preprocess:
    Preprocess_Keypoint(args)
    return
  if args.preprocess.flag_statistics:
    Analyze_statistics(args)
    return

  data_train, data_test, data_test_video, len_data = LoadDataset_Keypoint(args)

  if args.wandb.use_wandb:
    wandb.init(
          project = args.wandb.project, 
          entity = "gogoho88", 
          config = config, 
          notes = "test",
          name = args.result.name
          )
    wandb.config = {
          "learning_rate": args.train.learning_rate,
          "weight_decay": args.train.weight_decay,
          "delta": args.train.delta,
          "alpha_limb": args.train.alpha_limb,
          "alpha_limb_motion": args.train.alpha_limb_motion,
          }
  
  model = mobileVit_test23_xformer.main_Net(args).to(device)

  # Learning
  trainer = Trainer(model=model, 
                    data_train=data_train, 
                    data_valid=[],
                    data_test=data_test,
                    data_test_video=data_test_video, 
                    args=args, 
                    device=device,
                    )
  trainer.train()

  wandb.finish()


if __name__ == '__main__':
  main()