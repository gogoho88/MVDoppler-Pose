# MVDoppler-Pose

## Environment
* Python: 3.10.8
* Pytorch: 1.13.1
* CUDA: 11.6
* CuDNN: 8
* Environment can directly be imported through [Docker image](https://hub.docker.com/repository/docker/gogoho88/stanford_mmwave/tags/v3/sha256-481efb7f0500f3657296cd8e1320404887e18f49a2e6683fbcec18d6a9e7d212)

## Preparing the Dataset
* Download the dataset from this [Google Drive link](https://drive.google.com/drive/folders/11e_L9glHIoE5O8o1kukAA-M_2me60Vmy?usp=share_link)
* Unzip folders `Data` and `Metadata`
* `Data` has all the samples with structure of
```
dataset
    ├── 2022Jul13-1744
        ├── radar_v2
            ├── 20220713174506.h2
            ├── 20220713174732.h2	
            ...
        ├── 20220713174506
            ├── output_3D
                ├── keypoints.npz
        ├── 20220713174732
        ...
    ├── 2022Jul13-1806
    ...
```
* `des_all.csv` includes the metadata for the entire dataset in camma seperated values (CSV) format.<br>

## Argument configurations
This codebase uses [Hydra](https://github.com/facebookresearch/hydra) to manage and configure arguments. Hydra offers more flexibility for running and managing complex configurations, and supports rich hierarchical config structures.

The YAML configuration files are in folder `conf`. So you can have a set of arguments in your YAML file like
```
train:
  learning_rate: 1e-4
transforms: 
  win_size: 512
```

## Code Tree
`/main_multi_keypoint.py`: Main file to run the code<br>
`/conf/`: Configuration file for adjusting parameters<br>
`/model/`: Include transformer-based neural network models<br>
`/utils_multi/`: Utility functions for training and testing models such as dataloader, data transformation, model training/testing 

## Train the Baseline Model
* Specify data folder and metadata file through `data_dir` and `csv_file` in `/conf/config_keypoint_adjust.yaml`
* Rut it through python
```
cd mmWave-Pose
python main_multi_keypoint.py
```