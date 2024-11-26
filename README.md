# me-GCN
This repo is the official implementation for [Learning Mutual Excitation for Hand-to-Hand and Human-to-Human Interaction Recognition]

# Prerequisites

- Python >= 3.6
- PyTorch >= 1.1.0
- PyYAML, tqdm, tensorboardX

- We provide the dependency file of our experimental environment, you can install all dependencies by creating a new anaconda virtual environment and running `pip install -r requirements.txt `

# Data Preparation

### Download datasets.

#### There are 3 datasets to download:

- NTU60-Interaction
- NTU120-Interaction
- Assembely101

#### NTU60-Interaction and NTU120-Interaction

1. Request dataset here: https://rose1.ntu.edu.sg/dataset/actionRecognition
2. Download the skeleton-only datasets:
   1. `nturgbd_skeletons_s001_to_s017.zip` (NTU60-Interaction)
   2. `nturgbd_skeletons_s018_to_s032.zip` (NTU120-Interaction)
   3. Extract *A050.skeleton to *A060.skeleton files to data/ntu60/ntu60_interaction/ and data/ntu120/ntu120_interaction/
   4. Extract *A106.skeleton to *A120.skeleton files to data/ntu120/ntu120_interaction/

#### Assembely101

1. Download dataset from [here](https://github.com/assembly-101/assembly101-download-scripts)

### Data Processing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/Assembely101/train_data_joint_200.npy
- data/Assembely101/test_data_joint_200.npy
- data/ntu60/ntu60_interaction/*.skeleton
- data/ntu120/ntu120_interaction/*.skeleton
```

#### Generating Data

- Generate Assembely101 dataset:
```
 cd ./data/Assembely101
 python convert.py
```
- Generate NTU60-Interaction and NTU120-Interaction datasets:
```
 cd ./data/ntu60 # or cd ./data/ntu120  
 python get_raw_skes_data.py
 python get_raw_denoised_data.py
 python seq_transformation.py
```


# Training

### Training
```
nohup python main.py --config config/nturgbd120-cross-subject/joint.yaml --device 0 &
#nohup python main.py --config config/nturgbd120-cross-set/joint.yaml --device 0 &
#nohup python main.py --config config/nturgbd-cross-subject/joint.yaml --device 0 &
#nohup python main.py --config config/nturgbd-cross-view/joint.yaml --device 0 &
#python main.py --config config/baseline_ctrgcn_interaction_biBetaInit0.yaml --device 0
```

This repo is based on [2s-AGCN](https://github.com/lshiwjx/2s-AGCN). The data processing is borrowed from [SGN](https://github.com/microsoft/SGN) and [HCN](https://github.com/huguyuehuhu/HCN-pytorch).

Thanks to the original authors for their work!

# Citation

Please cite this work if you find it useful:.

      @inproceedings{liu2024learning,
        title={Learning Mutual Excitation for Hand-to-Hand and Human-to-Human Interaction Recognition},
        author={Liu, Mengyuan and Chen, Chen and Wu, Songtao and Meng, Fanyang and Liu, Hong},
        year={2024}
      }

# Contact
For any questions, feel free to contact: `liumengyuan@pku.edu.cn`
