# <p align="center"><b>`MS-TIP:Imputation Aware Pedestrian Trajectory Prediction`</b></p>

This repository contains the code for the paper "MS-TIP: Imputation Aware Pedestrian Trajectory Prediction," accepted at ICML 2024. MS-TIP is a novel approach designed to enhance pedestrian trajectory prediction by being imputation aware.


<p align="center">
<img src="/MSTIP.png" />
<p>
<hr />


### Dependencies

Install the dependencies from the `requirements.txt`:
```linux
conda activate mstip
```
or
```linux
conda env create -f environment.yml --name mstip
```

## Training
To train a MS-TIP model on the dataset, simply run:
```
./scripts/train.sh -d "eth" -i "0" -n 0.15
```
- ```-d``` flag is the dataset
- ```-i``` flag specifies the gpu id
- ```-n``` flag specifes how many percentage of data will be set as input during training time. Possible values : 0.05, 0.1, 0.15, 0.2

## Evaluating

To evalutate the model performance, simply run:
```linux
./scripts/test.sh -d "eth" -i "0" -n 0.15 -b true > output_eth
```
- The flags have same meaning as above. Set ```-b``` flag to true always.


## Acknowledgement
We would like to thank the authors of the following repositories for open-sourcing their code, parts of which were used or referenced in this work: [GroupNet](https://github.com/MediaBrain-SJTU/GroupNet/tree/main), [Graph-TERN](https://github.com/InhwanBae/GraphTERN/tree/main), [PECNet](https://github.com/HarshayuGirase/Human-Path-Prediction), [SAITS](https://github.com/WenjieDu/SAITS).


</center>

<hr />

# Citation


```
@inproceedings{Chib, Pranav Singh and Singh, Pravendra,
  title={MS-TIP:Imputation Aware Pedestrian Trajectory Prediction},
  author={Chib, Pranav Singh and Singh, Pravendra},
  booktitle={Proc. 40th International Conference on Machine Learning (ICML 2024)},
  location = {Vienna, Austria},
  pages={},
  year={2024}
}
