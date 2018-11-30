# RPNet: an End-to-End Network for Relative Camera Pose Estimation

# Setting up the environement:
  - Python3 
  - Tensorflow 1.6+
  - Numpy
  - Scipy
  - matplotlib
  - skimage
  - opencv
  - pyquaternion (http://kieranwynn.github.io/pyquaternion/)
  - mayavi (for toy_example)

# Data preparation
1. change the log_dir in abstract_network/setting.py
2. cd data
3. python3 absolute_cambridge.py # for absolute pose estimation
4. python3 relative_cambridge.py # for relative pose estimation

If you wish to have different train and test set from our experiments, delete train_set.txt, test_set.txt and validation_set.txt in log_dir/relative_cambridge/dataset_name/

# Training:
To train the network on the four datasets of Cambridge with default (best) parameters, simply run the following command:

```sh
$ python3 posenet.py --train_test_phase=train # in posenet folder to train posenet
```

or 

```sh
$ python3 rpnetplus.py --train_test_phase=train # in rpnet folder to train posenet
```
Each training will be logged at $log_dir/absolute_cambridge_network or $log_dir/relative_cambridge_network.

To see all the customizable parameters for each model, run with "--help" option. 

```sh
$ python3 rpnetplus.py --help
```
Train test split: https://goo.gl/vv3zxB

# Results on Cambridge Dataset using PoseNet

|                  | Paper       | This implementation |
|------------------|-------------|---------------------|
| King's Colleges  | 1.92m, 5.4  | 1.93m, 3.12         |
| Old Hopistal     | 2.31m, 5.40 | 2.41m, 4.81         |
| Shop Facade      | 1.46, 8.0   | 1.68, 7.07          |
| St Mary's church | 2.65, 8.48  | 2.29, 5.90          |

# Evaluation:
To evaluate the trained network, simply run the following command:
```sh
$ python3 rpnetplus.py --train_test_phase=test
```
It will generate a *.pkl file saving T(m), R(d) and T(d) in each conguration. 

# Toy example

The relative pose is computed in the second camera's reference system, following OpenCV. To better understand how the to compute these values, as well as, the importance of different parameters (focal length, principle point ...), you can work on our toy example. 

![3D rendering of the two camera poses and its projected points on the virtual image plan.](https://github.com/ensv/RPNet.git/toy_example/toy_example.png)

# Notice
All the codes and resources in GoogLeNet folder are mostly based on the work of https://github.com/kentsommer/tensorflow-posenet. 
