# ros_object_detection
ros package for deep learning based object detection with lidar

## Description

This repository is used to deploy trained object detection model in ros ecosystem. It requires model to be in torchscript model and its output should be Nx7 tensor where N is number of boxes and each box contains class, score, center_x, center_y, width, lenght, yaw information

## Getting Started

### Dependencies

* Pytorch 1.12.1
* Vispy
* Numpy
* Matplotlib

### Training
```
python main.py --config <location of config file>
```
#### Config File
Config file is written in json format and it allows choosing different models, classification encoding and adding custom dataset for training. Here I will go over important entries in config file. 

#### model
This entry contains information regarding the model. Here you can choose which backbone you want to train with ```backbone``` key. Available options are ```pixor``` , ```mobilepixor``` and ```rpn```. You can also choose what kind of encoding you want to use for classification head with ```cls_encoding```. Available options are ```gaussian``` (from Cornernet), ```inverse_distance``` (from Afdet) and ```binary``` (from PIXOR).

#### data
This entry contains necessary information for encoding and decoding pointcloud. ```out_size_factor``` refers to down sampling ratio of backbone. It is 1 for rpn, and 4 for pixor and mobilepixor. Each dataset is given specific name so that you can store them in different locations and use different values for encoding and decoding. Necessary values of this are min and max values in each of x, y, z dimensions and voxel resolution in meters. Object names might be different in each dataset, so you need to provide mapping between object name and class number with ```objects``` key.

#### augmentation
This entry is used to enable/disable augmentation. Available data augmentations are random rotation, scaling and translation.

#### Custom Dataset
* Choose frames from rosbag and save them in .bin format using ```tools/extract_data.py``` .
* Label extracted data using some labeling tool. It should be in kitti format.
* Create a txt file that contains names of data that you want to use for training. Format should be ```filename;dataset name``` ex: ```014529;lyft```. Refer to ```utils/split_dataset.py```. 
* Add relevant info to config file. 

#### Visualization
* ```vis/test_vis.py``` for visualizing model prediction
* ```vis/dataset_vis.py``` for visualizing voxels and box encodings
* ```vis/rosbag_vis.py``` for visualizing model prediction on rosbags
* ```vis/torchscript_vis.py``` for visualizing torchscrit model prediction

### Deployment
To deploy the trained model, we first need to convert it to torchscript model. We implement postprocessing (most part) in forward function of the model to utilize gpu. The output of torchscript model is Nx7 tensor where N is number of boxes and each box contains class, score, center_x, center_y, width, lenght, yaw information. Use [this repo](https://github.com/charyyev/ros_object_detection) to deploy it in ros ecosystem.

## References
* [PIXOR](https://arxiv.org/pdf/1902.06326.pdf)
* [AFDET](https://arxiv.org/pdf/2006.12671.pdf)
* [AFDETv2](https://arxiv.org/pdf/2112.09205.pdf)
* [MOBILENETv2](https://arxiv.org/pdf/1801.04381.pdf)
