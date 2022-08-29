# ros_object_detection
ros package for deep learning based object detection with lidar

## Description

This repository is used to deploy trained object detection model in ros ecosystem. It requires model to be in torchscript model and its output should be Nx7 tensor where N is number of boxes and each box contains class, score, center_x, center_y, width, length, yaw information. To achieve high inference speed, converting pointcloud to voxel is implemented on gpu. 

## Getting Started

### Dependencies

* CUDA
* libtorch
* boost

### Running
```
roslaunch object_detetion dl_detector.launch
```
