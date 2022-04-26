#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <torch/torch.h>
#include <torch/script.h>
#include "robot_msgs/DetectedObject.h"
#include "robot_msgs/DetectedObjectArray.h"
#include <visualization_msgs/MarkerArray.h>
#include <visualization_msgs/Marker.h>
#include <math.h>
#include <tf/tf.h>

class ObjectDetection
{
    private:
        ros::Subscriber pcl_sub;
        ros::Publisher objects_pub;
        ros::Publisher box_pub;
        ros::Publisher pcl_pub;

        torch::jit::script::Module model;
        pcl::PointCloud<pcl::PointXYZ> pointcloud;

        // voxel properties
        const float x_min = -35.0;
        const float x_max = 35.0;
        const float y_min = -40.0;
        const float y_max = 40.0;
        const float z_min = -2.5;
        const float z_max = 1.0;
        const float x_res = 0.1;
        const float y_res = 0.1;
        const float z_res = 0.1;
        const float score_threshold = 0.5;

        bool use_gpu = false;

        std::string object_classes[4] = {"background", "car", "pedestrian", "cyclist"};

    public:
        ObjectDetection(ros::NodeHandle *nh);
        void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input);
        torch::Tensor pcl_to_voxel();
        pcl::PointCloud<pcl::PointXYZ> voxel_to_pcl(torch::Tensor voxel);
        bool point_in_range(float x, float y, float z);
        torch::Tensor box_corner_to_center(torch::Tensor box);
        void publish_markers(torch::Tensor boxes);
};