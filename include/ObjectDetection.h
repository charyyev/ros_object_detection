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
#include <tf/transform_listener.h>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include <numeric>    
#include <algorithm>
#include <cmath>

class ObjectDetection
{
    private:
        ros::NodeHandle nh;
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
        const float iou_threshold = 0.1;
        
        bool use_gpu = true;

        std::string object_classes[4] = {"background", "car", "pedestrian", "cyclist"};

        float box_z_bot = -2.0;
        float box_z_top = 0.0;

        //tf::TransformListener tf_listener;

    public:
        ObjectDetection();
        void cloud_cb(const sensor_msgs::PointCloud2ConstPtr& input);
        torch::Tensor pcl_to_voxel();
        bool point_in_range(float x, float y, float z);
        void box_corner_to_center(float corners[8], float box[]);
        void publish_markers(torch::Tensor boxes, std::vector<int> indexes);
        std::vector<int> non_max_supression(float * scores, float corners[][8], int num_boxes);
        std::vector<int> get_sorted_indexes(float * scores, int length);
        geometry_msgs::Point transform_point(const geometry_msgs::Point &in_point, const tf::Transform &in_transform);
};