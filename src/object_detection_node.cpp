#include <ros/ros.h>
#include "ObjectDetection.h"

int main (int argc, char** argv)
{
    // Initialize ROS
    ros::init (argc, argv, "dl_detection");
    ros::NodeHandle nh("~");
    ObjectDetection detector = ObjectDetection(&nh);
    // Spin
    ros::spin ();
}