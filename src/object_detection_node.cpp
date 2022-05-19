#include <ros/ros.h>
#include "ObjectDetection.h"

int main (int argc, char** argv)
{
    // Initialize ROS
    ros::init (argc, argv, "dl_detection");
    ObjectDetection detector;
    // Spin
    ros::spin ();
    return 0;
}
