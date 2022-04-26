#include "ObjectDetection.h"


ObjectDetection::ObjectDetection(ros::NodeHandle * nh)
{
    pcl_sub = nh->subscribe ("/points_raw", 1, &ObjectDetection::cloud_cb, this);
    objects_pub = nh->advertise<robot_msgs::DetectedObjectArray>("/detection/dl_detector/objects", 1);
    model = torch::jit::load("/home/stpc/models/pixor.pt");
}

torch::Tensor ObjectDetection::pcl_to_voxel()
{
    int x_size = (int)((x_max - x_min) / x_res);
    int y_size = (int)((y_max - y_min) / y_res);
    int z_size = (int)((z_max - z_min) / z_res);

    torch::Tensor voxel = torch::zeros({1, z_size, y_size, x_size});

    for (pcl::PointCloud<pcl::PointXYZ>::const_iterator item = pointcloud.begin(); item != pointcloud.end(); item++)
    {
        float x = item->x;
        float y = item->y;
        float z = item->z;

        if(point_in_range(x, y, z))
        {
            int x_idx = (int)((x - x_min) / x_res);
            int y_idx = (int)((y - y_min) / y_res);
            int z_idx = (int)((z - z_min) / z_res);

            voxel[0][z_idx][y_idx][x_idx] = 1;
        }
    }
    std::cout<<voxel.sizes()<<std::endl;
    return voxel;
}

pcl::PointCloud<pcl::PointXYZ> ObjectDetection::voxel_to_pcl(torch::Tensor voxel)
{
    pcl::PointCloud<pcl::PointXYZ> cloud;

    for(int z_idx = 0; z_idx < voxel.size(0); z_idx++)
    {
        for(int y_idx = 0; y_idx < voxel.size(1); y_idx++)
        {
            for(int x_idx = 0; x_idx < voxel.size(2); x_idx++)
            {
                int val = voxel[z_idx][y_idx][x_idx].item().to<int>();
                if(val == 1)
                {
                    float c_x = x_idx + x_res / 2;
                    float c_y = y_idx + y_res / 2;
                    float c_z = z_idx + z_res / 2;

                    float x = c_x * x_res + x_min;
                    float y = c_y * y_res + y_min;
                    float z = c_z * z_res + z_min;

                    cloud.push_back(pcl::PointXYZ (x, y, z));
                }
            }
        }
    }

    return cloud;
}

bool ObjectDetection::point_in_range(float x, float y, float z)
{
    bool in_x_range = false;
    bool in_y_range = false;
    bool in_z_range = false;

    if(x_min < x && x < x_max)
    {
        in_x_range = true;
    }
    if(y_min < y && y < y_max)
    {
        in_y_range = true;
    }
    if(z_min < z && z < z_max)
    {
        in_z_range = true;
    }

    return in_x_range && in_y_range && in_z_range;
}

void ObjectDetection::cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
    pcl::fromROSMsg(*cloud_msg, pointcloud);
    torch::Tensor voxel = pcl_to_voxel();
    if(use_gpu)
    {
        model.to(torch::kCUDA);
        voxel.to(torch::kCUDA);
    }

    model.eval();
    torch::NoGradGuard no_grad_;

    std::vector<torch::jit::IValue> input = {voxel, x_min, y_min, x_res, y_res, threshold};
    
    auto output = model.forward(input).toTensor();
    std::cout<<output<<std::endl;
}
