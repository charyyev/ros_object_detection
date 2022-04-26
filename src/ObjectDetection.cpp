#include "ObjectDetection.h"

ObjectDetection::ObjectDetection(ros::NodeHandle * nh)
{
    pcl_sub = nh->subscribe ("/points_raw", 1, &ObjectDetection::cloud_cb, this);
    objects_pub = nh->advertise<robot_msgs::DetectedObjectArray>("/detection/lidar_detector/objects", 1);
    box_pub = nh->advertise<visualization_msgs::MarkerArray>("/bounding_boxes", 1);
    pcl_pub = nh->advertise<sensor_msgs::PointCloud2> ("/points", 1);
    
    model = torch::jit::load("/home/stpc/models/pixor.pt");
    if(use_gpu)
    {
        model.to(torch::kCUDA);
    }
    model.eval();
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


// box is tensor with shape{10} corresponding to class, score and 4 corners [bot left, bot right, top right, top left]
// return tensor with shape{5} corresponding to x, y, l, w, yaw 
torch::Tensor ObjectDetection::box_corner_to_center(torch::Tensor box)
{
    torch::Tensor output = torch::zeros(5);
    float bl_x = box[2].item().to<float>();
    float bl_y = box[3].item().to<float>();
    float br_x = box[4].item().to<float>();
    float br_y = box[5].item().to<float>();
    float tr_x = box[6].item().to<float>();
    float tr_y = box[7].item().to<float>();
    float tl_x = box[8].item().to<float>();
    float tl_y = box[9].item().to<float>();

    float l = (sqrt(pow(bl_x - tl_x, 2) + pow(bl_y - tl_y, 2)) + sqrt(pow(br_x - tr_x, 2) + pow(br_y - tr_y, 2))) / 4;
    float w = (sqrt(pow(bl_x - br_x, 2) + pow(bl_y - br_y, 2)) + sqrt(pow(tl_x - tr_x, 2) + pow(tl_y - tr_y, 2))) / 4;

    float x = (bl_x + br_x + tr_x + tl_x) / 4;
    float y = (bl_y + br_y + tr_y + tl_y) / 4;

    float yaw = (atan2(tr_x - br_x, tr_y - br_y) + atan2(tl_x - bl_x, tl_y - bl_y)) / 2;
    yaw -= M_PI / 2;

    output[0] = x;
    output[1] = y;
    output[2] = l;
    output[3] = w;
    output[4] = yaw;
    
    return output;
}

void ObjectDetection::cloud_cb(const sensor_msgs::PointCloud2ConstPtr& cloud_msg)
{
    //cloud_msg->header.frame_id = "/map";
    std::string frame_id = cloud_msg->header.frame_id;
	ros::Time stamp = cloud_msg->header.stamp;

    pcl::fromROSMsg(*cloud_msg, pointcloud);
    torch::Tensor voxel = pcl_to_voxel();
    if(use_gpu)
    {
        voxel.to(torch::kCUDA);
    }

    torch::NoGradGuard no_grad_;

    std::vector<torch::jit::IValue> input = {voxel, x_min, y_min, x_res, y_res, score_threshold};
    
    // output of model is Nx10 tensor with N boxes and [cls, score and bot left, bot right, top right, top left corners]
    auto pred = model.forward(input).toTensor();
    
    // robot_msgs::DetectedObjectArray detected_objects;
	// detected_objects.header.stamp = stamp;
	// detected_objects.header.frame_id = frame_id;

    // for(int i = 0; i < pred.size(0); i++)
    // {
    //     torch::Tensor box = pred[i];
    //     robot_msgs::DetectedObject detected_object;
	//     detected_object.header = detected_objects.header;
    //     detected_object.label = object_classes[box[0].item().to<int>()];
    //     detected_object.score = box[1].item().to<float>();
    //     detected_object.space_frame = frame_id;

    //     torch::Tensor new_box = box_corner_to_center(box);
    //     geometry_msgs::Pose pose;
    //     pose.position.x = new_box[0].item().to<float>();
    //     pose.position.y = new_box[1].item().to<float>();

    //     tf2::Quaternion quat;
    //     quat.setRPY(0, 0, new_box[4].item().to<float>());
    //     quat = quat.normalize();

    //     pose.orientation.x = quat.x();
    //     pose.orientation.y = quat.y();
    //     pose.orientation.z = quat.z();
    //     pose.orientation.w = quat.w();

    //     geometry_msgs::Vector3 dim;
    //     dim.x = new_box[2].item().to<float>();
    //     dim.y = new_box[3].item().to<float>();
    //     dim.z = 2.0;

    //     detected_object.pose = pose;
    //     detected_object.dimensions = dim;

    //     detected_objects.objects.push_back(detected_object);
    // }

    // objects_pub.publish(detected_objects);
    publish_markers(pred);

    sensor_msgs::PointCloud2 msg;
    msg.header = cloud_msg->header;
    msg.header.frame_id = "/map";
    msg.height = cloud_msg->height;
    msg.width = cloud_msg->width;
    msg.fields = cloud_msg->fields;
    msg.is_bigendian = cloud_msg->is_bigendian;
    msg.point_step = cloud_msg->point_step;
    msg.row_step = cloud_msg->row_step;
    msg.data = cloud_msg->data;
    msg.is_dense = cloud_msg->is_dense;

    pcl_pub.publish(msg);
}

// used only for visualizaiton
void ObjectDetection::publish_markers(torch::Tensor boxes)
{
    visualization_msgs::MarkerArray object_boxes;
    int marker_id = 0;
    for (int i = 0; i < boxes.size(0); i++)
    {
        visualization_msgs::Marker box;
        //box.lifetime = ros::Duration(marker_display_duration_);
        box.header.frame_id = "/map";

        box.ns = "box_markers";
        box.id = marker_id++;
        box.scale.x = 0.02;
        box.color.a = 1.0;
        box.color.r = 1.0;
        box.color.g = 0.0;
        box.color.b = 0.0;
        box.type = visualization_msgs::Marker::LINE_LIST;
        

        std::vector<geometry_msgs::Point> points;
        for (int j = 0; j < 4; j++)
        {
            geometry_msgs::Point p;
            p.x =  boxes[i][2 * j + 2].item().to<float>(); 
            p.y = boxes[i][2 * j + 3].item().to<float>();
            p.z = 0;
            points.push_back(p);
        }
        for (int j = 0; j < 4; j++)
        {
            geometry_msgs::Point p;
            p.x =  boxes[i][2 * j + 2].item().to<float>(); 
            p.y = boxes[i][2 * j + 3].item().to<float>();
            p.z = -2;
            points.push_back(p);
        }

        //mid surface
        box.points.push_back(points[0]);
        box.points.push_back(points[4]);

        box.points.push_back(points[1]);
        box.points.push_back(points[5]);

        box.points.push_back(points[2]);
        box.points.push_back(points[6]);

        box.points.push_back(points[3]);
        box.points.push_back(points[7]);

        //upface
        box.points.push_back(points[0]);
        box.points.push_back(points[1]);

        box.points.push_back(points[1]);
        box.points.push_back(points[2]);

        box.points.push_back(points[2]);
        box.points.push_back(points[3]);

        box.points.push_back(points[0]);
        box.points.push_back(points[3]);
        //downface
        box.points.push_back(points[4]);
        box.points.push_back(points[5]);

        box.points.push_back(points[5]);
        box.points.push_back(points[6]);

        box.points.push_back(points[6]);
        box.points.push_back(points[7]);

        box.points.push_back(points[4]);
        box.points.push_back(points[7]);

        object_boxes.markers.push_back(box);
    }
    box_pub.publish(object_boxes);
}


