#include "ObjectDetection.h"

ObjectDetection::ObjectDetection(ros::NodeHandle * nh)
{
    pcl_sub = nh->subscribe ("/velodyne_points", 1, &ObjectDetection::cloud_cb, this);
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
    float* data = voxel.data_ptr<float>();

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

            //voxel[0][z_idx][y_idx][x_idx] = 1;
            *(data + z_idx * y_size * x_size + y_idx * x_size + x_idx) = 1;
        }
    }
    return voxel.resize_({1, z_size, y_size, x_size});
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
torch::Tensor ObjectDetection::box_corner_to_center(torch::Tensor box, const tf::Transform &transform)
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

    // for(int i = 2; i < 10; i+=2)
    // {
    //     geometry_msgs::Point point;
    //     point.x = box[i].item().to<float>();
    //     point.y = box[i + 1].item().to<float>();
    //     point.z = 0;

    //     geometry_msgs::Point transformed_point = transform_point(point, transform);
    //     box[i] = point.x;
    //     box[i + 1] = point.y;
    // }

    float l = (sqrt(pow(bl_x - tl_x, 2) + pow(bl_y - tl_y, 2)) + sqrt(pow(br_x - tr_x, 2) + pow(br_y - tr_y, 2))) / 2;
    float w = (sqrt(pow(bl_x - br_x, 2) + pow(bl_y - br_y, 2)) + sqrt(pow(tl_x - tr_x, 2) + pow(tl_y - tr_y, 2))) / 2;

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
    std::string lidar_frame_id = cloud_msg->header.frame_id;
    std::string map_frame_id = "base_link";
	ros::Time stamp = cloud_msg->header.stamp;

    tf::StampedTransform transform;
    tf::TransformListener tf_listener;
    //std::cout<<"just before transform"<<std::endl;
	// try{
	// 	tf_listener.waitForTransform(lidar_frame_id, map_frame_id, stamp, ros::Duration(0.1));
	// 	tf_listener.lookupTransform(lidar_frame_id, map_frame_id, cloud_msg->header.stamp, transform);
	// }
	// catch (tf::TransformException& ex) {
	// 	ROS_INFO_STREAM(ex.what());
	// 	return;
	// }
    //std::cout<<"passed transform"<<std::endl;
    pcl::fromROSMsg(*cloud_msg, pointcloud);
    //ros::WallTime start = ros::WallTime::now();
    torch::Tensor voxel = pcl_to_voxel();
    //ros::WallTime end = ros::WallTime::now();
    //std::cout<<"time elapsed: "<< end.toSec() - start.toSec()<<std::endl;
    if(use_gpu)
    {
        voxel = voxel.to(torch::kCUDA);
    }

    torch::NoGradGuard no_grad_;

    std::vector<torch::jit::IValue> input = {voxel, x_min, y_min, x_res, y_res, score_threshold};
    
    
    // output of model is Nx10 tensor with N boxes and [cls, score and bot left, bot right, top right, top left corners]
    auto pred = model.forward(input).toTensor();
    float * pred_data = pred.data_ptr<float>();
    int num_boxes = pred.size(0);
    int classes[num_boxes];
    float scores[num_boxes];
    float corners[num_boxes][8];

    for(int i = 0; i < num_boxes; i++)
    {
        classes[i] = (int)(*(pred_data + i * 10 + 0));
        scores[i] = *(pred_data + i * 10 + 1);

        for(int j = 2; j < 10; j++)
        {
            corners[i][j-2] = *(pred_data + i * 10 + j);
        }
    }

    std::vector<int> indexes = non_max_supression(pred);
    
    
    robot_msgs::DetectedObjectArray detected_objects;
	detected_objects.header.stamp = stamp;
	detected_objects.header.frame_id = map_frame_id;

    for(int i = 0; i < indexes.size(); i++)
    {
        torch::Tensor box = pred[indexes[i]];
        robot_msgs::DetectedObject detected_object;
	    detected_object.header = detected_objects.header;
        detected_object.label = object_classes[box[0].item().to<int>()];
        detected_object.score = box[1].item().to<float>();
        detected_object.space_frame = map_frame_id;
        detected_object.pose_reliable = true;


        torch::Tensor new_box = box_corner_to_center(box, transform);
        geometry_msgs::Pose pose;
        pose.position.x = new_box[0].item().to<float>();
        pose.position.y = new_box[1].item().to<float>();

        tf2::Quaternion quat;
        quat.setRPY(0, 0, -new_box[4].item().to<float>());
        quat = quat.normalize();

        pose.orientation.x = quat.x();
        pose.orientation.y = quat.y();
        pose.orientation.z = quat.z();
        pose.orientation.w = quat.w();

        geometry_msgs::Vector3 dim;
        dim.x = new_box[2].item().to<float>();
        dim.y = new_box[3].item().to<float>();
        dim.z = 2.0;

        detected_object.pose = pose;
        detected_object.dimensions = dim;

        detected_objects.objects.push_back(detected_object);
    }

    objects_pub.publish(detected_objects);
    // publish_markers(pred, indexes);

    // sensor_msgs::PointCloud2 msg;
    // msg.header = cloud_msg->header;
    // msg.header.frame_id = "/map";
    // msg.height = cloud_msg->height;
    // msg.width = cloud_msg->width;
    // msg.fields = cloud_msg->fields;
    // msg.is_bigendian = cloud_msg->is_bigendian;
    // msg.point_step = cloud_msg->point_step;
    // msg.row_step = cloud_msg->row_step;
    // msg.data = cloud_msg->data;
    // msg.is_dense = cloud_msg->is_dense;

    // pcl_pub.publish(msg);
}

std::vector<int> ObjectDetection::non_max_supression(torch::Tensor pred)
{
    typedef boost::geometry::model::d2::point_xy<double> point_type;
    typedef boost::geometry::model::polygon<point_type> polygon_type;

    polygon_type poly[pred.size(0)];

    // convert bboxes to boost polygons
    for(int i = 0; i < pred.size(0); i++)
    {
        for(int j = 0; j < 4; j++)
        {
            float x = pred[i][2 * j + 2].item().to<float>();
            float y = pred[i][2 * j + 3].item().to<float>();
            boost::geometry::append(poly[i], boost::geometry::make<point_type>(x, y));
        }
        float x = pred[i][2].item().to<float>();
        float y = pred[i][3].item().to<float>();
        boost::geometry::append(poly[i], boost::geometry::make<point_type>(x, y));
        boost::geometry::correct(poly[i]);
    }
    
    // get sorted indices of scores
    std::vector<float> scores;
    for(int i = 0; i < pred.size(0); i++)
    {
        scores.push_back(pred[i][1].item().to<float>());
    }
    std::vector<int> idxs = get_sorted_indexes(scores);

    std::vector<int> selected;
    while(idxs.size() > 0)
    {
        // pick top box and add its index to the list
        int index = idxs[0];
        selected.push_back(index);
        idxs.erase(idxs.begin() + 0);

        // calculate iou of current box with each other box
        for(int i = 0; i < idxs.size(); i++)
        {
            std::vector<polygon_type> intersection;
            std::vector<polygon_type> uni;
            boost::geometry::intersection(poly[index], poly[idxs[i]], intersection);
            boost::geometry::union_(poly[index], poly[idxs[i]], uni);
            float iou;

            if(intersection.size() > 1)
            {
                std::cout<<"WARNING: intersection is not single polygon"<<std::endl;
            }

            if(intersection.size() == 0)
            {
                iou = 0;
            }
            else
            {
                iou = boost::geometry::area(intersection[0]) / boost::geometry::area(uni[0]);
            }
            if(iou > iou_threshold)
            {
                idxs.erase(idxs.begin() + i);
                i--;
            }
        }
    }
    return selected;    
}

std::vector<int> ObjectDetection::get_sorted_indexes(std::vector<float> scores)
{
    std::vector<int> idx(scores.size());
    iota(idx.begin(), idx.end(), 0);
    stable_sort(idx.begin(), idx.end(), [&scores](size_t i1, size_t i2) {return scores[i1] > scores[i2];});

    return idx;
}

geometry_msgs::Point ObjectDetection::transform_point(const geometry_msgs::Point &in_point, const tf::Transform &in_transform)
{
    tf::Point tf_point;
    tf::pointMsgToTF(in_point, tf_point);

    tf_point = in_transform * tf_point;
    geometry_msgs::Point geometry_point;
    tf::pointTFToMsg(tf_point, geometry_point);

    return geometry_point;
}

// used only for visualizaiton
void ObjectDetection::publish_markers(torch::Tensor boxes, std::vector<int> indexes)
{
    visualization_msgs::MarkerArray object_boxes;
    int marker_id = 0;
    for (int i = 0; i < indexes.size(); i++)
    {
        int index = indexes[i];
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
            p.x =  boxes[index][2 * j + 2].item().to<float>(); 
            p.y = boxes[index][2 * j + 3].item().to<float>();
            p.z = 0;
            points.push_back(p);
        }
        for (int j = 0; j < 4; j++)
        {
            geometry_msgs::Point p;
            p.x =  boxes[index][2 * j + 2].item().to<float>(); 
            p.y = boxes[index][2 * j + 3].item().to<float>();
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


