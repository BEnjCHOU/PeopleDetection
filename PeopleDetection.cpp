#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <thread>
//math
#include <math.h>
//#include <vector>
#include <pcl/point_types.h>
//#include <boost/asio.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/search/search.h>
#include <pcl/segmentation/segment_differences.h>
#include <pcl/filters/radius_outlier_removal.h>
//#include <pcl/kdtree/kdtree_flann.h>
//#include <pcl/point_cloud.h>
#include <pcl/segmentation/extract_clusters.h>
// write file
#include <fstream>
//pcl person libraries
#include <pcl/people/head_based_subcluster.h>
#include <pcl/people/person_cluster.h>
#include <pcl/features/moment_of_inertia_estimation.h>
//pcl io
#include <pcl/io/pcd_io.h>
// downsampling
#include <pcl/filters/voxel_grid.h>
// centroid
#include <pcl/common/centroid.h>

// background const variable
const int BACKGROUND = 1;
// frame cont variable
int START_FRAME = 2;
const int END_FRAME = 18815;
// cluster const variable
const double RADIUS_CONDITION = 0.1;
const int MIN_CLUSTER_POINTS = 5;
const int MAX_CLUSTER_POINTS = 1000;
// color density variable
const int DENSITY = 230;
//cluster same check variables
const float DISTANCE_SAME_MIN = 0.03;
const float DISTANCE_SAME_MAX = 0.25;
const float Z_SAME_MAX = 0.1;
const float Z_SAME_MIN = 0.01;
const float Y_SAME_MAX = 0.1;
const float Y_SAME_MIN = 0.25;
const float X_SAME_MAX = 0.1;
const float X_SAME_MIN = 0.25;
// detection variable
const int DETECTION_VARIABLE = 1;
// structure to put high and low z coordinate point
struct high_low_point
{
    int index_id;
    float center_x;
    float center_y;
    float center_z;
    //float low_z;
    float distance;
};

// read file function and return pointcloud
pcl::PointCloud<pcl::PointXYZ>::Ptr ReadFilePointCloud(int frame)
{
    std::string source = "../17_02_2019_14_46_01/M8-192.168.1.9/frame-";
    std::string file_type = ".txt";
    std::string file = source + std::to_string(frame) + file_type;
    std::ifstream infile(file);
    // pcl output
    std::vector<pcl::PointCloud<pcl::PointXYZ>> output;
    // pcl statements
    pcl::PointCloud<pcl::PointXYZ>::Ptr DefaultData (new pcl::PointCloud<pcl::PointXYZ>);
    // read file
    for(std::string line; std::getline(infile, line);)
    {
        // create structure
        pcl::PointXYZ Points;
        std::istringstream in (line);
        in >> Points.x >> Points.y >> Points.z;
        DefaultData->push_back(Points);
    }
    //std::cout << file << std::endl;
    return DefaultData;
}

// pcl save point cloud as pcd file
void saveToPcd( pcl::PointCloud< pcl::PointXYZ >::Ptr cloud, int counting )
{
    for(int i = 0 ; i < counting ; ++i)
    {
        cloud->width = cloud->points.size ();
        cloud->height = 1;
        cloud->is_dense = true;
        std::string filename;
        filename = "cloud_cluster-" + std::to_string(i) + ".pcd";
        pcl::io::savePCDFileASCII (filename, *cloud);
    }
}

// DownSampling
pcl::PointCloud < pcl::PointXYZ >::Ptr DownSampling (pcl::PointCloud<pcl::PointXYZ>::Ptr data)
{
    // Create the filtering object
    pcl::PointCloud < pcl::PointXYZ >::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::VoxelGrid<pcl::PointXYZ> sor;
    sor.setInputCloud (data);
    sor.setLeafSize (0.07f, 0.07f, 0.07f);
    sor.filter (*cloud_filtered);
    return cloud_filtered;
}

// background subtractor and return filter data
pcl::PointCloud<pcl::PointXYZ>::Ptr RemoveBackground(pcl::PointCloud<pcl::PointXYZ>::Ptr background, pcl::PointCloud<pcl::PointXYZ>::Ptr frontground)
{
    // output statement
    pcl::PointCloud<pcl::PointXYZ>::Ptr output (new pcl::PointCloud<pcl::PointXYZ>);
    // search method Kdtree
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    // segmentation statement
    pcl::SegmentDifferences<pcl::PointXYZ> sdiff;
    // Target cloud statement
    sdiff.setTargetCloud(background);
    // Input cloud statement
    sdiff.setInputCloud(frontground);
    // search
    sdiff.setSearchMethod(tree);
    // Distance to the model threshold
    sdiff.setDistanceThreshold(0.1);
    // segmentation ouput
    sdiff.segment(*output);
    return output;
}

// get cluster indices
std::vector < pcl::PointIndices > GetClusterIndices(pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudData)
{
    // put all the cluster index in it, example: cluster_indices[0] contains all the index for the first cluster
    std::vector <pcl::PointIndices> cluster_indices;
    // kdTree search statement
    pcl::search::KdTree< pcl::PointXYZ >::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    // input cloud to kdtree for searching
    tree->setInputCloud (PointCloudData);
    // cluster extraction statement
    pcl::EuclideanClusterExtraction < pcl::PointXYZ > ec;
    // cluster conditions
    ec.setClusterTolerance (RADIUS_CONDITION);
    ec.setMinClusterSize (MIN_CLUSTER_POINTS);
    ec.setMaxClusterSize (MAX_CLUSTER_POINTS);
    ec.setSearchMethod (tree);
    ec.setInputCloud (PointCloudData);
    ec.extract (cluster_indices);
    return cluster_indices;
}

// cluster grouping to pointcloud vector
std::vector< pcl::PointCloud< pcl::PointXYZ > > ClusterGrouping(pcl::PointCloud<pcl::PointXYZ>::Ptr DefaultData)
{
    // vector in which include every cluster one by one
    std::vector< pcl::PointCloud< pcl::PointXYZ > > output;
    // put all the cluster index in it, example: cluster_indices[0] contains all the index for the first cluster
    std::vector< pcl::PointIndices > cluster_indices;
    // kdTree search statement
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
    // input cloud to kdtree for searching
    tree->setInputCloud (DefaultData);
    // cluster extraction statement
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    // cluster conditions
    ec.setClusterTolerance (RADIUS_CONDITION);
    ec.setMinClusterSize (MIN_CLUSTER_POINTS);
    ec.setMaxClusterSize (MAX_CLUSTER_POINTS);
    ec.setSearchMethod (tree);
    ec.setInputCloud (DefaultData);
    ec.extract (cluster_indices);
    // retrieve data by indices(index)
    for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it)
    {
        // create cloud cluster for all point cloud in a cluster
        pcl::PointCloud<pcl::PointXYZ> cloud_cluster;
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
        {
            cloud_cluster.points.push_back (DefaultData->points[*pit]);
        }
        output.push_back(cloud_cluster);
    }
    return output;
}

// draw bounding box
/*void DrawBoundingBox(pcl::PointCloud< pcl::PointXYZ >::Ptr cloud, pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor)
{
    // draw bounding box
    /pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
    feature_extractor.setInputCloud(cloud_cluster);
    feature_extractor.compute();
    pcl::PointXYZ min_point_AABB;
    pcl::PointXYZ max_point_AABB;
    feature_extractor.getAABB (min_point_AABB, max_point_AABB);
    vis.addCube (min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 1.0, 0.0, std::to_string(box_id));
    vis.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, std::to_string(box_id));
}*/
// procedure
/*
    get
*/
// Person cluster detection
std::vector< std::vector < high_low_point > > PersonCluster( std::vector<pcl::PointIndices> normalCluster, pcl::PointCloud<pcl::PointXYZ>::Ptr DefaultData )
{
    // vector to save structure
    std::vector < high_low_point > struc_high_low_distance_z;
    // cluster index
    int cluster_index = 0;
    // put indices to pointcloud and get high low z coordinate point struc to vector
    for(std::vector< pcl::PointIndices >::const_iterator it = normalCluster.begin (); it != normalCluster.end (); ++it)
    {
        // point cloud statement to save indices
        pcl::PointCloud < pcl::PointXYZ >::Ptr ClusterPointCloud (new pcl::PointCloud<pcl::PointXYZ>);
        // find Center Point
        //pcl::PointXYZ center;
        //pcl::compute3DCentroid(*it, center);
        //float CenterPoint_x = center.x;
        //float CenterPoint_y = center.y;
        //float CenterPoint_z = center.z;
        float CenterPoint_x = 0.0;
        float CenterPoint_y = 0.0;
        float CenterPoint_z = 0.0;
        // number of point
        int number_points = 1;
        for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit)
        {
            //ClusterPointCloud->points.push_back (DefaultData->points[*pit]);
            CenterPoint_x += DefaultData->points[*pit].x;
            CenterPoint_y += DefaultData->points[*pit].y;
            CenterPoint_z += DefaultData->points[*pit].z;
            number_points++;
        }
        // extractor estimation
        //pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
        //feature_extractor.setInputCloud(ClusterPointCloud);
        //feature_extractor.compute();
        //AABB statements
        //pcl::PointXYZ min_point_AABB;
        //pcl::PointXYZ max_point_AABB;
        // get AABB max min points
        //feature_extractor.getAABB (min_point_AABB, max_point_AABB);
        
        // Devide center point to number of point
        CenterPoint_x = CenterPoint_x/number_points;
        CenterPoint_y = CenterPoint_y/number_points;
        CenterPoint_z = CenterPoint_z/number_points;
        // structure statements
        high_low_point same;
        same.index_id = cluster_index;
        same.center_x = CenterPoint_x;
        same.center_y = CenterPoint_y;
        same.center_z = CenterPoint_z;
        //point_z.low_z = min_point_AABB.z;
        same.distance = sqrt((CenterPoint_x * CenterPoint_x) + (CenterPoint_y * CenterPoint_y) + (CenterPoint_z * CenterPoint_z));
        struc_high_low_distance_z.push_back(same);
        // add frame
        cluster_index++;
    }
    // person cluster statements
    std::vector< std::vector < high_low_point > > detectCluster;
    // find near cluster
    while(struc_high_low_distance_z.size() > 1)
    {
        // put first cluster
        bool have_neighbor = false;
        // save indices that are together
        std::vector < high_low_point > cluster_together;
        // save indices that are not together
        std::vector < high_low_point > cluster_not_together;
        // compute
        int frame_counter_check = 0;
        for(std::vector < high_low_point >::iterator it = struc_high_low_distance_z.begin(); it != struc_high_low_distance_z.end(); ++it)
        {
            // check if same quadrant
            //bool same_quadrant = false;
            // first cluster to check z coordinates
            float distance_to_check = struc_high_low_distance_z.begin()->distance;
            float x_to_check = struc_high_low_distance_z.begin()->center_x;
            float y_to_check = struc_high_low_distance_z.begin()->center_y;
            float z_to_check = struc_high_low_distance_z.begin()->center_z;
            // rest cluster to check with first cluster ; check distance and z coordinate
            int next_cluster = frame_counter_check + 1;
            float distance_check = struc_high_low_distance_z[next_cluster].distance;
            float x_check = struc_high_low_distance_z[next_cluster].center_x;
            float y_check = struc_high_low_distance_z[next_cluster].center_y;
            float z_check = struc_high_low_distance_z[next_cluster].center_z;
            // substraction to check
            float distance_substract = distance_to_check - distance_check;
            float z_substract = z_to_check - z_check;
            float y_substract = y_to_check - y_check;
            float x_substract = x_to_check - x_check;
            // to positive
            if(distance_substract < 0.0)
                distance_substract = fabsf(distance_substract);
            if (z_substract < 0.0)
                z_substract = fabsf(z_substract);
            if (y_substract < 0.0)
                y_substract = fabsf(y_substract);
            if (x_substract < 0.0)
                x_substract = fabsf(x_substract);
            // check if are same cluster
            if( distance_substract < DISTANCE_SAME_MAX && z_substract > Z_SAME_MIN && y_substract < Y_SAME_MIN && x_substract < X_SAME_MIN)
            {
                // add indice position to vector
                cluster_together.push_back(struc_high_low_distance_z[next_cluster]);
                have_neighbor = true;
            }
            // put first cluster if true
            if( have_neighbor == true && it == struc_high_low_distance_z.end())
            {
                // add checked cluster
                cluster_together.push_back(struc_high_low_distance_z[0]);
            }
            else
            {
                cluster_not_together.push_back(*it);
            }
            frame_counter_check++;
        }
        // remove together cluster
        if( have_neighbor == true )
        {
            if(cluster_together.size() > DETECTION_VARIABLE)
                detectCluster.push_back(cluster_together);
            // put not together back to for loop
            int size_of_last_frame = struc_high_low_distance_z.size();
            struc_high_low_distance_z.clear();
            for(std::vector < high_low_point >::iterator it = cluster_not_together.begin() ; it != cluster_not_together.end() ; ++it)
            {
                struc_high_low_distance_z.push_back(*it);
            }
            if(struc_high_low_distance_z.size() == size_of_last_frame)
                struc_high_low_distance_z.erase(struc_high_low_distance_z.begin());
        }
        else
        {
            struc_high_low_distance_z.erase(struc_high_low_distance_z.begin());
        }
        //std::cout << struc_high_low_distance_z.size() << std::endl;
    }
    return detectCluster;
}

int main()
{
    //vis statements
    pcl::visualization::PCLVisualizer vis("3D View");
    // add coordinate system
    //vis.addCoordinateSystem(100);
    //pcl statements
    pcl::PointCloud<pcl::PointXYZ>::Ptr Background (new pcl::PointCloud<pcl::PointXYZ>);
    // read file
    Background = ReadFilePointCloud(BACKGROUND);
    // visualization statements
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> blue(Background,0,0,255);
    // box id
    int box_id = 0;
    std::vector < int > set_box_id;
    while(!vis.wasStopped())
    {
        // remove point cloud
        vis.removeAllPointClouds();
        vis.removeAllShapes();
        // read front ground
        pcl::PointCloud<pcl::PointXYZ>::Ptr Frontground (new pcl::PointCloud<pcl::PointXYZ>);
        Frontground = ReadFilePointCloud(START_FRAME);
        // downsampling
        Frontground = DownSampling(Frontground);
        // remove background
        pcl::PointCloud<pcl::PointXYZ>::Ptr AfterSegmentation (new pcl::PointCloud<pcl::PointXYZ>);
        AfterSegmentation = RemoveBackground(Background, Frontground);
        // get cluster statement grounping ; output vector conataining sack of poincloud
        std::vector<pcl::PointCloud<pcl::PointXYZ> > input_cluster_cloud;
        input_cluster_cloud = ClusterGrouping(AfterSegmentation);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(AfterSegmentation,0,255,0);
        /*person detection*/
        //get cluster index
        std::vector< pcl::PointIndices > cluster_indices;
        cluster_indices = GetClusterIndices(AfterSegmentation);
        // index of cluster that are grouped
        std::vector < std::vector < high_low_point > > index_grouped_cluster;
        // detect person cluster and regrouping
        //std::cout << "Hello" << std::endl;
        index_grouped_cluster = PersonCluster(cluster_indices, AfterSegmentation);
        //std::cout << "Bye" << std::endl;
        std::cout << std::to_string(index_grouped_cluster.size()) << std::endl;
        if(!index_grouped_cluster.empty())
        {
            for(std::vector < std::vector < high_low_point > >::iterator it = index_grouped_cluster.begin() ; it != index_grouped_cluster.end() ; ++it)
            {
                pcl::PointCloud<pcl::PointXYZ>::Ptr person_cloud (new pcl::PointCloud<pcl::PointXYZ>);
                // retrieve point cloud from indices
                if( it->size() > 0 )
                {
                    for(std::vector < high_low_point >::iterator ti = it->begin() ; ti != it->end() ; ++ti)
                    {
                        //std::cout << "ByeBye" << std::endl;
                        for(std::vector < int >::const_iterator pit = cluster_indices[ti->index_id].indices.begin () ; pit != cluster_indices[ti->index_id].indices.end () ; ++pit)
                        {
                            //std::cout << "ByeByeBye" << std::endl;
                            person_cloud->points.push_back( AfterSegmentation->points[*pit] );
                            //std::cout << "ByeByeBye1" << std::endl;
                        }
                    }
                }
                set_box_id.push_back(box_id);
                std::cout << std::to_string(person_cloud->size()) << std::endl;
                //draw bounding box;
                pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
                feature_extractor.setInputCloud(person_cloud);
                feature_extractor.compute();
                pcl::PointXYZ min_point_AABB;
                pcl::PointXYZ max_point_AABB;
                feature_extractor.getAABB (min_point_AABB, max_point_AABB);
                vis.addCube (min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 1.0, 0.0, std::to_string(box_id));
                vis.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, std::to_string(box_id));
                box_id++;
            }
        }
        //std::cout << "OK" << std::endl;
        // person cluster
        //std::vector <pcl::people::PersonCluster <pcl::PointXYZ> > clusters = PersonCluster(START_FRAME);
        // render people
        /*unsigned int k = 0;
        for(std::vector<pcl::people::PersonCluster<pcl::PointXYZ> >::iterator it = clusters.begin(); it != clusters.end(); ++it)
        {
            // draw theoretical person bounding box in the PCL viewer:
            it->drawTBoundingBox(vis, k);
            k++;
        }
        std::cout << "people found : " << k << std::endl;*/
        
        //  visualizer color statement red for people > 20 , green for people < 20
        /*if(input_cluster_cloud.size() > DENSITY)
        {
            pcl::PointCloud< pcl::PointXYZ >::Ptr cloud_cluster (new pcl::PointCloud< pcl::PointXYZ >);
            for(std::vector<pcl::PointCloud< pcl::PointXYZ > >::iterator it = input_cluster_cloud.begin (); it != input_cluster_cloud.end (); ++it)
            {
                for(pcl::PointCloud< pcl::PointXYZ >::iterator pit = it->points.begin(); pit!= it->points.end(); ++pit)
                {
                    cloud_cluster->points.push_back(*pit);
                }
            }
            pcl::visualization::PointCloudColorHandlerCustom< pcl::PointXYZ > red(cloud_cluster,255,0,0);
            vis.addPointCloud(cloud_cluster,red,"cloud_cluster",0);
            // draw bounding box
            pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
            feature_extractor.setInputCloud(cloud_cluster);
            feature_extractor.compute();
            pcl::PointXYZ min_point_AABB;
            pcl::PointXYZ max_point_AABB;
            feature_extractor.getAABB (min_point_AABB, max_point_AABB);
            vis.addCube (min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 1.0, 0.0, std::to_string(box_id));
            vis.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, std::to_string(box_id));
            box_id++;
        }
        else
        {
            pcl::PointCloud< pcl::PointXYZ >::Ptr cloud_cluster (new pcl::PointCloud< pcl::PointXYZ >);
            for(std::vector< pcl::PointCloud< pcl::PointXYZ > >::iterator it = input_cluster_cloud.begin (); it != input_cluster_cloud.end (); ++it)
            {
                //pcl::PointCloud< pcl::PointXYZ >::Ptr cloud_cluster (new pcl::PointCloud< pcl::PointXYZ >);
                for(pcl::PointCloud< pcl::PointXYZ >::iterator pit = it->points.begin(); pit!= it->points.end(); ++pit)
                {
                    cloud_cluster->points.push_back(*pit);
                }
            }
            pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(cloud_cluster,0,255,0);
            vis.addPointCloud(cloud_cluster,green,"cloud_cluster",0);
            // draw bounding box
            pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
            feature_extractor.setInputCloud(cloud_cluster);
            feature_extractor.compute();
            pcl::PointXYZ min_point_AABB;
            pcl::PointXYZ max_point_AABB;
            feature_extractor.getAABB (min_point_AABB, max_point_AABB);
            vis.addCube (min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 1.0, 0.0, std::to_string(box_id));
            vis.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, std::to_string(box_id));
            box_id++;
        }*/
        // render
        vis.addPointCloud(AfterSegmentation,green,"AfterSegmentation",0);
        vis.addPointCloud(Background,blue,"Background",0);
        //test
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(AfterSegmentation,0,255,0);
        //vis.addPointCloud(AfterSegmentation,green,"Background",0);
        
        //vis.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_cluster");
        
        // start render cluster person
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green(person_cloud,0,255,0);
        //vis.addPointCloud(person_cloud,green,"person_cloud",0);
        //end
        vis.spinOnce();
        START_FRAME++;
        //std::cout << std::to_string(input_cluster_cloud.size()) << std::endl;
        //cluster_counter = 0;
    }
    return 0;
}
