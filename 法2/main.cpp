#pragma once
#include"thickness_cal.h"
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/point_tests.h> 
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <queue>
#include <list>
#include <cmath>
#include <ctime>
#include <vector>
#include <iostream>
#include <pcl/console/parse.h>
#include <algorithm>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/features/boundary.h>
#include <math.h>
#include <boost/make_shared.hpp>
#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/covariance_sampling.h>
#include <pcl/filters/normal_space.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/io/ply_io.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/features/principal_curvatures.h>
#include <utility>
#include <iostream>
#include <vector>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/features/boundary.h>
#include <math.h>
#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <string>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/organized_edge_detection.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/covariance_sampling.h>
#include <pcl/filters/normal_space.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/features/boundary.h>
#include <pcl/io/ply_io.h>
#include <pcl/features/integral_image_normal.h>
#include <pcl/common/common.h>
#include <boost/thread/thread.hpp>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/progressive_morphological_filter.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/console/parse.h>
#include <math.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/console/parse.h>
#include <pcl/search/organized.h>
#include<pcl/common/distances.h>
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloudT;
typedef pcl::PointNormal PointNT;
typedef pcl::PointCloud<PointNT> PointNCloudT;
typedef pcl::PointXYZL PointLT;
typedef pcl::PointCloud<PointLT> PointLCloudT;
using namespace std;
using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;
using pointDirection = std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;
using PointDirectionPtr = std::shared_ptr<pointDirection>;

void cal_direction(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, PointDirectionPtr& A)
{
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);//方向
	pcl::PointCloud<pcl::Normal>::Ptr normals1(new pcl::PointCloud<pcl::Normal>);//改进的方向
	pcl::KdTreeFLANN<pcl::PointXYZ> kd;
	kd.setInputCloud(cloud);
	std::vector<std::pair<float, int>> cal_dis_result;
	int k = 15;//超参数
	std::vector<int> pointIdxNKNSearch(k);
	std::vector<float> pointNKNSquaredDistance(k);
	pcl::PointXYZ searchPoint;
	pcl::PointCloud<pcl::PointXYZ>::Ptr line_cloud(new pcl::PointCloud<pcl::PointXYZ>());//k邻域的点集
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_dection(new pcl::PointCloud<pcl::PointXYZ>());
	Eigen::Vector3f ax(1, 0, 0), ay(0, 1, 0), az(0, 0, 1);

	for (int i = 0; i < cloud->size(); i++)
	{
		searchPoint = cloud->points[i];
		if (kd.nearestKSearch(searchPoint, k, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
		{
			for (int j = 0; j < pointIdxNKNSearch.size(); j++)
			{
				line_cloud->push_back(cloud->points[pointIdxNKNSearch[j]]);
			}

			pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
			//主成分法
			Eigen::Vector4f pcaCentroid;
			pcl::compute3DCentroid(*line_cloud, pcaCentroid);
			Eigen::Matrix3f covariance;
			pcl::computeCovarianceMatrix(*line_cloud, pcaCentroid, covariance);

			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
			Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
			Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();
			float t1 = eigenValuesPCA(0);
			int ii = 0;
			if (t1 < eigenValuesPCA(1))
			{
				ii = 1;
				t1 = eigenValuesPCA(1);
			}
			if (t1 < eigenValuesPCA(2))
			{
				ii = 2;
				t1 = eigenValuesPCA(2);
			}
			Eigen::Vector3f v(eigenVectorsPCA(0, ii), eigenVectorsPCA(1, ii), eigenVectorsPCA(2, ii));
			v /= v.norm();
			double angle_x = pcl::getAngle3D(ax, v, false);
			double angle_y = pcl::getAngle3D(ay, v, false);
			double angle_z = pcl::getAngle3D(az, v, false);
			pcl::Normal normal_tmp;
			pcl::PointXYZ dection_tmp;
			if (angle_x > M_PI / 2 && angle_y > M_PI / 2 && angle_z > M_PI / 2)
			{
				normal_tmp.normal_x = -v(0);
				normal_tmp.normal_y = -v(1);
				normal_tmp.normal_z = -v(2);
				dection_tmp.x = -v(0);
				dection_tmp.y = -v(1);
				dection_tmp.z = -v(2);
				normals->push_back(normal_tmp);
				cloud_dection->push_back(dection_tmp);
				line_cloud->clear();
			}
			else
			{
				normal_tmp.normal_x = v(0);
				normal_tmp.normal_y = v(1);
				normal_tmp.normal_z = v(2);
				dection_tmp.x = v(0);
				dection_tmp.y = v(1);
				dection_tmp.z = v(2);
				normals->push_back(normal_tmp);
				cloud_dection->push_back(dection_tmp);
				line_cloud->clear();

			}

		}
	}

	pcl::KdTreeFLANN<pcl::PointXYZ> kd1;
	kd1.setInputCloud(cloud);
	int k1 = 20;
	std::vector<int> pointIdxNKNSearch1(k1);
	std::vector<float> pointNKNSquaredDistance1(k1);
	pcl::PointXYZ searchPoint1;
	for (int i = 0; i < cloud->size(); i++)
	{
		searchPoint1 = cloud->points[i];
		if (kd1.nearestKSearch(searchPoint1, k1, pointIdxNKNSearch1, pointNKNSquaredDistance1) > 0)
		{
			//均值滤波
			float sum_x = 0; float sum_y = 0; float sum_z = 0;
			for (int j = 0; j < pointIdxNKNSearch1.size(); j++)
			{
				sum_x += cloud_dection->points[pointIdxNKNSearch1[j]].x;
				sum_y += cloud_dection->points[pointIdxNKNSearch1[j]].y;
				sum_z += cloud_dection->points[pointIdxNKNSearch1[j]].z;
			}
			pcl::Normal normal_tmp1;
			Eigen::Vector3f v1;
			v1(0) = sum_x / pointIdxNKNSearch1.size();
			v1(1) = sum_y / pointIdxNKNSearch1.size();
			v1(2) = sum_z / pointIdxNKNSearch1.size();
			v1.normalize();
			normal_tmp1.normal_x = v1(0);
			normal_tmp1.normal_y = v1(1);
			normal_tmp1.normal_z = v1(2);
			normals1->push_back(normal_tmp1);
			A->push_back(v1);
		}
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void cal_julie(float th, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, PointDirectionPtr& A, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out, PointDirectionPtr& seg_direction_all)
{
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud);
	vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance(th);
	ec.setMinClusterSize(90);
	ec.setMaxClusterSize(70000);
	ec.setSearchMethod(tree);
	ec.setInputCloud(cloud);
	ec.extract(cluster_indices);
	int j = 0;
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster_all(new pcl::PointCloud<pcl::PointXYZ>);
	PointDirectionPtr cloud_direction_all(new pointDirection());
	for (vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
		PointDirectionPtr cloud_direction(new pointDirection());
		for (vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
		{
			cloud_cluster_all->points.push_back(cloud->points[*pit]);
			cloud_direction_all->push_back((*A)[*pit]);
		}
	}
	cloud_out = cloud_cluster_all;
	seg_direction_all = cloud_direction_all;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
std::vector<std::vector<int>> voxel_search(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
	float resolution = 0.006;				//Octree分辨率，即最低级叶子节点的大小
	pcl::octree::OctreePointCloudSearch<PointXYZ> octree(resolution);	//初始化八叉树
	octree.setInputCloud(cloud);		//设置输入点云
	octree.addPointsFromInputCloud();	//构建八叉树
	pcl::octree::OctreePointCloud<pcl::PointXYZ>::LeafNodeIterator iter_leaf;
	std::vector<std::vector<int>> index;
	for (iter_leaf = octree.leaf_depth_begin(); iter_leaf != octree.leaf_depth_end(); iter_leaf++)
	{
		std::vector<int> pointsIdx;
		pointsIdx = iter_leaf.getLeafContainer().getPointIndicesVector();
		index.push_back(pointsIdx);
		pointsIdx.clear();
	}
	return index;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv)
{
	float max_z = 1.5;//质心z坐标阈值
	float max_view_z = 50;//视角阈值
	int select_min = 0;
	int select_max = 100;
	int select_step = 1;
	int debug = 1;
	string output_name = "output_hou";
	string output2d_name = "output2d";
	string input2d_name = "input2d";
	parse_argument(argc, argv, "maxz", max_z);
	parse_argument(argc, argv, "maxviewz", max_view_z);
	parse_argument(argc, argv, "min", select_min);
	parse_argument(argc, argv, "max", select_max);
	parse_argument(argc, argv, "step", select_step);
	parse_argument(argc, argv, "output", output_name);
	parse_argument(argc, argv, "output2d", output2d_name);
	parse_argument(argc, argv, "debug", debug);
	parse_argument(argc, argv, "input", input2d_name);



	//求两种边界
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());//导入原始废钢有序点云
	PointCloudT::Ptr voxel_filtered(new PointCloudT);//简化后的点云
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	if (argc < 2)
	{
		PCL_ERROR("请输入原始点云文件路径 \n");
		ofstream out("output_hou.txt");
		out.close();
		return 0;
	}
	std::string Dir = argv[1];
	if (loadPCDFile<PointXYZRGB>(Dir, *cloud) == -1)
	{
		PCL_ERROR("未能正确加载原始点云数据，请检查路径 \n");
		return 0;
	}
	//调试：判断nan点并赋值
	for (int i = 0; i < cloud->points.size(); i++)
	{
		if (cloud->points[i].x == 0 && cloud->points[i].y == 0 && cloud->points[i].z == 0)
		{
			cloud->points[i].x = NAN;
			cloud->points[i].y = NAN;
			cloud->points[i].z = NAN;
		}
	}
	//边缘点提取
	IntegralImageNormalEstimation<PointXYZRGB, Normal> ne;//计算法线
	ne.setDepthDependentSmoothing(true);
	ne.setNormalEstimationMethod(ne.AVERAGE_DEPTH_CHANGE);
	ne.setMaxDepthChangeFactor(0.001f);
	ne.setNormalSmoothingSize(10.0f);
	ne.setBorderPolicy(ne.BORDER_POLICY_MIRROR);
	ne.setInputCloud(cloud);
	ne.compute(*normals);
	OrganizedEdgeFromRGBNormals<PointXYZRGB, Normal, Label> oed;
	oed.setInputNormals(normals);
	oed.setInputCloud(cloud);
	oed.setDepthDisconThreshold(0.015);
	oed.setMaxSearchNeighbors(100);
	oed.setHCCannyLowThreshold(0.9);
	oed.setHCCannyHighThreshold(1.15);//参数越大点越少
	oed.setEdgeType(oed.EDGELABEL_NAN_BOUNDARY | oed.EDGELABEL_OCCLUDING | oed.EDGELABEL_OCCLUDED | oed.EDGELABEL_HIGH_CURVATURE | oed.EDGELABEL_RGB_CANNY);
	PointCloud<Label> labels;
	vector<PointIndices> label_indices;
	oed.compute(labels, label_indices);
	PointCloud<PointXYZ>::Ptr occluding_edges(new PointCloud<PointXYZ>),
		occluded_edges(new PointCloud<PointXYZ>),
		nan_boundary_edges(new PointCloud<PointXYZ>),
		high_curvature_edges(new PointCloud<PointXYZ>),
		rgb_edges(new PointCloud<PointXYZ>);

	copyPointCloud(*cloud, label_indices[0].indices, *nan_boundary_edges);
	copyPointCloud(*cloud, label_indices[1].indices, *occluding_edges);//前景
	copyPointCloud(*cloud, label_indices[2].indices, *occluded_edges);//背景
	copyPointCloud(*cloud, label_indices[3].indices, *high_curvature_edges);
	copyPointCloud(*cloud, label_indices[4].indices, *rgb_edges);

	//if (nan_boundary_edges->size() > 0)
	//{
	//	pcl::io::savePCDFileASCII("nan_boundary_edges.pcd", *nan_boundary_edges);
	//}
	//if (occluding_edges->size() > 0)
	//{
	//	pcl::io::savePCDFileASCII("occluding_edges.pcd", *occluding_edges);
	//}
	//if (occluded_edges->size() > 0)
	//{
	//	pcl::io::savePCDFileASCII("occluded_edges.pcd", *occluded_edges);
	//}
	//if (high_curvature_edges->size() > 0)
	//{
	//	pcl::io::savePCDFileASCII("high_curvature_edges.pcd", *high_curvature_edges);
	//}
	//if (rgb_edges->size() > 0)
	//{
	//	pcl::io::savePCDFileASCII("rgb_edges.pcd", *rgb_edges);
	//}

	//计算方向
	PointDirectionPtr A(new pointDirection());
	PointDirectionPtr B(new pointDirection());
	PointDirectionPtr A_seg(new pointDirection());
	PointDirectionPtr B_seg(new pointDirection());
	pcl::PointCloud<pcl::PointXYZ>::Ptr occluding_seg, high_curvature_seg;
	cal_direction(occluding_edges, A);
	cal_direction(high_curvature_edges, B);
	cout << "方向计算完毕" << endl;

	//进行分割
	cal_julie(0.016, occluding_edges, A, occluding_seg, A_seg);//深度突变聚类
	cal_julie(0.025, high_curvature_edges, B, high_curvature_seg, B_seg);//曲率突变聚类
	/*occluding_seg->width = occluding_seg->points.size();
	occluding_seg->height = 1;
	occluding_seg->is_dense = true;
	stringstream ss;
	ss << "occ" << ".pcd";
	pcl::PCDWriter writer;
	writer.write<pcl::PointXYZ>(ss.str(), *occluding_seg, false);*/

	cout << "聚类完毕" << endl;
	//体素化
	std::vector<std::vector<int>> A_index;
	std::vector<std::vector<int>> B_index;
	B_index = voxel_search(high_curvature_seg);
	A_index = voxel_search(occluding_seg);
	//厚度检测
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudXYZ(new pcl::PointCloud<pcl::PointXYZ>());
	for (int i = 0; i < cloud->points.size(); i++)
	{
		pcl::PointXYZ p;
		p.x = cloud->points[i].x;
		p.y = cloud->points[i].y;
		p.z = cloud->points[i].z;
		cloudXYZ->points.push_back(p);
	}
	thickness::Thickness<PointXYZ>* test = new thickness::Thickness<PointXYZ>;
	std::vector<std::pair<float, int>> s;//厚度
	std::vector<std::vector<float>> aabb;//box
	test->set_search_method(tree);
	/*test->input_cloud(occluding_seg,high_curvature_seg);
	test->input_cloud_direction(A_seg, B_seg);*/
	test->input_original_cloud(cloudXYZ);
	test->input_cloud(high_curvature_seg, occluding_seg);
	test->input_index(B_index, A_index);
	test->input_cloud_direction(B_seg, A_seg);
	test->clustering();
	test->plane_detection();
	test->parallel_detection();
	test->density_detection(max_z, max_view_z);
	test->statistics(s, aabb, cloud);
	test->feature_select();
	auto Statistics_before_sort = s;
	sort(s.begin(), s.end());
	vector<pair<string, float>>  histogram;//定义一个直方图容器
	int sections = ceil((select_max - select_min) / static_cast<float>(select_step));
	vector <int> index_num(sections, 0.0);
	int sum_point = 0;
	for (auto it : s)//计算所有边界点数量
	{
		sum_point += it.second;
	}
	for (int i = 0; i < s.size(); i++)
	{
		float temp = s[i].first;
		for (int ind = 0; ind < sections; ind++)
		{
			if ((temp * 1000) > ((ind * select_step) + select_min) && (temp * 1000) <= ((ind + 1) * select_step) + select_min)
			{
				index_num[ind] += s[i].second;
			}
		}
	}
	for (int region = 0; region < sections; region++)
	{
		string temp_region = to_string((region * select_step) + select_min) + "-" + to_string(((region + 1) * select_step) + select_min);

		float c = (index_num[region] / static_cast<float>(sum_point));
		histogram.push_back(make_pair(temp_region, c * 100));
	}
	ofstream out(output_name + ".txt");
	for (int i = 0; i < sections; i++)
	{
		if (!isfinite(histogram[i].second))
		{
			histogram[i].second = 0;
		}
		out << histogram[i].first << " " << histogram[i].second << "%" << " " << index_num[i] << endl;
	}
	out.close();
	cout << "输出直方图完毕" << endl;
	//输出包围盒的像素坐标
	ofstream out1(output2d_name + ".txt");
	int W = cloud->width;
	int H = cloud->height;
	int i = 0;
	for (auto ite = aabb.begin(); ite != aabb.end(); ite++)
	{
		int min_x, min_y, max_x, max_y;
		int u_min, u_max, v_min, v_max;
		for (int j = 0; j < cloud->size(); j++)
		{
			if (cloud->points[j].x == (*ite)[0])
			{
				min_x = j;
			}
			if (cloud->points[j].y == (*ite)[1])
			{
				min_y = j;
			}
			if (cloud->points[j].x == (*ite)[2])
			{
				max_x = j;
			}
			if (cloud->points[j].y == (*ite)[3])
			{
				max_y = j;
			}
		}
		v_min = ceil(min_y / W);
		u_min = min_x % W;
		v_max = ceil(max_y / W);
		u_max = max_x % W;
		out1 << u_min << " " << v_min << " " << u_max << " " << v_max << " " << (Statistics_before_sort[i].first) * 1000 << endl;
		i++;
	}

	
	
}