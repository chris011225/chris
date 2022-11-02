#include "thickness_cal.h"
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

#include "supersize_cal.h"
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/console/parse.h>
#include <pcl/search/organized.h>
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

void cal_aabb(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointXYZ& min_point_AABB, pcl::PointXYZ& max_point_AABB)
{
	pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
	feature_extractor.setInputCloud(cloud);
	feature_extractor.compute();
	feature_extractor.getAABB(min_point_AABB, max_point_AABB);
}
void cal_fangxiang(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, PointDirectionPtr& A)
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
std::vector<pcl::PointIndices> cal_julie(string type, float th, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, PointDirectionPtr& A, vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& seg_cloud, vector<PointDirectionPtr>& seg_direction, \
	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& seg_cloud_all, vector<PointDirectionPtr>& seg_direction_all)
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
			cloud_cluster->points.push_back(cloud->points[*pit]);
			cloud_direction->push_back((*A)[*pit]);
			cloud_cluster_all->points.push_back(cloud->points[*pit]);
			cloud_direction_all->push_back((*A)[*pit]);
		}
		seg_cloud.push_back(cloud_cluster);
		seg_direction.push_back(cloud_direction);
	}
	seg_cloud_all.push_back(cloud_cluster_all);
	seg_direction_all.push_back(cloud_direction_all);
	return cluster_indices;
	/*cloud_cluster->width = cloud_cluster->points.size();
	cloud_cluster->height = 1;
	cloud_cluster->is_dense = true;*/
	/*stringstream ss;
	ss << type << "聚类后结果整体" << ".pcd";
	pcl::PCDWriter writer;
	writer.write<pcl::PointXYZ>(ss.str(), *cloud_cluster, false);*/
}
std::vector<pair<float, float>> tongji_houdu(int min, int max, int step, string name, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudXYZ, pcl::PointCloud<pcl::PointXYZ>::Ptr& occluded_edges, pcl::PointCloud<pcl::PointXYZ>::Ptr& high_curvature_edges, PointDirectionPtr& A, PointDirectionPtr& B, vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& seg_cloud_A, vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& seg_cloud_B, vector<PointDirectionPtr>& seg_direction_A, vector<PointDirectionPtr>& seg_direction_B, \
	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& seg_cloud_A_all, vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& seg_cloud_B_all, vector<PointDirectionPtr>& seg_direction_A_all, vector<PointDirectionPtr>& seg_direction_B_all, std::vector<vector<pair<int, int>>>& AABB_2d, vector<pcl::PointIndices>& cluster_indices_d, float max_z, float max_view_ang, pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZL>::Ptr& supersize_labeled_cloud, supersize::SuperSize<PointT>& super, std::map < std::uint32_t, std::vector<float>>& label_with_aabb,int debug)
{
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr all_thickness(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr all_thickness_self(new pcl::PointCloud<pcl::PointXYZRGB>);
	std::vector<pair<float, int>> Statistics;
	std::vector<pair<float, float>> Statistics1;//厚度和面积统计
	std::map < std::uint32_t, std::vector<float>> label_with_houdu;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	std::vector<std::vector<std::pair<int, int>>> all_merge;//将每次小聚类的厚度在原始深度突变点云中的索引保存到一起

	pcl::KdTreeFLANN<pcl::PointXYZL>::Ptr tree_super(new pcl::KdTreeFLANN<pcl::PointXYZL>());
	tree_super->setInputCloud(supersize_labeled_cloud);
	


	//厚度识别（class内部的识别）
	for (int i = 0; i < seg_cloud_A.size(); i++)
	{
		thickness::Thickness<PointXYZ>* test = new thickness::Thickness<PointXYZ>;
		test->setMinClusterSize(5);
		test->setMaxClusterSize(10000);
		test->setRadius1(0.01);
		test->setMaxClusterSize(10);
		test->setDistanceToPlane(0.0006);
		test->setSearchMethod(tree);
		test->setAngelThreshold(M_PI / 6);
		test->setAngelThresholdWithDirection(M_PI / 6);
		test->inputCloud(seg_cloud_A[i]);
		test->inputCloudDirection(seg_direction_A[i]);
		std::vector<std::vector<std::pair<int, int>>> all;
		int atype = 2;
		test->thicknessDetection(all, atype);
		//cout << all.size() << endl;
		if (all.size() == 0)
		{
			delete test;
			continue;
		}
		//进行内点为空的过滤
		test->inputOriginalCloud(cloudXYZ);
		all = test->validation();//去除不是厚度的点对（中间无点）
		all = test->validationDirection_classWithin();//去除两边主方向不平行的厚度对--类内
		//聚类操作中把每个结果放到all_merge里，然后在外层用深度突变点来聚类
		auto one_seg_indexs = cluster_indices_d[i];
		for (int i = 0; i < all.size(); i++)
		{
			std::vector<std::pair<int, int>> temp_thickness;
			std::pair<int, int> temp_pair;
			for (int j = 0; j < all[i].size(); j++)
			{
				temp_pair.first = one_seg_indexs.indices[all[i][j].first];
				temp_pair.second = one_seg_indexs.indices[all[i][j].second];
				temp_thickness.push_back(temp_pair);
			}
			all_merge.push_back(temp_thickness);
		}

		//统计厚度
		/*std::vector<std::pair<float, int>> thickness_result;
		test->thicknessCal(all, thickness_result, atype);

		for (int k = 0; k < thickness_result.size(); k++)
		{
			Statistics.push_back(thickness_result[k]);
		}*/
		//int size = 0;
		//for (std::vector<std::vector<std::pair<int, int>>>::const_iterator it = all.begin(); it != all.end(); ++it)
		//{
		//	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
		//	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_aabb(new pcl::PointCloud<pcl::PointXYZ>);
		//	int Random_color_r, Random_color_g, Random_color_b;
		//	Random_color_r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		//	Random_color_g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		//	Random_color_b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		//	for (std::vector<std::pair<int, int>>::const_iterator pit = it->begin(); pit != it->end(); ++pit)
		//	{
		//		pcl::PointXYZRGB tempa, tempb;
		//		tempa.x = seg_cloud_A[i]->points[(*pit).first].x;
		//		tempa.y = seg_cloud_A[i]->points[(*pit).first].y;
		//		tempa.z = seg_cloud_A[i]->points[(*pit).first].z;
		//		tempa.r = Random_color_r;
		//		tempa.g = Random_color_g;
		//		tempa.b = Random_color_b;
		//		cloud_cluster->push_back(tempa);
		//		tempb.x = seg_cloud_A[i]->points[(*pit).second].x;
		//		tempb.y = seg_cloud_A[i]->points[(*pit).second].y;
		//		tempb.z = seg_cloud_A[i]->points[(*pit).second].z;
		//		tempb.r = Random_color_r;
		//		tempb.g = Random_color_g;
		//		tempb.b = Random_color_b;
		//		cloud_cluster->push_back(tempb);
		//		pcl::PointXYZ aabb_point;
		//		aabb_point.x = seg_cloud_A[i]->points[(*pit).first].x;
		//		aabb_point.y = seg_cloud_A[i]->points[(*pit).first].y;
		//		aabb_point.z = seg_cloud_A[i]->points[(*pit).first].z;
		//		cloud_aabb->push_back(aabb_point);
		//		aabb_point.x = seg_cloud_A[i]->points[(*pit).second].x;
		//		aabb_point.y = seg_cloud_A[i]->points[(*pit).second].y;
		//		aabb_point.z = seg_cloud_A[i]->points[(*pit).second].z;
		//		cloud_aabb->push_back(aabb_point);
		//	}
		//	cloud_cluster->width = cloud_cluster->points.size();
		//	cloud_cluster->height = 1;
		//	cloud_cluster->is_dense = true;
		//	cloud_aabb->width = cloud_aabb->points.size();
		//	cloud_aabb->height = 1;
		//	cloud_aabb->is_dense = true;
		//	//求二维包围盒
		//	pcl::PointXYZ min_point_aabb, max_point_aabb;
		//	cal_aabb(cloud_aabb, min_point_aabb, max_point_aabb);
		//	vector<pair<int, int>> aabb_points(4);//xy坐标最小值和最大值在原始点云里的索引
		//	for (int i = 0; i < occluded_edges->size(); i++)
		//	{
		//		if (occluded_edges->points[i].x == min_point_aabb.x)
		//		{
		//			aabb_points[0].first = 1;
		//			aabb_points[0].second = i;
		//		}
		//		if (occluded_edges->points[i].y == min_point_aabb.y)
		//		{
		//			aabb_points[1].first = 1;
		//			aabb_points[1].second = i;
		//		}
		//		if (occluded_edges->points[i].x == max_point_aabb.x)
		//		{
		//			aabb_points[2].first = 1;
		//			aabb_points[2].second = i;
		//		}
		//		if (occluded_edges->points[i].y == max_point_aabb.y)
		//		{
		//			aabb_points[3].first = 1;
		//			aabb_points[3].second = i;
		//		}
		//	}
		//	AABB_2d.push_back(aabb_points);

		//	/*stringstream ss;
		//	ss << i << "CLASS的" << size << ".pcd";
		//	pcl::PCDWriter writer;
		//	writer.write<pcl::PointXYZRGB>(ss.str(), *cloud_cluster, false);
		//	size++;*/
		//	*all_thickness_self += *cloud_cluster;
		//}
		delete test;
	}
	//聚类
	thickness::Cluster<PointXYZ>* cluster = new thickness::Cluster<PointXYZ>;
	std::vector<std::vector<std::pair<int, int>>> all_after_clusting;
	std::vector<std::pair<float, int>> thickness_result;
	cluster->inputMargin(occluded_edges);
	cluster->inputThicknessIndex(all_merge, 0);
	cluster->setMaxZ(max_z);
	cluster->setMaxViewAng(max_view_ang);
	cluster->clustering(all_after_clusting);
	thickness::Thickness<PointXYZ>* test = new thickness::Thickness<PointXYZ>;
	test->inputCloud(occluded_edges);
	test->inputCloudDirection(seg_direction_A_all[0]);
	int t = 5;
	std::vector<float> std_result_all;//方差
	std::vector<std::vector<std::pair<int, int>>> all_after_clusting_std;
	//test->thicknessCal(all_after_clusting, all_after_clusting_std, thickness_result, std_result_all, t);//方差滤除
	test->thicknessCal_1(all_after_clusting,thickness_result, t, cloud);//不用方差滤除

	for (int k = 0; k < thickness_result.size(); k++)
	{
		Statistics.push_back(thickness_result[k]);
	}
	int size = 0;
	int ii = 0;
	for (std::vector<std::vector<std::pair<int, int>>>::const_iterator it = all_after_clusting.begin(); it != all_after_clusting.end(); ++it)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_aabb(new pcl::PointCloud<pcl::PointXYZ>);
		std::vector<int> cur_labels(4);//索引前后得到的分割类别标签
		int Random_color_r, Random_color_g, Random_color_b;
		Random_color_r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		for (std::vector<std::pair<int, int>>::const_iterator pit = it->begin(); pit != it->end(); ++pit)
		{
			pcl::PointXYZRGB tempa, tempb;
			tempa.x = occluded_edges->points[(*pit).first].x;
			tempa.y = occluded_edges->points[(*pit).first].y;
			tempa.z = occluded_edges->points[(*pit).first].z;
			tempa.r = Random_color_r;
			tempa.g = Random_color_g;
			tempa.b = Random_color_b;
			cloud_cluster->push_back(tempa);
			tempb.x = occluded_edges->points[(*pit).second].x;
			tempb.y = occluded_edges->points[(*pit).second].y;
			tempb.z = occluded_edges->points[(*pit).second].z;
			tempb.r = Random_color_r;
			tempb.g = Random_color_g;
			tempb.b = Random_color_b;
			cloud_cluster->push_back(tempb);

			if (pit == it->begin())
			{
				pcl::PointXYZL temp_al, temp_bl;
				temp_al.x = occluded_edges->points[(*pit).first].x;
				temp_al.y = occluded_edges->points[(*pit).first].y;
				temp_al.z = occluded_edges->points[(*pit).first].z;
				temp_al.label = -1;
				temp_bl.x = occluded_edges->points[(*pit).second].x;
				temp_bl.y = occluded_edges->points[(*pit).second].y;
				temp_bl.z = occluded_edges->points[(*pit).second].z;
				temp_bl.label = -1;
				std::vector<int> nn_indices(1);
				std::vector<float> nn_distances(1);
				tree_super->nearestKSearch(temp_al, 1, nn_indices, nn_distances);
				cur_labels[0] = nn_indices[0];
				nn_indices.clear();
				nn_distances.clear();
				tree_super->nearestKSearch(temp_bl, 1, nn_indices, nn_distances);
				cur_labels[1] = nn_indices[0];
			}
			if (pit == it->end() - 1)
			{
				pcl::PointXYZL temp_al, temp_bl;
				temp_al.x = occluded_edges->points[(*pit).first].x;
				temp_al.y = occluded_edges->points[(*pit).first].y;
				temp_al.z = occluded_edges->points[(*pit).first].z;
				temp_al.label = -1;
				temp_bl.x = occluded_edges->points[(*pit).second].x;
				temp_bl.y = occluded_edges->points[(*pit).second].y;
				temp_bl.z = occluded_edges->points[(*pit).second].z;
				temp_bl.label = -1;
				std::vector<int> nn_indices(1);
				std::vector<float> nn_distances(1);
				tree_super->nearestKSearch(temp_al, 1, nn_indices, nn_distances);
				cur_labels[2] = nn_indices[0];
				nn_indices.clear();
				nn_distances.clear();
				tree_super->nearestKSearch(temp_bl, 1, nn_indices, nn_distances);
				cur_labels[3] = nn_indices[0];
			}

			pcl::PointXYZ aabb_point;
			aabb_point.x = occluded_edges->points[(*pit).first].x;
			aabb_point.y = occluded_edges->points[(*pit).first].y;
			aabb_point.z = occluded_edges->points[(*pit).first].z;
			cloud_aabb->push_back(aabb_point);
			aabb_point.x = occluded_edges->points[(*pit).second].x;
			aabb_point.y = occluded_edges->points[(*pit).second].y;
			aabb_point.z = occluded_edges->points[(*pit).second].z;
			cloud_aabb->push_back(aabb_point);
		}
		auto label_1 = supersize_labeled_cloud->points[cur_labels[0]].label;
		auto label_2 = supersize_labeled_cloud->points[cur_labels[1]].label;
		auto label_3 = supersize_labeled_cloud->points[cur_labels[2]].label;
		auto label_4 = supersize_labeled_cloud->points[cur_labels[3]].label;
		if (label_1 == label_2 && label_2 == label_3 && label_3 == label_4)
		{
			label_with_houdu[label_1].push_back(thickness_result[ii].first+100);//加100为了给大件进行最后的厚度赋值用于判断类内还是类间
		}
		cloud_cluster->width = cloud_cluster->points.size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;
		cloud_aabb->width = cloud_aabb->points.size();
		cloud_aabb->height = 1;
		cloud_aabb->is_dense = true;
		pcl::PointXYZ min_point_aabb, max_point_aabb;
		cal_aabb(cloud_aabb, min_point_aabb, max_point_aabb);
		ii++;
		/*stringstream ss;
		ss << i << "CLASS的" << size << ".pcd";
		pcl::PCDWriter writer;
		writer.write<pcl::PointXYZRGB>(ss.str(), *cloud_cluster, false);
		size++;*/
		*all_thickness_self += *cloud_cluster;
	}
	cout << "厚度内部检测完毕" << endl;

	//只一次类间检测
	/*thickness::Thickness<PointXYZ> test;
	test.setMinClusterSize(5);
	test.setMaxClusterSize(10000);
	test.setRadius1(0.01);
	test.setMaxClusterSize(10);
	test.setSearchMethod(tree);
	test.setDistanceToPlane(0.0006);
	test.setAngelThreshold(M_PI/7);
	test.setAngelThresholdWithDirection(M_PI/7);
	test.inputCloud(occluded_edges, high_curvature_edges);
	test.inputCloudDirection(A,B);
	std::vector<std::vector<std::pair<int, int>>> all;
	int atype = 1;
	test.thicknessDetection(all,atype);
	std::vector<std::pair<float, int>> thickness_result;
	test.thicknessCal(thickness_result, atype);
	for (int k = 0; k < thickness_result.size(); k++)
	{
		Statistics.push_back(thickness_result[k]);
	}
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster_all(new pcl::PointCloud<pcl::PointXYZRGB>);
	int i = 0;
	for (std::vector<std::vector<std::pair<int, int>>>::const_iterator it=all.begin();it!=all.end();++it)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
		int Random_color_r, Random_color_g, Random_color_b;
		Random_color_r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		if (it->size() < 25)
		{
			continue;
		}
		for (std::vector<std::pair<int, int>>::const_iterator pit = it->begin(); pit != it->end(); ++pit)
		{
			pcl::PointXYZRGB tempa,tempb;
			tempa.x = occluded_edges->points[(*pit).first].x;
			tempa.y = occluded_edges->points[(*pit).first].y;
			tempa.z = occluded_edges->points[(*pit).first].z;
			tempa.r = Random_color_r;
			tempa.g = Random_color_g;
			tempa.b = Random_color_b;
			cloud_cluster->push_back(tempa);
			tempb.x = high_curvature_edges->points[(*pit).second].x;
			tempb.y = high_curvature_edges->points[(*pit).second].y;
			tempb.z = high_curvature_edges->points[(*pit).second].z;
			tempb.r = Random_color_r;
			tempb.g = Random_color_g;
			tempb.b = Random_color_b;
			cloud_cluster->push_back(tempb);
		}
		cloud_cluster->width = cloud_cluster->points.size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;
		stringstream ss;
		ss << "厚度" << i<< ".pcd";
		pcl::PCDWriter writer;
		writer.write<pcl::PointXYZRGB>(ss.str(), *cloud_cluster, false);
		i++;
		*all_thickness += *cloud_cluster;
	}*/

	//遍历法类间检测
	for (int i = 0; i < seg_cloud_A_all.size(); i++)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_clusters_all(new pcl::PointCloud<pcl::PointXYZRGB>);
		for (int j = 0; j < seg_cloud_B_all.size(); j++)
		{
			thickness::Thickness<PointXYZ>* test = new thickness::Thickness<PointXYZ>;
			test->setMinClusterSize(5);
			test->setMaxClusterSize(10000);
			test->setRadius1(0.01);
			test->setSearchMethod(tree);
			test->setMaxClusterSize(10);
			test->setDistanceToPlane(0.0006);
			test->setAngelThreshold(M_PI / 6);
			test->setAngelThresholdWithDirection(M_PI / 6);
			test->inputCloud(seg_cloud_A_all[i], seg_cloud_B_all[j]);
			test->inputCloudDirection(seg_direction_A_all[i], seg_direction_B_all[j]);
			std::vector<std::vector<std::pair<int, int>>> all;
			int atype = 1;
			test->thicknessDetection(all, atype);
			all = test->validationDirection_classBetween();
			//test->inputOriginalCloud(cloudXYZ);
			all = test->validation_fackThickness(all, cloud);
			//聚类
			thickness::Cluster<PointXYZ>* cluster = new thickness::Cluster<PointXYZ>;
			std::vector<std::vector<std::pair<int, int>>> all_after_clusting;
			std::vector<std::pair<float, int>> thickness_result;
			cluster->inputMargin(seg_cloud_A_all[i], seg_cloud_B_all[j]);
			cluster->inputThicknessIndex(all, 1);
			cluster->setMaxZ(max_z);
			cluster->setMaxViewAng(max_view_ang);
			cluster->clustering(all_after_clusting);

			////////////////////////////////////////////////////////////////////////////////////////////////////
			std::vector<float> std_result;
			std::vector<std::vector<std::pair<int, int>>> all_after_clusting_std;//方差滤除之后的结果厚度对
			test->thicknessCal(all_after_clusting, all_after_clusting_std, thickness_result, std_result, atype);
			thickness_result = test->validation_fackAngle(all_after_clusting_std, thickness_result, cloud);
			for (int k = 0; k < std_result.size(); k++)
			{
				std_result_all.push_back(std_result[k]);
			}
			for (int k = 0; k < thickness_result.size(); k++)
			{
				Statistics.push_back(thickness_result[k]);
			}
			if (all.size() == 0)
			{
				delete test;
				continue;
			}
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster_all(new pcl::PointCloud<pcl::PointXYZRGB>);
			int size = 0;
			int ii = 0;
			for (std::vector<std::vector<std::pair<int, int>>>::const_iterator it = all_after_clusting_std.begin(); it != all_after_clusting_std.end(); ++it)
			{
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_aabb(new pcl::PointCloud<pcl::PointXYZ>);
				std::vector<int> cur_labels(4);//索引前后得到的分割类别标签
				int Random_color_r, Random_color_g, Random_color_b;
				Random_color_r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
				Random_color_g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
				Random_color_b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
				for (std::vector<std::pair<int, int>>::const_iterator pit = it->begin(); pit != it->end(); ++pit)
				{
					pcl::PointXYZRGB tempa, tempb;
					tempa.x = seg_cloud_A_all[i]->points[(*pit).first].x;
					tempa.y = seg_cloud_A_all[i]->points[(*pit).first].y;
					tempa.z = seg_cloud_A_all[i]->points[(*pit).first].z;
					tempa.r = Random_color_r;
					tempa.g = Random_color_g;
					tempa.b = Random_color_b;
					cloud_cluster->push_back(tempa);
					tempb.x = seg_cloud_B_all[j]->points[(*pit).second].x;
					tempb.y = seg_cloud_B_all[j]->points[(*pit).second].y;
					tempb.z = seg_cloud_B_all[j]->points[(*pit).second].z;
					tempb.r = Random_color_r;
					tempb.g = Random_color_g;
					tempb.b = Random_color_b;
					cloud_cluster->push_back(tempb);

					if (pit == it->begin())
					{
						pcl::PointXYZL temp_al, temp_bl;
						temp_al.x = seg_cloud_A_all[i]->points[(*pit).first].x;
						temp_al.y = seg_cloud_A_all[i]->points[(*pit).first].y;
						temp_al.z = seg_cloud_A_all[i]->points[(*pit).first].z;
						temp_al.label = -1;
						temp_bl.x = seg_cloud_B_all[i]->points[(*pit).second].x;
						temp_bl.y = seg_cloud_B_all[i]->points[(*pit).second].y;
						temp_bl.z = seg_cloud_B_all[i]->points[(*pit).second].z;
						temp_bl.label = -1;
						std::vector<int> nn_indices(1);
						std::vector<float> nn_distances(1);
						tree_super->nearestKSearch(temp_al, 1, nn_indices, nn_distances);
						cur_labels[0] = nn_indices[0];
						nn_indices.clear();
						nn_distances.clear();
						tree_super->nearestKSearch(temp_bl, 1, nn_indices, nn_distances);
						cur_labels[1] = nn_indices[0];
					}
					if (pit == it->end()-1)
					{
						pcl::PointXYZL temp_al, temp_bl;
						temp_al.x = seg_cloud_A_all[i]->points[(*pit).first].x;
						temp_al.y = seg_cloud_A_all[i]->points[(*pit).first].y;
						temp_al.z = seg_cloud_A_all[i]->points[(*pit).first].z;
						temp_al.label = -1;
						temp_bl.x = seg_cloud_B_all[i]->points[(*pit).second].x;
						temp_bl.y = seg_cloud_B_all[i]->points[(*pit).second].y;
						temp_bl.z = seg_cloud_B_all[i]->points[(*pit).second].z;
						temp_bl.label = -1;
						std::vector<int> nn_indices(1);
						std::vector<float> nn_distances(1);
						tree_super->nearestKSearch(temp_al, 1, nn_indices, nn_distances);
						cur_labels[2] = nn_indices[0];
						nn_indices.clear();
						nn_distances.clear();
						tree_super->nearestKSearch(temp_bl, 1, nn_indices, nn_distances);
						cur_labels[3] = nn_indices[0];
					}
					pcl::PointXYZ aabb_point;
					aabb_point.x = seg_cloud_A_all[i]->points[(*pit).first].x;
					aabb_point.y = seg_cloud_A_all[i]->points[(*pit).first].y;
					aabb_point.z = seg_cloud_A_all[i]->points[(*pit).first].z;
					cloud_aabb->push_back(aabb_point);
					aabb_point.x = seg_cloud_B_all[j]->points[(*pit).second].x;
					aabb_point.y = seg_cloud_B_all[j]->points[(*pit).second].y;
					aabb_point.z = seg_cloud_B_all[j]->points[(*pit).second].z;
					cloud_aabb->push_back(aabb_point);
				}
				auto label_1 = supersize_labeled_cloud->points[cur_labels[0]].label;
				auto label_2 = supersize_labeled_cloud->points[cur_labels[1]].label;
				auto label_3 = supersize_labeled_cloud->points[cur_labels[2]].label;
				auto label_4 = supersize_labeled_cloud->points[cur_labels[3]].label;
				if (label_1 == label_2&& label_2 == label_3&& label_3 == label_4)
				{
					label_with_houdu[label_1].push_back(thickness_result[ii].first);
				}
				cloud_cluster->width = cloud_cluster->points.size();
				cloud_cluster->height = 1;
				cloud_cluster->is_dense = true;
				cloud_aabb->width = cloud_aabb->points.size();
				cloud_aabb->height = 1;
				cloud_aabb->is_dense = true;

				/*stringstream ss;
				ss <<i << "厚度" <<j<< "的"<<size << ".pcd";
				pcl::PCDWriter writer;
				writer.write<pcl::PointXYZRGB>(ss.str(), *cloud_cluster, false);*/
				size++;
				*cloud_cluster_all += *cloud_cluster;
				ii++;
			}
			delete test;
			*cloud_clusters_all += *cloud_cluster_all;
		}
		*all_thickness += *cloud_clusters_all;
	}

	cout << "厚度类间检测完毕" << endl;

	*all_thickness += *all_thickness_self;
	if (debug == 1)
	{
		stringstream ss1;
		ss1 << "法1所有厚度" << ".pcd";
		pcl::PCDWriter writer;
		writer.write<pcl::PointXYZRGB>(ss1.str(), *all_thickness, false);
	}
	
	
	for (auto one_surper = label_with_houdu.begin(); one_surper != label_with_houdu.end(); one_surper++)
	{

		sort(one_surper->second.begin(), one_surper->second.end());
		if (one_surper->second.size() == 1)
		{
			if (one_surper->second[0] > 100)
			{
				one_surper->second[0] = one_surper->second[0] - 100;
			}
		}
		if (one_surper->second.size() > 1)
		{
			if ((*one_surper->second.begin()) > 100 && *(one_surper->second.end() - 1) > 100)
			{
				for (int i = 0; i < one_surper->second.size(); i++)
				{
					one_surper->second[i] = one_surper->second[i] - 100;
				}
				continue;
			}
			else if ((*one_surper->second.begin()) < 100 && *(one_surper->second.end() - 1) > 100)
			{
				float temp = *(one_surper->second.end() - 1) - 100;
				for (int i = 1; i < one_surper->second.size(); i++)
				{
					if (one_surper->second[i] < 100 && one_surper->second[i - 1] < 100)
					{
						float diff = one_surper->second[i] - one_surper->second[i - 1];
						if (diff > 0.0015)
						{
							one_surper->second.clear();
							one_surper->second.push_back(temp);
							break;
						}



					}
					else if (one_surper->second[i] > 100 && one_surper->second[i - 1] < 100)
					{
						float diff = one_surper->second[i] - 100 - one_surper->second[i - 1];
						if (diff > 0.0015)
						{
							one_surper->second.clear();
							one_surper->second.push_back(temp);
							break;
						}

					}
					else if (one_surper->second[i] > 100 && one_surper->second[i - 1] > 100)
					{
						float diff = one_surper->second[i] - 100 - one_surper->second[i - 1] - 100;
						if (diff > 0.0015)
						{
							one_surper->second.clear();
							one_surper->second.push_back(temp);
							break;
						}

					}
				}

			}
			else if ((*one_surper->second.begin()) < 100 && *(one_surper->second.end() - 1) < 100)
			{
				for (int i = 1; i < one_surper->second.size(); i++)
				{
					float diff = one_surper->second[i] - one_surper->second[i - 1];
					if (diff > 0.0015)
					{
						one_surper->second.clear();
						break;
					}
				}

			}

		}

	}
	for (auto one_surper = label_with_houdu.begin(); one_surper != label_with_houdu.end(); one_surper++)
	{
		for (int i = 0; i < one_surper->second.size(); i++)
		{
			if (one_surper->second[i] > 100)
			{
				one_surper->second[i] = one_surper->second[i] - 100;
			}
		}
	}
	for (auto one_surper = label_with_houdu.begin(); one_surper != label_with_houdu.end(); one_surper++)
	{
		if (one_surper->second.size() == 0)
		{
			continue;
		}
		if (one_surper->second.size() > 1)
		{
			float sum = 0;
			int num = one_surper->second.size();
			for (int i = 0; i < one_surper->second.size(); i++)
			{
				sum += one_surper->second[i];
			}
			one_surper->second.clear();
			one_surper->second.push_back(sum / num);
		}
	}
	for (auto one_surper = label_with_houdu.begin(); one_surper != label_with_houdu.end(); one_surper++)
	{
		if (one_surper->second.size() == 0)
		{
			continue;
		}
		pair<float, float> one_pair;
		one_pair.first = one_surper->second[0];
		for (auto ite = super.m_supersizes_properties.begin(); ite != super.m_supersizes_properties.end(); ite++)
		{
			if (ite->first == one_surper->first)
			{
				one_pair.second = ite->second.area;
				label_with_aabb[one_surper->first].push_back(ite->second.aabb_min.x);
				label_with_aabb[one_surper->first].push_back(ite->second.aabb_min.y);
				label_with_aabb[one_surper->first].push_back(ite->second.aabb_max.x);
				label_with_aabb[one_surper->first].push_back(ite->second.aabb_max.y);
			}
		}

		Statistics1.push_back(one_pair);
	}

	auto Statistics_before_sort = Statistics1;
	sort(Statistics1.begin(), Statistics1.end());
	vector<pair<string, float>>  histogram;//定义一个直方图容器
	int sections = ceil((max - min) / static_cast<float>(step));
	vector <int> index_num(sections, 0.0);
	float sum_point = 0;
	for (auto it : Statistics1)//计算所有边界点数量
	{
		sum_point += it.second;
	}
	for (int i = 0; i < Statistics1.size(); i++)
	{
		float temp = Statistics1[i].first;
		for (int ind = 0; ind < sections; ind++)
		{
			if ((temp * 1000) > ((ind * step) + min) && (temp * 1000) <= ((ind + 1) * step) + min)
			{
				index_num[ind] += Statistics1[i].second;
			}
		}
	}
	for (int region = 0; region < sections; region++)
	{
		string temp_region = to_string((region * step) + min) + "-" + to_string(((region + 1) * step) + min);

		float c = (index_num[region] / static_cast<float>(sum_point));
		histogram.push_back(make_pair(temp_region, c * 100));
	}
	ofstream out(name + ".txt");
	for (int i = 0; i < sections; i++)
	{
		if (!isfinite(histogram[i].second))
		{
			histogram[i].second = 0;
		}
		out << histogram[i].first << " " << histogram[i].second << "%" << " " << index_num[i]/1000000.0 << endl;
	}
	out.close();


	//auto Statistics_before_sort = Statistics;
	//sort(Statistics.begin(), Statistics.end());
	//vector<pair<string, float>>  histogram;//定义一个直方图容器
	//int sections = ceil((max - min) / static_cast<float>(step));
	//vector <int> index_num(sections, 0.0);
	//int sum_point = 0;
	//for (auto it : Statistics)//计算所有边界点数量
	//{
	//	sum_point += it.second;
	//}
	//for (int i = 0; i < Statistics.size(); i++)
	//{
	//	float temp = Statistics[i].first;
	//	for (int ind = 0; ind < sections; ind++)
	//	{
	//		if ((temp * 1000) > ((ind * step) + min) && (temp * 1000) <= ((ind + 1) * step) + min)
	//		{
	//			index_num[ind] += Statistics[i].second;
	//		}
	//	}
	//}
	//for (int region = 0; region < sections; region++)
	//{
	//	string temp_region = to_string((region * step) + min) + "-" + to_string(((region + 1) * step) + min);

	//	float c = (index_num[region] / static_cast<float>(sum_point));
	//	histogram.push_back(make_pair(temp_region, c * 100));
	//}
	//ofstream out(name + ".txt");
	//for (int i = 0; i < sections; i++)
	//{
	//	if (!isfinite(histogram[i].second))
	//	{
	//		histogram[i].second = 0;
	//	}
	//	out << histogram[i].first << " " << histogram[i].second << "%" << " " << index_num[i] << endl;
	//}
	//out.close();
	cout << "输出直方图完毕" << endl;
	return Statistics_before_sort;
}
double
compute_cloud_resolution(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
	double resolution = 0.0;
	int numberOfPoints = 0;
	int nres;
	std::vector<int> indices(2);
	std::vector<float> squaredDistances(2);
	pcl::search::KdTree<pcl::PointXYZ> tree;
	tree.setInputCloud(cloud);

	for (size_t i = 0; i < cloud->size(); ++i)
	{    //检查是否存在无效点
		if (!pcl::isFinite(cloud->points[i]))
			continue;

		//Considering the second neighbor since the first is the point itself.
		//在同一个点云内进行k近邻搜索时，k=1的点为查询点本身。
		nres = tree.nearestKSearch(i, 2, indices, squaredDistances);
		if (nres == 2)
		{
			resolution += sqrt(squaredDistances[1]);
			++numberOfPoints;
		}
	}
	if (numberOfPoints != 0)
		resolution /= numberOfPoints;

	return resolution;
}
struct result2d
{
	int umin, vmin, umax, vmax;
	string label;
};

int main(int argc, char** argv)
{
	float max_z = 1000;
	float max_view_z = 60;
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
	std::string Dir_input2d = argv[2];
	ifstream input(Dir_input2d);
	if (!input.is_open()) {
		std::cout << "读取input2d文件失败!" << std::endl;
	}
	result2d res2d;
	vector<result2d> results_2d;
	string line;
	while (getline(input, line))
	{
		stringstream ss(line);
		ss >> res2d.umin;
		ss >> res2d.vmin;
		ss >> res2d.umax;
		ss >> res2d.vmax;
		ss >> res2d.label;
		results_2d.push_back(res2d);
	}

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


	//大件检测
	//体素化滤波减小点数量
	pcl::VoxelGrid<pcl::PointXYZRGB> grid;
	grid.setInputCloud(cloud);
	Eigen::Vector4f leaf_size{ 0.004,0.004,0.004,0 };//超参数
	grid.setLeafSize(leaf_size);
	grid.setMinimumPointsNumberPerVoxel(3);      // 超参数：设置每一个体素内需要包含的最小点个数
	grid.filter(*voxel_filtered);
	pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
	kdtree.setInputCloud(cloud);
	pcl::PointIndicesPtr inds = std::shared_ptr<pcl::PointIndices>(new pcl::PointIndices());
	for (size_t i = 0; i < voxel_filtered->points.size(); i++) {
		pcl::PointXYZRGB searchPoint;
		searchPoint.x = voxel_filtered->points[i].x;
		searchPoint.y = voxel_filtered->points[i].y;
		searchPoint.z = voxel_filtered->points[i].z;
		searchPoint.r = voxel_filtered->points[i].r;
		searchPoint.g = voxel_filtered->points[i].g;
		searchPoint.b = voxel_filtered->points[i].b;
		int K = 1;//最近邻搜索
		vector<int> pointIdxNKNSearch(K);
		vector<float> pointNKNSquaredDistance(K);
		if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {

			inds->indices.push_back(pointIdxNKNSearch[0]);
		}

	}
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr final_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::copyPointCloud(*cloud, inds->indices, *final_filtered);
	//---------------------构造超体素-------------------------------------
	//-NT禁用单视角转换，对无序点云默认禁用该选项（但用户可以设置为真），适用于kinect,xtion深度摄像头获取的
	//点云数据，PCL内部假设该数据原点的坐标原点为相机位置且z轴正方向于相机朝向一致。
	bool disable_transform = pcl::console::find_switch(argc, argv, "--NT");
	//-v 设置体素大小，该设置决定底层八叉树的叶子尺寸。
	float voxel_resolution = 0.005f;
	bool voxel_res_specified = pcl::console::find_switch(argc, argv, "-v");
	if (voxel_res_specified)
		pcl::console::parse(argc, argv, "-v", voxel_resolution);
	//-s 设置种子大小，该设置决定超体素的大小。
	float seed_resolution = 0.009f;
	bool seed_res_specified = pcl::console::find_switch(argc, argv, "-s");
	if (seed_res_specified)
		pcl::console::parse(argc, argv, "-s", seed_resolution);
	//-c 设置颜色在距离测试公式中的权重，也就是说颜色影响超体素分割结果的比重。
	float color_importance = 0.0f;
	if (pcl::console::find_switch(argc, argv, "-c"))
		pcl::console::parse(argc, argv, "-c", color_importance);
	//-z设置空间距离在距离测试公式中的权重。较高的值将会导致非常规则的超级体素，较低的值产生的体素会按照法线
	//和/或颜色建立分割线，但形状会不规则。
	float spatial_importance = 0.1f;
	if (pcl::console::find_switch(argc, argv, "-z"))
		pcl::console::parse(argc, argv, "-z", spatial_importance);
	//-n设置法向量的权重， 也就是说表面法向批影响超体素分割结果的比重。
	float normal_importance = 0.9f;
	if (pcl::console::find_switch(argc, argv, "-n"))
		pcl::console::parse(argc, argv, "-n", normal_importance);

	pcl::SupervoxelClustering<PointT> super(voxel_resolution, seed_resolution);
	if (disable_transform)
		super.setUseSingleCameraTransform(false);
	super.setInputCloud(final_filtered);
	super.setColorImportance(color_importance);
	super.setSpatialImportance(spatial_importance);
	super.setNormalImportance(normal_importance);

	std::map <std::uint32_t, pcl::Supervoxel<PointT>::Ptr > supervoxel_clusters;
	super.extract(supervoxel_clusters);
	std::multimap<std::uint32_t, std::uint32_t> supervoxel_adjacency;
	super.getSupervoxelAdjacency(supervoxel_adjacency); //读取多重映射容器supervoxel_adjacency构造的邻接图
	cout << "超体素构建完毕" << endl;
	pcl::PointCloud<pcl::PointXYZL>::Ptr sv_labeled_cloud = super.getLabeledCloud();

	//超尺寸检测
	supersize::SuperSize<PointT> supersize;
	supersize.set_input_supervoxels(supervoxel_clusters, supervoxel_adjacency);
	supersize.input_labeled_cloud(sv_labeled_cloud);
	supersize.segment();
	pcl::PointCloud<pcl::PointXYZL>::Ptr supersize_labeled_cloud = sv_labeled_cloud->makeShared();
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudrgb(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_bou_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_public_bou_rgb(new pcl::PointCloud<pcl::PointXYZRGB>);
	supersize.relabel_cloud(*supersize_labeled_cloud);
	supersize.get_recolor_cloud(*supersize_labeled_cloud, *cloudrgb);
	supersize.get_color_bou_cloud(*sv_labeled_cloud, *cloud_bou_rgb);
	supersize.get_color_public_bou_cloud(*sv_labeled_cloud, *cloud_public_bou_rgb);
	supersize.cal_supersize_properties(*supersize_labeled_cloud);
	if(debug==1)
	{
		pcl::io::savePCDFile("recolor.pcd", *cloudrgb);
	}
	//pcl::io::savePCDFile("recolor.pcd", *cloudrgb);
	//pcl::io::savePCDFile("supersizelabeledcloud.pcd" ,*supersize_labeled_cloud);
	//pcl::io::savePCDFile("recolor_bou.pcd", *cloud_bou_rgb);
	//pcl::io::savePCDFile("recolor_public_bou.pcd", *cloud_public_bou_rgb);
	cout << "点云分割完毕" << endl;

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
	int W = cloud->width;
	int H = cloud->height;
	if (results_2d.size() != 0)
	{
		for (auto temp : results_2d)
		{
			for (int i = temp.umin; i < temp.umax; i++)
			{
				for (int j = temp.vmin; j < temp.vmax; j++)
				{
					cloud->points[(j * W) + i].x = NAN;
					cloud->points[(j * W) + i].y = NAN;
					cloud->points[(j * W) + i].z = NAN;
				}
			}
		}
	}
	//pcl::io::savePCDFile("cc.pcd", *cloud);
	//厚度提取
	IntegralImageNormalEstimation<PointXYZRGB, Normal> ne;
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
	oed.setHCCannyHighThreshold(1.15);//参数越大点约少
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
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudXYZ(new pcl::PointCloud<pcl::PointXYZ>());
	for (int i = 0; i < cloud->points.size(); i++)
	{
		pcl::PointXYZ p;
		p.x = cloud->points[i].x;
		p.y = cloud->points[i].y;
		p.z = cloud->points[i].z;
		cloudXYZ->points.push_back(p);
	}
	cloudXYZ->width = cloud->points.size();
	cloudXYZ->height = 1;
	cloudXYZ->is_dense = true;
	/*if (nan_boundary_edges->size() > 0)
	{
		pcl::io::savePCDFileASCII("nan_boundary_edges.pcd", *nan_boundary_edges);
	}
	if (occluding_edges->size() > 0)
	{
		pcl::io::savePCDFileASCII("occluding_edges.pcd", *occluding_edges);
	}
	if (occluded_edges->size() > 0)
	{
		pcl::io::savePCDFileASCII("occluded_edges.pcd", *occluded_edges);
	}
	if (high_curvature_edges->size() > 0)
	{
		pcl::io::savePCDFileASCII("high_curvature_edges.pcd", *high_curvature_edges);
	}
	if (rgb_edges->size() > 0)
	{
		pcl::io::savePCDFileASCII("rgb_edges.pcd", *rgb_edges);
	}*/

	//调试:把前景赋给背景这样就不用改后边参数了
	occluded_edges = occluding_edges;


	//计算方向
	PointDirectionPtr A(new pointDirection());
	PointDirectionPtr B(new pointDirection());
	vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> seg_cloud_A, seg_cloud_B, seg_cloud_A_all, seg_cloud_B_all;
	vector<PointDirectionPtr> seg_direction_A, seg_direction_B, seg_direction_A_all, seg_direction_B_all;
	cal_fangxiang(occluded_edges, A);
	cal_fangxiang(high_curvature_edges, B);
	cout << "方向计算完毕" << endl;
	//进行分割
	vector<pcl::PointIndices> cluster_indices_d;
	vector<pcl::PointIndices> cluster_indices_c;
	cluster_indices_d = cal_julie("深度突变边界", 0.016, occluded_edges, A, seg_cloud_A, seg_direction_A, seg_cloud_A_all, seg_direction_A_all);
	cluster_indices_c = cal_julie("曲率突变边界", 0.025, high_curvature_edges, B, seg_cloud_B, seg_direction_B, seg_cloud_B_all, seg_direction_B_all);
	cout << "聚类完毕" << endl;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr all_thickness(new pcl::PointCloud<pcl::PointXYZRGB>);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr all_thickness_self(new pcl::PointCloud<pcl::PointXYZRGB>);
	std::vector<vector<pair<int, int>>> AABB_2d;
	std::vector<pair<float, float>> Statistics;
	std::map < std::uint32_t, std::vector<float>> label_with_aabb;
	//输出直方图
	Statistics = tongji_houdu(select_min, select_max, select_step, output_name, cloudXYZ, occluded_edges, high_curvature_edges, A, B, seg_cloud_A, seg_cloud_B, seg_direction_A, seg_direction_B\
		, seg_cloud_A_all, seg_cloud_B_all, seg_direction_A_all, seg_direction_B_all, AABB_2d, cluster_indices_d, max_z, max_view_z ,cloud, supersize_labeled_cloud, supersize, label_with_aabb,debug);
	//输出包围盒的像素坐标(分割的大件）
	ofstream out(output2d_name + ".txt");
	int i = 0;
	for (auto ite=label_with_aabb.begin(); ite!= label_with_aabb.end();ite++)
	{
		int min_x, min_y, max_x, max_y;
		int u_min, u_max, v_min, v_max;
		for (int j = 0; j < cloud->size(); j++)
		{
			if (cloud->points[j].x == ite->second[0])
			{
				min_x = j;
			}
			if (cloud->points[j].y == ite->second[1])
			{
				min_y = j;
			}
			if (cloud->points[j].x == ite->second[2])
			{
				max_x = j;
			}
			if (cloud->points[j].y == ite->second[3])
			{
				max_y = j;
			}
		}
		v_min = ceil(min_y / W);
		u_min = min_x % W;
		v_max = ceil(max_y / W);
		u_max = max_x % W;
		out << u_min << " " << v_min << " " << u_max << " " << v_max << " " << (Statistics[i].first) * 1000 << endl;
		i++;
	}

	////输出包围盒的像素坐标
	//int W = cloud->width;
	//int H = cloud->height;
	//std::vector<int> one_seg_2d(4);
	//std::vector<vector<int>> seg_2d_aabb;
	//for (int i = 0; i < AABB_2d.size(); i++)
	//{
	//	for (int j = 0; j < 4; j++)
	//	{
	//		if (AABB_2d[i][j].first == 1)
	//		{
	//			one_seg_2d[j] = label_indices[2].indices[AABB_2d[i][j].second];
	//		}
	//		else if (AABB_2d[i][j].first == 2)
	//		{
	//			one_seg_2d[j] = label_indices[3].indices[AABB_2d[i][j].second];
	//		}
	//	}
	//	seg_2d_aabb.push_back(one_seg_2d);
	//}
	//ofstream out(output2d_name + ".txt");
	//int count = 0;
	//for (auto a : seg_2d_aabb)
	//{
	//	int u_min, u_max, v_min, v_max;
	//	v_min = ceil(a[1] / W);
	//	u_min = a[0] % W;
	//	v_max = ceil(a[3] / W);
	//	u_max = a[2] % W;
	//	out << u_min << " " << v_min << " " << u_max << " " << v_max << " " << (Statistics[count].first) * 1000 << endl;
	//	count++;
	//}
	out.close();
	cout << "输出包围盒坐标完毕" << endl;

	

}
