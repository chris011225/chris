#pragma once
#include "kMeans.cpp"
#include "thickness_cal.h"
#include <pcl/point_cloud.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/point_types.h>
#include <pcl/common/point_tests.h> 
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/distances.h>
#include <pcl/filters/extract_indices.h>       
#include <pcl/segmentation/extract_clusters.h> 
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/johnson_all_pairs_shortest.hpp>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include<time.h>
#include <queue>
#include <list>
#include <cmath>
#include <ctime>
#include <vector>
#include <pcl/search/organized.h>
#include<algorithm>
#include<pcl/point_types.h>
#include<pcl/io/pcd_io.h>
#include<pcl/visualization/pcl_visualizer.h>
#include<math.h>
#include<iostream>
using namespace std;
using namespace pcl;
using namespace Eigen;
typedef pcl::PointCloud<pcl::PointXYZ> Cloud;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
thickness::Thickness<PointT>::Thickness() :
	m_min_pts_per_cluster_(1),
	m_max_pts_per_cluster_(std::numeric_limits<int>::max())
{
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
thickness::Thickness<PointT>::~Thickness()
{

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::set_min_cluster_size(int min_cluster_size)
{
	m_min_pts_per_cluster_ = min_cluster_size;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::set_max_cluster_size(int max_cluster_size)
{
	m_max_pts_per_cluster_ = max_cluster_size;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::set_search_method(const KdTreePtr& tree)
{
	m_search_ = tree;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::input_cloud(const PointCloudConstPtr& cloudA, const PointCloudConstPtr& cloudB)
{
	m_A_cloud_ = cloudA;
	m_B_cloud_ = cloudB;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::input_index(const std::vector<std::vector<int>>& A_index, const std::vector<std::vector<int>>& B_index)
{
	m_A_index = A_index;
	m_B_index = B_index;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::input_original_cloud(const PointCloudConstPtr& cloud)
{
	m_original_cloud_ = cloud;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::input_cloud_direction(const PointDirectionPtr& CloudDirectionPtrA, const PointDirectionPtr& CloudDirectionPtrB)
{
	m_A_cloud_direction_ = CloudDirectionPtrA;
	m_B_cloud_direction_ = CloudDirectionPtrB;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::clustering()
{
	bool cluster_is_possible = this->prepare();
	if (!cluster_is_possible)
	{
		std::cout << "未检测出类内厚度特征！！" << std::endl;
		return;
	}
	initialization1();

	//第一次计算距离矩阵
	m_dis_matrix_.resize(m_node_list_A.size());

#pragma omp parallel for
	for (int i = 0; i < m_node_list_A.size(); i++)
	{
		m_dis_matrix_[i].resize(m_node_list_A.size());

	}

#pragma omp parallel for
	for (int i = 0; i < m_node_list_A.size(); i++)
	{
		for (int j = 0; j < m_node_list_A.size(); j++)
		{
			if (i < j)
			{
				Node* A = m_node_list_A[i];
				Node* B = m_node_list_A[j];
				float dis_curr_min = cal_min_dis(A, B);
				float dis_curr_fit = cal_fit_line_dis(A, B);
				float dis_add_weight = cal_line_weight_dis(dis_curr_min, dis_curr_fit);
				m_dis_matrix_[i][j] = dis_add_weight;

			}
		}
	}

	while (true)
	{
		float MaxDst = 0;
		int find_i, find_j; //用于记录最小两簇的索引
		for (int i = 0; i < m_dis_matrix_.size(); i++) //遍历的方法找到距离最小的两簇
		{
			for (int j = i + 1; j < m_dis_matrix_[i].size(); j++)
			{
				if (m_dis_matrix_[i][j] > MaxDst)
				{
					find_i = i;
					find_j = j;
					MaxDst = m_dis_matrix_[i][j];
				}
			}
		}
		//进行聚类前判断加权间距是否大于阈值
		if (MaxDst < m_th_cen_)
		{
			break;
		}
		//更新nodeA,把nodeB加到A上，成为新的node
		Node* temp_a_node = m_node_list_A[find_i];
		Node* temp_b_node = m_node_list_A[find_j];
		Node* temp_node = new Node(temp_a_node, temp_b_node);

		//更新node点云
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_a(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_b(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_new_a(new pcl::PointCloud<pcl::PointXYZ>);
		cloud_a = temp_a_node->m_cloud_;
		cloud_b = temp_b_node->m_cloud_;
		*cloud_new_a = (*cloud_a) + (*cloud_b);
		temp_node->m_cloud_ = cloud_new_a;

		if (cloud_new_a->size() >= 3)
		{
			//更新node长度和粗度
			pcl::PointXYZ aabb_max, aabb_min, obb_max, obb_min;
			float first_size, second_size, third_size;
			pcl::PointXYZ position_OBB;
			Eigen::Matrix3f rotational_matrix_OBB;
			pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
			feature_extractor.setInputCloud(cloud_new_a);
			feature_extractor.compute();
			feature_extractor.getAABB(aabb_min, aabb_max);
			feature_extractor.getOBB(obb_min, obb_max, position_OBB, rotational_matrix_OBB);
			first_size = abs(obb_max.x - obb_min.x);
			second_size = abs(obb_max.y - obb_min.y);
			third_size = abs(obb_max.z - obb_min.z);
			if (third_size > second_size) {
				float temp = third_size;
				third_size = second_size;
				second_size = temp;
			}
			if (second_size > first_size) {
				float temp = second_size;
				second_size = first_size;
				first_size = temp;
			}
			if (third_size > second_size) {
				float temp = third_size;
				third_size = second_size;
				second_size = temp;
			}
			temp_node->m_length_ = first_size;
			temp_node->m_roughness_ = second_size;



			//更新node点密度
			temp_node->m_density_ = static_cast<int>(cloud_new_a->size()) / first_size;


			/*
				if (static_cast<int>(cloud_new_a->size()) / first_size > 2000)
				{
					cout << "m_density=" << static_cast<int>(cloud_new_a->size()) / first_size << endl;

				}*/



				//更新node拟合的直线点向式方程
			pcl::PCA<pcl::PointXYZ> pca;
			pca.setInputCloud(cloud_new_a);
			Eigen::RowVector3f V1 = pca.getEigenVectors().col(0);
			Eigen::Vector4f line_point;
			Eigen::Vector4f line_dir;
			line_point = pca.getMean();
			line_dir[0] = V1[0];
			line_dir[1] = V1[1];
			line_dir[2] = V1[2];
			line_dir[3] = 0;
			temp_node->m_pca_point = line_point;
			temp_node->m_pca_dir = line_dir;
		}


		//更新node的level
		temp_node->m_level_ = temp_a_node->m_level_ + temp_b_node->m_level_ + 1;

		//更新node内点个数
		temp_node->m_cluster_num_ = temp_a_node->m_cluster_num_ + temp_b_node->m_cluster_num_;

		//更新node类别
		temp_node->m_type_ = 0;

		//更新node端点
		pcl::PointXYZ pmin, pmax;
		float L = pcl::getMaxSegment(*cloud_new_a, pmin, pmax);
		temp_node->m_pmin_ = pmin;
		temp_node->m_pmax_ = pmax;

		//更新node长度点密度
		if (cloud_new_a->size() < 3)
		{
			temp_node->m_length_ = L;
			temp_node->m_density_ = L / 2;
			Eigen::Vector4f line_point;
			Eigen::Vector4f line_dir;
			line_dir[0] = pmin.x - pmax.x;
			line_dir[1] = pmin.y - pmax.y;
			line_dir[2] = pmin.z - pmax.z;
			line_dir[3] = 0;
			line_point[0] = pmax.x;
			line_point[1] = pmax.y;
			line_point[2] = pmax.z;
			line_point[3] = 0;
			temp_node->m_pca_point = line_point;
			temp_node->m_pca_dir = line_dir;
		}

		//更新node不带顺序的index列表
		auto node_index_a = temp_a_node->m_node_index_list;
		auto node_index_b = temp_b_node->m_node_index_list;
		for (auto i : node_index_b)
		{
			node_index_a.push_back(i);
		}
		temp_node->m_node_index_list = node_index_a;

		//更新node起始点和终点与带顺序的index列表


		m_node_list_A.erase(m_node_list_A.cbegin() + find_j);
		m_node_list_A[find_i] = temp_node;

		//更新距离矩阵
		m_dis_matrix_.erase(m_dis_matrix_.cbegin() + find_j);
#pragma omp parallel for
		for (int i = 0; i < m_dis_matrix_.size(); i++)
		{
			m_dis_matrix_[i].erase(m_dis_matrix_[i].cbegin() + find_j);
		}
#pragma omp parallel for
		for (int i = 0; i < m_node_list_A.size(); i++)
		{
#pragma omp parallel for
			for (int j = 0; j < m_node_list_A.size(); j++)
			{
				if (j == find_i && j > i)
				{
					Node* A = m_node_list_A[i];
					Node* B = m_node_list_A[j];
					float dis_curr_min = cal_min_dis(A, B);
					float dis_curr_fit = cal_fit_line_dis(A, B);
					float dis_add_weight = cal_line_weight_dis(dis_curr_min, dis_curr_fit);
					m_dis_matrix_[i][j] = dis_add_weight;
				}
			}
			if (i == find_i)
			{
#pragma omp parallel for
				for (int k = 0; k < m_dis_matrix_[i].size(); k++)
				{
					if (k > i)
					{
						Node* A = m_node_list_A[i];
						Node* B = m_node_list_A[k];
						float dis_curr_min = cal_min_dis(A, B);
						float dis_curr_fit = cal_fit_line_dis(A, B);
						float dis_add_weight = cal_line_weight_dis(dis_curr_min, dis_curr_fit);
						m_dis_matrix_[i][k] = dis_add_weight;
					}
					else
					{
						m_dis_matrix_[i][k] = 0;
					}
				}
			}

		}

	}

	//	//第2次计算距离矩阵
	m_dis_matrix_.clear();
	m_dis_matrix_.resize(m_node_list_B.size());

#pragma omp parallel for
	for (int i = 0; i < m_node_list_B.size(); i++)
	{
		m_dis_matrix_[i].resize(m_node_list_B.size());
	}

#pragma omp parallel for
	for (int i = 0; i < m_node_list_B.size(); i++)
	{
		for (int j = 0; j < m_node_list_B.size(); j++)
		{
			if (i < j)
			{
				Node* A = m_node_list_B[i];
				Node* B = m_node_list_B[j];
				float dis_curr_min = cal_min_dis(A, B);
				float dis_curr_fit = cal_fit_line_dis(A, B);
				float dis_add_weight = cal_line_weight_dis(dis_curr_min, dis_curr_fit);
				m_dis_matrix_[i][j] = dis_add_weight;

			}
		}
	}

	while (true)
	{
		float MaxDst = 0;
		int find_i, find_j; //用于记录最小两簇的索引
		for (int i = 0; i < m_dis_matrix_.size(); i++) //遍历的方法找到距离最小的两簇
		{
			for (int j = i + 1; j < m_dis_matrix_[i].size(); j++)
			{
				if (m_dis_matrix_[i][j] > MaxDst)
				{
					find_i = i;
					find_j = j;
					MaxDst = m_dis_matrix_[i][j];
				}
			}
		}
		//进行聚类前判断加权间距是否大于阈值
		if (MaxDst < m_th_cen_)
		{
			break;
		}
		//更新nodeA,把nodeB加到A上，成为新的node
		Node* temp_a_node = m_node_list_B[find_i];
		Node* temp_b_node = m_node_list_B[find_j];
		Node* temp_node = new Node(temp_a_node, temp_b_node);

		//更新node点云
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_a(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_b(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_new_a(new pcl::PointCloud<pcl::PointXYZ>);
		cloud_a = temp_a_node->m_cloud_;
		cloud_b = temp_b_node->m_cloud_;
		*cloud_new_a = (*cloud_a) + (*cloud_b);
		temp_node->m_cloud_ = cloud_new_a;

		if (cloud_new_a->size() >= 3)
		{
			//更新node长度和粗度
			pcl::PointXYZ aabb_max, aabb_min, obb_max, obb_min;
			float first_size, second_size, third_size;
			pcl::PointXYZ position_OBB;
			Eigen::Matrix3f rotational_matrix_OBB;
			pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
			feature_extractor.setInputCloud(cloud_new_a);
			feature_extractor.compute();
			feature_extractor.getAABB(aabb_min, aabb_max);
			feature_extractor.getOBB(obb_min, obb_max, position_OBB, rotational_matrix_OBB);
			first_size = abs(obb_max.x - obb_min.x);
			second_size = abs(obb_max.y - obb_min.y);
			third_size = abs(obb_max.z - obb_min.z);
			if (third_size > second_size) {
				float temp = third_size;
				third_size = second_size;
				second_size = temp;
			}
			if (second_size > first_size) {
				float temp = second_size;
				second_size = first_size;
				first_size = temp;
			}
			if (third_size > second_size) {
				float temp = third_size;
				third_size = second_size;
				second_size = temp;
			}
			temp_node->m_length_ = first_size;
			temp_node->m_roughness_ = second_size;

			//更新node点密度
			temp_node->m_density_ = static_cast<int>(cloud_new_a->size()) / first_size;

			//更新node拟合的直线点向式方程-------------------
			pcl::PCA<pcl::PointXYZ> pca;
			pca.setInputCloud(cloud_new_a);
			Eigen::RowVector3f V1 = pca.getEigenVectors().col(0);
			Eigen::Vector4f line_point;
			Eigen::Vector4f line_dir;
			line_point = pca.getMean();
			line_dir[0] = V1[0];
			line_dir[1] = V1[1];
			line_dir[2] = V1[2];
			line_dir[3] = 0;
			temp_node->m_pca_point = line_point;
			temp_node->m_pca_dir = line_dir;
		}


		//更新node的level
		temp_node->m_level_ = temp_a_node->m_level_ + temp_b_node->m_level_ + 1;

		//更新node内点个数
		temp_node->m_cluster_num_ = temp_a_node->m_cluster_num_ + temp_b_node->m_cluster_num_;

		//更新node类别
		temp_node->m_type_ = 1;

		//更新node端点
		pcl::PointXYZ pmin, pmax;
		float L = pcl::getMaxSegment(*cloud_new_a, pmin, pmax);
		temp_node->m_pmin_ = pmin;
		temp_node->m_pmax_ = pmax;

		//更新node长度点密度
		if (cloud_new_a->size() < 3)
		{
			temp_node->m_length_ = L;
			temp_node->m_density_ = L / 2;
			Eigen::Vector4f line_point;
			Eigen::Vector4f line_dir;
			line_dir[0] = pmin.x - pmax.x;
			line_dir[1] = pmin.y - pmax.y;
			line_dir[2] = pmin.z - pmax.z;
			line_dir[3] = 0;
			line_point[0] = pmax.x;
			line_point[1] = pmax.y;
			line_point[2] = pmax.z;
			line_point[3] = 0;
			temp_node->m_pca_point = line_point;
			temp_node->m_pca_dir = line_dir;
		}

		//更新node不带顺序的index列表
		auto node_index_a = temp_a_node->m_node_index_list;
		auto node_index_b = temp_b_node->m_node_index_list;
		for (auto i : node_index_b)
		{
			node_index_a.push_back(i);
		}
		temp_node->m_node_index_list = node_index_a;

		//更新node起始点和终点与带顺序的index列表


		m_node_list_B.erase(m_node_list_B.cbegin() + find_j);
		m_node_list_B[find_i] = temp_node;

		//更新距离矩阵
		m_dis_matrix_.erase(m_dis_matrix_.cbegin() + find_j);
#pragma omp parallel for
		for (int i = 0; i < m_dis_matrix_.size(); i++)
		{
			m_dis_matrix_[i].erase(m_dis_matrix_[i].cbegin() + find_j);
		}
#pragma omp parallel for
		for (int i = 0; i < m_node_list_B.size(); i++)
		{
#pragma omp parallel for
			for (int j = 0; j < m_node_list_B.size(); j++)
			{
				if (j == find_i && j > i)
				{
					Node* A = m_node_list_B[i];
					Node* B = m_node_list_B[j];
					float dis_curr_min = cal_min_dis(A, B);
					float dis_curr_fit = cal_fit_line_dis(A, B);
					float dis_add_weight = cal_line_weight_dis(dis_curr_min, dis_curr_fit);
					m_dis_matrix_[i][j] = dis_add_weight;
				}
			}
			if (i == find_i)
			{
#pragma omp parallel for
				for (int k = 0; k < m_dis_matrix_[i].size(); k++)
				{
					if (k > i)
					{
						Node* A = m_node_list_B[i];
						Node* B = m_node_list_B[k];
						float dis_curr_min = cal_min_dis(A, B);
						float dis_curr_fit = cal_fit_line_dis(A, B);
						float dis_add_weight = cal_line_weight_dis(dis_curr_min, dis_curr_fit);
						m_dis_matrix_[i][k] = dis_add_weight;
					}
					else
					{
						m_dis_matrix_[i][k] = 0;
					}
				}
			}

		}

	}

	for (auto node : m_node_list_B)
	{
		m_node_list_A.push_back(node);
	}
	//filter
	for (auto onenode = m_node_list_A.cbegin(); onenode != m_node_list_A.cend();)
	{
		if ((*onenode)->m_cluster_num_ < 5)
		{
			onenode = m_node_list_A.erase(onenode);
		}
		else
		{
			onenode++;
		}
	}
	//去除点密度小于1000的点
	/*for (auto onenode = m_node_list_A.cbegin(); onenode != m_node_list_A.cend();)
	{
		if ((*onenode)->m_density_<1000)
		{
			onenode = m_node_list_A.erase(onenode);
		}
		else
		{
			onenode++;
		}
	}*/

	//测试用,用于保存聚类后的点云
	/*pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_after(new pcl::PointCloud<pcl::PointXYZRGB>);
	for (auto onenode : m_node_list_A)
	{
		int Random_color_r, Random_color_g, Random_color_b;
		Random_color_r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		temp_cloud = onenode->m_cloud_;
		for (auto point : temp_cloud->points)
		{
			pcl::PointXYZRGB tempa;
			tempa.x = point.x;
			tempa.y = point.y;
			tempa.z = point.z;
			tempa.r = Random_color_r;
			tempa.g = Random_color_g;
			tempa.b = Random_color_b;
			cloud_after->push_back(tempa);
		}
	}
		cloud_after->width = cloud_after->points.size();
		cloud_after->height = 1;
		cloud_after->is_dense = true;
		stringstream ss;
		ss << "聚类后" << ".pcd";
		pcl::PCDWriter writer;
		writer.write<pcl::PointXYZRGB>(ss.str(), *cloud_after, false);*/

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> bool thickness::Thickness<PointT>::prepare()
{
	if (m_A_cloud_ == nullptr || m_B_cloud_ == nullptr)
		return(false);
	if (m_A_cloud_direction_ == nullptr || m_B_cloud_direction_ == nullptr)
		return(false);
	if (!m_search_)
	{
		m_search_.reset(new pcl::search::KdTree<PointT>);
		return(true);
	}
	else
		return(true);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
//A是曲率，B是深度
thickness::Thickness<PointT>::initialization1()
{
	for (auto vindex : m_A_index)
	{
		if (vindex.size() == 0)
		{
			continue;
		}
		Node* temp_node = new Node();
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto i : vindex)
		{
			cloud->push_back(m_A_cloud_->points[i]);
		}
		temp_node->m_cloud_ = cloud;
		if (cloud->size() >= 3)
		{
			//更新node长度和粗度
			pcl::PointXYZ aabb_max, aabb_min, obb_max, obb_min;
			float first_size, second_size, third_size;
			pcl::PointXYZ position_OBB;
			Eigen::Matrix3f rotational_matrix_OBB;
			pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;//有向包围盒OBB,坐标轴对齐包围盒AABB
			feature_extractor.setInputCloud(cloud);
			feature_extractor.compute();
			feature_extractor.getAABB(aabb_min, aabb_max);
			feature_extractor.getOBB(obb_min, obb_max, position_OBB, rotational_matrix_OBB);
			first_size = abs(obb_max.x - obb_min.x);
			second_size = abs(obb_max.y - obb_min.y);
			third_size = abs(obb_max.z - obb_min.z);
			if (third_size > second_size) {
				float temp = third_size;
				third_size = second_size;
				second_size = temp;
			}
			if (second_size > first_size) {
				float temp = second_size;
				second_size = first_size;
				first_size = temp;
			}
			if (third_size > second_size) {
				float temp = third_size;
				third_size = second_size;
				second_size = temp;
			}
			temp_node->m_length_ = first_size;//长度（最长）
			temp_node->m_roughness_ = second_size;//粗度（第二长）

			//更新node点密度
			temp_node->m_density_ = static_cast<int>(cloud->size()) / first_size;

			//更新node拟合的直线点向式方程
			pcl::PCA<pcl::PointXYZ> pca;
			pca.setInputCloud(cloud);
			Eigen::RowVector3f V1 = pca.getEigenVectors().col(0);
			Eigen::Vector4f line_point;
			Eigen::Vector4f line_dir;
			line_point = pca.getMean();
			line_dir[0] = V1[0];
			line_dir[1] = V1[1];
			line_dir[2] = V1[2];
			line_dir[3] = 0;
			temp_node->m_pca_point = line_point;
			temp_node->m_pca_dir = line_dir;
		}


		//更新node的level
		temp_node->m_level_ = 0;

		//更新node内点个数
		temp_node->m_cluster_num_ = cloud->size();

		//更新node类别
		temp_node->m_type_ = 0;

		//更新node端点
		pcl::PointXYZ pmin, pmax;
		float L = pcl::getMaxSegment(*cloud, pmin, pmax);
		temp_node->m_pmin_ = pmin;
		temp_node->m_pmax_ = pmax;

		//更新node长度点密度
		if (cloud->size() < 3)
		{
			temp_node->m_length_ = L;
			temp_node->m_density_ = L / 2;
			Eigen::Vector4f line_point;
			Eigen::Vector4f line_dir;
			int i = vindex[0];
			line_dir[0] = (*m_A_cloud_direction_)[i][0];
			line_dir[1] = (*m_A_cloud_direction_)[i][1];
			line_dir[2] = (*m_A_cloud_direction_)[i][2];
			line_dir[3] = 0;
			line_point[0] = m_A_cloud_->points[i].x;
			line_point[1] = m_A_cloud_->points[i].y;
			line_point[2] = m_A_cloud_->points[i].z;
			line_point[3] = 0;
			temp_node->m_pca_point = line_point;
			temp_node->m_pca_dir = line_dir;
		}


		//更新node不带顺序的index列表
		std::deque<int> node_index_list;
		for (auto i : vindex)
		{
			node_index_list.push_back(i);
		}
		temp_node->m_node_index_list = node_index_list;
		m_node_list_A.push_back(temp_node);
	}


	for (auto vindex : m_B_index)
	{
		if (vindex.size() == 0)
		{
			continue;
		}
		Node* temp_node = new Node();
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		for (auto i : vindex)
		{
			cloud->push_back(m_B_cloud_->points[i]);
		}
		temp_node->m_cloud_ = cloud;
		if (cloud->size() >= 3)
		{
			//更新node长度和粗度
			pcl::PointXYZ aabb_max, aabb_min, obb_max, obb_min;
			float first_size, second_size, third_size;
			pcl::PointXYZ position_OBB;
			Eigen::Matrix3f rotational_matrix_OBB;
			pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
			feature_extractor.setInputCloud(cloud);
			feature_extractor.compute();
			feature_extractor.getAABB(aabb_min, aabb_max);
			feature_extractor.getOBB(obb_min, obb_max, position_OBB, rotational_matrix_OBB);
			first_size = abs(obb_max.x - obb_min.x);
			second_size = abs(obb_max.y - obb_min.y);
			third_size = abs(obb_max.z - obb_min.z);
			if (third_size > second_size) {
				float temp = third_size;
				third_size = second_size;
				second_size = temp;
			}
			if (second_size > first_size) {
				float temp = second_size;
				second_size = first_size;
				first_size = temp;
			}
			if (third_size > second_size) {
				float temp = third_size;
				third_size = second_size;
				second_size = temp;
			}
			temp_node->m_length_ = first_size;
			temp_node->m_roughness_ = second_size;

			//更新node点密度
			temp_node->m_density_ = static_cast<int>(cloud->size()) / first_size;

			//更新node拟合的直线点向式方程
			pcl::PCA<pcl::PointXYZ> pca;
			pca.setInputCloud(cloud);
			Eigen::RowVector3f V1 = pca.getEigenVectors().col(0);
			Eigen::Vector4f line_point;
			Eigen::Vector4f line_dir;
			line_point = pca.getMean();
			line_dir[0] = V1[0];
			line_dir[1] = V1[1];
			line_dir[2] = V1[2];
			line_dir[3] = 0;
			temp_node->m_pca_point = line_point;
			temp_node->m_pca_dir = line_dir;
		}



		//更新node的level
		temp_node->m_level_ = 0;

		//更新node内点个数
		temp_node->m_cluster_num_ = cloud->size();

		//更新node类别
		temp_node->m_type_ = 1;

		//更新node端点
		pcl::PointXYZ pmin, pmax;
		float L = pcl::getMaxSegment(*cloud, pmin, pmax);
		temp_node->m_pmin_ = pmin;
		temp_node->m_pmax_ = pmax;

		//更新node长度点密度
		if (cloud->size() < 3)
		{
			temp_node->m_length_ = L;
			temp_node->m_density_ = L / 2;
			Eigen::Vector4f line_point;
			Eigen::Vector4f line_dir;
			int i = vindex[0];
			line_dir[0] = (*m_B_cloud_direction_)[i][0];
			line_dir[1] = (*m_B_cloud_direction_)[i][1];
			line_dir[2] = (*m_B_cloud_direction_)[i][2];
			line_dir[3] = 0;
			line_point[0] = m_B_cloud_->points[i].x;
			line_point[1] = m_B_cloud_->points[i].y;
			line_point[2] = m_B_cloud_->points[i].z;
			line_point[3] = 0;
			temp_node->m_pca_point = line_point;
			temp_node->m_pca_dir = line_dir;
		}


		//更新node不带顺序的index列表
		std::deque<int> node_index_list;
		for (auto i : vindex)
		{
			node_index_list.push_back(i);
		}
		temp_node->m_node_index_list = node_index_list;
		m_node_list_B.push_back(temp_node);
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::initialization()
{
	for (int i = 0; i < static_cast<int>(m_A_cloud_->size()); i++)
	{
		Node* temp_node = new Node();
		Eigen::Vector4f temp_p, temp_dir;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		cloud->push_back(m_A_cloud_->points[i]);
		temp_node->m_pmin_ = m_A_cloud_->points[i];
		temp_node->m_pmax_ = m_A_cloud_->points[i];
		temp_node->m_cloud_ = cloud;
		temp_node->m_type_ = 0;
		temp_node->m_node_index_list.push_back(i);
		temp_node->m_level_ = 0;//初始的聚类层数，0意味是在最底层
		temp_node->m_cluster_num_ = 1;
		temp_node->m_begin_index_ = i;
		temp_node->m_end_index_ = i;
		temp_node->m_length_ = 0;
		temp_node->m_roughness_ = 0;
		temp_node->m_density_ = 0;
		temp_p[0] = m_A_cloud_->points[i].x;
		temp_p[1] = m_A_cloud_->points[i].y;
		temp_p[2] = m_A_cloud_->points[i].z;
		temp_p[3] = 0;
		temp_dir[0] = (*m_A_cloud_direction_)[i][0];
		temp_dir[1] = (*m_A_cloud_direction_)[i][1];
		temp_dir[2] = (*m_A_cloud_direction_)[i][2];
		temp_dir[3] = 0;
		temp_node->m_pca_point = temp_p;
		temp_node->m_pca_dir = temp_dir;
		m_node_list_A.push_back(temp_node);
	}
	for (int i = 0; i < static_cast<int>(m_B_cloud_->size()); i++)
	{
		Node* temp_node = new Node();
		Eigen::Vector4f temp_p, temp_dir;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		cloud->push_back(m_B_cloud_->points[i]);
		temp_node->m_pmin_ = m_B_cloud_->points[i];
		temp_node->m_pmax_ = m_B_cloud_->points[i];
		temp_node->m_cloud_ = cloud;
		temp_node->m_type_ = 1;
		temp_node->m_node_index_list.push_back(i);
		temp_node->m_level_ = 0;//初始的聚类层数，0意味是在最底层
		temp_node->m_cluster_num_ = 1;
		temp_node->m_begin_index_ = i;
		temp_node->m_end_index_ = i;
		temp_node->m_length_ = 0;
		temp_node->m_roughness_ = 0;
		temp_node->m_density_ = 0;
		temp_p[0] = m_B_cloud_->points[i].x;
		temp_p[1] = m_B_cloud_->points[i].y;
		temp_p[2] = m_B_cloud_->points[i].z;
		temp_p[3] = 0;
		temp_dir[0] = (*m_B_cloud_direction_)[i][0];
		temp_dir[1] = (*m_B_cloud_direction_)[i][1];
		temp_dir[2] = (*m_B_cloud_direction_)[i][2];
		temp_dir[3] = 0;
		temp_node->m_pca_point = temp_p;
		temp_node->m_pca_dir = temp_dir;
		m_node_list_B.push_back(temp_node);
	}

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> float
thickness::Thickness<PointT>::cal_fit_line_dis(Node* A, Node* B)
{
	int num_a, num_b;
	float sum_a, sum_b;
	sum_a = 0;
	sum_b = 0;
	Eigen::Vector4f temp_point;
	num_a = A->m_cloud_->size();
	num_b = B->m_cloud_->size();
	for (auto p : A->m_cloud_->points)
	{
		temp_point[0] = p.x;
		temp_point[1] = p.y;
		temp_point[2] = p.z;
		temp_point[3] = 0;
		double dis_p_line = sqrt(pcl::sqrPointToLineDistance(temp_point, B->m_pca_point, B->m_pca_dir));
		sum_a += dis_p_line;
	}
	float avg_a_fit_dis = sum_a / static_cast<float>(num_a);

	for (auto p : B->m_cloud_->points)
	{
		temp_point[0] = p.x;
		temp_point[1] = p.y;
		temp_point[2] = p.z;
		temp_point[3] = 0;
		double dis_p_line = sqrt(pcl::sqrPointToLineDistance(temp_point, A->m_pca_point, A->m_pca_dir));
		sum_b += dis_p_line;
	}
	float avg_b_fit_dis = sum_b / static_cast<float>(num_b);
	float dis = (avg_b_fit_dis + avg_a_fit_dis) / 2;
	return dis;

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> float
thickness::Thickness<PointT>::cal_min_dis(Node* A, Node* B)
{
	float dis1, dis2, dis3, dis4;
	float min_dis;
	vector <double> dis_list(4);
	/*if (A->m_type_ == 0)
	{
		dis1 = pcl::euclideanDistance(m_A_cloud_->points[A->m_begin_index_], m_A_cloud_->points[B->m_begin_index_]);
		dis2 = pcl::euclideanDistance(m_A_cloud_->points[A->m_begin_index_], m_A_cloud_->points[B->m_end_index_]);
		dis3 = pcl::euclideanDistance(m_A_cloud_->points[A->m_end_index_], m_A_cloud_->points[B->m_begin_index_]);
		dis4 = pcl::euclideanDistance(m_A_cloud_->points[A->m_end_index_], m_A_cloud_->points[B->m_end_index_]);
		dis_list[0] = dis1;
		dis_list[1] = dis2;
		dis_list[2] = dis3;
		dis_list[3] = dis4;
		sort(dis_list.begin(), dis_list.end());
		min_dis = dis_list[0];
	}
	else if (A->m_type_ == 1)
	{
		dis1 = pcl::euclideanDistance(m_B_cloud_->points[A->m_begin_index_], m_B_cloud_->points[B->m_begin_index_]);
		dis2 = pcl::euclideanDistance(m_B_cloud_->points[A->m_begin_index_], m_B_cloud_->points[B->m_end_index_]);
		dis3 = pcl::euclideanDistance(m_B_cloud_->points[A->m_end_index_], m_B_cloud_->points[B->m_begin_index_]);
		dis4 = pcl::euclideanDistance(m_B_cloud_->points[A->m_end_index_], m_B_cloud_->points[B->m_end_index_]);
		dis_list[0] = dis1;
		dis_list[1] = dis2;
		dis_list[2] = dis3;
		dis_list[3] = dis4;
		sort(dis_list.begin(), dis_list.end());
		min_dis = dis_list[0];
	}*/

	dis1 = pcl::euclideanDistance(A->m_pmin_, B->m_pmin_);
	dis2 = pcl::euclideanDistance(A->m_pmin_, B->m_pmax_);
	dis3 = pcl::euclideanDistance(A->m_pmax_, B->m_pmin_);
	dis4 = pcl::euclideanDistance(A->m_pmax_, B->m_pmax_);
	dis_list[0] = dis1;
	dis_list[1] = dis2;
	dis_list[2] = dis3;
	dis_list[3] = dis4;
	sort(dis_list.begin(), dis_list.end());
	min_dis = dis_list[0];
	return min_dis;

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> float
thickness::Thickness<PointT>::cal_line_weight_dis(float min_dis, float fit_dis)
{
	float add_weight_dis = sqrt(m_line_w1_ * (min_dis * min_dis) + m_line_w2_ * (fit_dis * fit_dis));
	float dis = exp(-add_weight_dis);
	return dis;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::plane_detection()
{
	//去除点小于3的node
	for (auto onenode = m_node_list_A.cbegin(); onenode != m_node_list_A.cend();)
	{
		if ((*onenode)->m_cloud_->size() < 3)
		{
			onenode = m_node_list_A.erase(onenode);
		}
		else
		{
			onenode++;
		}
	}
	vector<int> is_merge(static_cast<int>(m_node_list_A.size()));
	for (int i = 0; i < static_cast<int>(m_node_list_A.size()); i++)
	{
		Plane temp_plane;
		if (is_merge[i] != 0)
		{
			continue;
		}
		temp_plane.node_list.push_back(m_node_list_A[i]);
		is_merge[i] = 1;
		pcl::PointCloud<pcl::PointXYZ>::Ptr p1_cloud;
		pcl::PointCloud<pcl::PointXYZ>::Ptr new_cloud(new pcl::PointCloud<pcl::PointXYZ>);;
		p1_cloud = m_node_list_A[i]->m_cloud_;
		for (auto p : (*p1_cloud))
		{
			new_cloud->push_back(p);
		}
		Eigen::Vector4d centroid;
		Eigen::Matrix3d covariance_matrix;
		pcl::computeMeanAndCovarianceMatrix(*p1_cloud, covariance_matrix, centroid);//计算点云的标准化协方差矩阵
		Eigen::Matrix3d eigenVectors;
		Eigen::Vector3d eigenValues;
		pcl::eigen33(covariance_matrix, eigenVectors, eigenValues);
		Eigen::Vector3d::Index minRow, minCol;
		eigenValues.minCoeff(&minRow, &minCol);
		Eigen::Vector3d normal = eigenVectors.col(minCol);
		double A = normal[0];
		double B = normal[1];
		double C = normal[2];
		double D = -normal.dot(centroid.head<3>());
		//内循环
		for (int j = 0; j < static_cast<int>(m_node_list_A.size()); j++)
		{
			if (i == j || is_merge[j] != 0)
			{
				continue;
			}
			pcl::PointCloud<pcl::PointXYZ>::Ptr p2_cloud;
			p2_cloud = m_node_list_A[j]->m_cloud_;
			double sum_dis = 0;
			for (auto point : (*p2_cloud))
			{
				auto dis = pcl::pointToPlaneDistance(point, A, B, C, D);
				sum_dis += dis;
			}
			double avg_dis = sum_dis / static_cast<int>(p2_cloud->size());
			if (avg_dis < 0.05)//超参数点到平面平均距离阈值
			{
				is_merge[j] = 1;
				temp_plane.node_list.push_back(m_node_list_A[j]);
				(*new_cloud) = (*new_cloud) + (*p2_cloud);
			}
		}
		temp_plane.plane_cloud = new_cloud;
		m_plane_list_.push_back(temp_plane);

	}
	//去除node小于2的plane
	for (auto onenode = m_plane_list_.cbegin(); onenode != m_plane_list_.cend();)
	{
		if ((*onenode).node_list.size() < 2)
		{
			onenode = m_plane_list_.erase(onenode);
		}
		else
		{
			onenode++;
		}
	}
	//测试用,用于保存聚类后的点云
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_after(new pcl::PointCloud<pcl::PointXYZRGB>);
	//for (auto onenode : m_plane_list_)
	//{
	//	int random_color_r, random_color_g, random_color_b;
	//	random_color_r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
	//	random_color_g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
	//	random_color_b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
	//	pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	//	temp_cloud = onenode.plane_cloud;
	//	for (auto point : temp_cloud->points)
	//	{
	//		pcl::PointXYZRGB tempa;
	//		tempa.x = point.x;
	//		tempa.y = point.y;
	//		tempa.z = point.z;
	//		tempa.r = random_color_r;
	//		tempa.g = random_color_g;
	//		tempa.b = random_color_b;
	//		cloud_after->push_back(tempa);
	//	}
	//}
	//cloud_after->width = cloud_after->points.size();
	//cloud_after->height = 1;
	//cloud_after->is_dense = true;
	//stringstream ss;
	//ss << "平面检测后" << ".pcd";
	//pcl::PCDWriter writer;
	//writer.write<pcl::PointXYZRGB>(ss.str(), *cloud_after, false);

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//11月8日，求视角
template<typename PointT> float
thickness::Thickness<PointT>::calNodeAngle(std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>> corr)
{

	Cloud::Ptr cloud(new Cloud);
	for (auto point : corr)
	{
		cloud->push_back(point.first);
		cloud->push_back(point.second);
	}
	Eigen::Vector4f centroid;                    // 质心
	pcl::compute3DCentroid(*cloud, centroid);
	Eigen::Matrix3f covariance_matrix;           // 协方差矩阵
	// 计算归一化协方差矩阵和质心
	pcl::computeMeanAndCovarianceMatrix(*cloud, covariance_matrix, centroid);
	// 计算协方差矩阵的特征值与特征向量
	Eigen::Matrix3f eigenVectors;
	Eigen::Vector3f eigenValues;
	pcl::eigen33(covariance_matrix, eigenVectors, eigenValues);
	// 查找最小特征值的位置
	Eigen::Vector3f::Index minRow, minCol;
	eigenValues.minCoeff(&minRow, &minCol);
	// 获取平面方程：AX+BY+CZ+D = 0的系数
	Eigen::Vector3f normal = eigenVectors.col(minCol);
	double D = -normal.dot(centroid.head<3>());
	Eigen::Vector3f vector_with_origin;
	vector_with_origin[0] = 0 - centroid(0);
	vector_with_origin[1] = 0 - centroid(1);
	vector_with_origin[2] = 0 - centroid(2);
	float ang = pcl::getAngle3D(normal, vector_with_origin, true);
	if (ang > 90)
	{
		normal[0] = -normal[0];
		normal[1] = -normal[1];
		normal[2] = -normal[2];
		ang = pcl::getAngle3D(normal, vector_with_origin, true);
	}
	return abs(ang);

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::parallel_detection()
{
	for (int i = 0; i < static_cast<int>(m_plane_list_.size()); i++)
	{
		vector<int> is_merge(static_cast<int>(m_plane_list_[i].node_list.size()));
		auto node_list = m_plane_list_[i].node_list;
		for (int j = 0; j < static_cast<int>(m_plane_list_[i].node_list.size()); j++)
		{
			Parallel temp_parallel;
			if (is_merge[j] != 0)
			{
				continue;
			}
			temp_parallel.node_list.push_back(node_list[j]);
			is_merge[j] = 1;
			pcl::PointCloud<pcl::PointXYZ>::Ptr a_cloud;
			pcl::PointCloud<pcl::PointXYZ>::Ptr new_cloud(new pcl::PointCloud<pcl::PointXYZ>);;
			a_cloud = node_list[j]->m_cloud_;
			for (auto p : (*a_cloud))
			{
				new_cloud->push_back(p);
			}
			auto dir_a = node_list[j]->m_pca_dir;
			float l_a = node_list[j]->m_length_;
			for (int k = 0; k < static_cast<int>(m_plane_list_[i].node_list.size()); k++)
			{
				if (j == k || is_merge[k] != 0)
				{
					continue;
				}
				pcl::PointCloud<pcl::PointXYZ>::Ptr b_cloud;
				b_cloud = node_list[k]->m_cloud_;
				auto dir_b = node_list[k]->m_pca_dir;
				float l_b = node_list[k]->m_length_;
				float ang = getAngle3D(dir_a, dir_b, true);
				if (ang > 90)
				{
					ang = 180 - ang;
				}
				float min_l;
				if (l_a < l_b)
				{
					min_l = l_a;
				}
				else
				{
					min_l = l_b;
				}
				float alpha = abs(l_a - l_b) / min_l;

				if (ang > 13 || alpha > 5)//参数包括角度阈值参数和长度比值阈值参数
				{
					continue;
				}
				else
				{
					is_merge[k] = 1;
					temp_parallel.node_list.push_back(node_list[k]);
					(*new_cloud) = (*new_cloud) + (*b_cloud);
				}

			}
			temp_parallel.parallel_cloud = new_cloud;
			m_parallel_list_.push_back(temp_parallel);
		}
	}
	//去除node小于2的平行集合
	for (auto onenode = m_parallel_list_.cbegin(); onenode != m_parallel_list_.cend();)
	{
		if ((*onenode).node_list.size() < 2)
		{
			onenode = m_parallel_list_.erase(onenode);
		}
		else
		{
			onenode++;
		}
	}
	//测试用,用于保存聚类后的点云
	/*pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_after(new pcl::PointCloud<pcl::PointXYZRGB>);
	for (auto onenode : m_parallel_list_)
	{
		int Random_color_r, Random_color_g, Random_color_b;
		Random_color_r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		temp_cloud = onenode.parallel_cloud;
		for (auto point : temp_cloud->points)
		{
			pcl::PointXYZRGB tempa;
			tempa.x = point.x;
			tempa.y = point.y;
			tempa.z = point.z;
			tempa.r = Random_color_r;
			tempa.g = Random_color_g;
			tempa.b = Random_color_b;
			cloud_after->push_back(tempa);
		}
	}
	cloud_after->width = cloud_after->points.size();
	cloud_after->height = 1;
	cloud_after->is_dense = true;
	stringstream ss;
	ss << "平行检测后" << ".pcd";
	pcl::PCDWriter writer;
	writer.write<pcl::PointXYZRGB>(ss.str(), *cloud_after, false);*/

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PoinT> float
thickness::Thickness<PoinT>::BinaryClassification(std::vector<float>& std_result)
{
	sort(std_result.begin(), std_result.end());
	float interval;
	interval = (std_result[std_result.size() - 1] - std_result[0]) / float(30.0);
	std::vector<std::pair<float, int>> std_result_num;
	int num = 0;
	int k = 0;
	for (int i = 0; i < std_result.size(); i++)
	{
		if (std_result[i] >= std_result[0] + interval * k && std_result[i] < std_result[0] + interval * (k + 1))
		{
			num++;
		}
		if (std_result[i] >= std_result[0] + interval * (k + 1))
		{
			std_result_num.push_back(make_pair(std_result[i], num));
			num = 0;
			k = k + 1;
			if (std_result[i] >= std_result[0] + interval * (k + 1))
			{
				k = floor((std_result[i] - std_result[0]) / interval);
			}
			i = i - 1;
		}
	}
	std::vector<std::pair<float, int>> std_result_difference;
	for (int i = 1; i < std_result_num.size(); i++)
	{
		std_result_difference.push_back(make_pair(std_result_num[i].first, (std_result_num[i - 1].second - std_result_num[i].second)));
	}
	vector<string> line_kmeans;
	for (int i = 0; i < std_result_difference.size(); i++)
	{
		string line_t;
		line_t = to_string(std_result_difference[i].first);
		line_t.append(" ");
		line_t.append(to_string(std_result_difference[i].second));
		line_kmeans.push_back(line_t);
	}
	int pointId = 1;
	vector<Point> all_points;
	for (int i = 0; i < line_kmeans.size(); i++)
	{
		Point point(pointId, line_kmeans[i]);
		all_points.push_back(point);
		pointId++;
	}
	int K = 2;
	if ((int)all_points.size() <= K)
	{
		return 1;
	}
	int iters = 100;
	KMeans kmeans(K, iters);
	double threshold = kmeans.run(all_points);
	return threshold;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::thicknessCal(std::vector<ThicknessPair>& ThicknessPair_list, std::vector<ThicknessPair>& ThicknessPair_list_std, std::vector<std::pair<float, int>>& thickresult, std::vector<float>& std_result)
{
	for (int thickness_index = 0; thickness_index < static_cast<int>(ThicknessPair_list.size()); thickness_index++)
	{
		int num = ThicknessPair_list[thickness_index].point_num;
		float thickness_res_sum = 0;
		ThicknessPair one_thickness = ThicknessPair_list[thickness_index];
		for (int index = 0; index < static_cast<int>(one_thickness.corr.size()); index++)
		{
			pcl::PointXYZ p1, p2;
			p1 = one_thickness.corr[index].first;
			p2 = one_thickness.corr[index].second;
			float cal_res_one = pcl::euclideanDistance(p1, p2);
			thickness_res_sum += cal_res_one;
		}
		float avg_thickness_res = thickness_res_sum / static_cast<float>(num);
		float thickness_res_sum_std = 0;
		for (int index = 0; index < static_cast<int>(one_thickness.corr.size()); index++)
		{
			pcl::PointXYZ p1, p2;
			p1 = one_thickness.corr[index].first;
			p2 = one_thickness.corr[index].second;
			float cal_res_one = pcl::euclideanDistance(p1, p2);
			thickness_res_sum_std += (cal_res_one - avg_thickness_res) * (cal_res_one - avg_thickness_res);
		}
		float avg_thickness_res_std = thickness_res_sum_std / static_cast<float>(num);
		std_result.push_back(avg_thickness_res_std);
	}
	double threshold;
	if (std_result.size() <= 1)
	{
		threshold = 2.0;
	}
	if (std_result.size() > 1)
	{
		threshold = BinaryClassification(std_result);
	}
	if (threshold == 1.0)
	{
		sort(std_result.begin(), std_result.end());
		threshold = std_result[floor(std_result.size() / 2.0)];
	}
	for (int thickness_index = 0; thickness_index < static_cast<int>(ThicknessPair_list.size()); thickness_index++)
	{
		int num = ThicknessPair_list[thickness_index].point_num;
		float thickness_res_sum = 0;
		ThicknessPair one_thickness = ThicknessPair_list[thickness_index];
		for (int index = 0; index < static_cast<int>(one_thickness.corr.size()); index++)
		{
			pcl::PointXYZ p1, p2;
			p1 = one_thickness.corr[index].first;
			p2 = one_thickness.corr[index].second;
			float cal_res_one = pcl::euclideanDistance(p1, p2);
			thickness_res_sum += cal_res_one;
		}
		float avg_thickness_res = thickness_res_sum / static_cast<float>(num);
		float thickness_res_sum_std = 0;
		for (int index = 0; index < static_cast<int>(one_thickness.corr.size()); index++)
		{
			pcl::PointXYZ p1, p2;
			p1 = one_thickness.corr[index].first;
			p2 = one_thickness.corr[index].second;
			float cal_res_one = pcl::euclideanDistance(p1, p2);
			thickness_res_sum_std += (cal_res_one - avg_thickness_res) * (cal_res_one - avg_thickness_res);
		}
		float avg_thickness_res_std = thickness_res_sum_std / static_cast<float>(num);
		if (avg_thickness_res_std < threshold)
		{
			ThicknessPair_list_std.push_back(one_thickness);
			thickresult.push_back(make_pair(avg_thickness_res, num));
		}
	}
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::density_detection(float a, float b)
{
	for (int i = 0; i < static_cast<int>(m_parallel_list_.size()); i++)
	{
		auto node_list = m_parallel_list_[i].node_list;
		for (int j = 0; j < static_cast<int>(m_parallel_list_[i].node_list.size()); j++)
		{
			ThicknessPair temp_thicknesspair;
			int flag = 0;//0为j,k,1为k.j
			for (int k = 0; k < static_cast<int>(m_parallel_list_[i].node_list.size()); k++)
			{
				if (j <= k)
				{
					continue;
				}
				std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>> corr_AB;
				std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>> corr_BA;
				std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>> corr;

				corr_AB = cal_node_corr(node_list[j], node_list[k]);
				corr_BA = cal_node_corr(node_list[k], node_list[j]);

				if (corr_AB.size() == 0 || corr_BA.size() == 0)
				{
					continue;
				}
				if ((double)corr_AB.size() / (double)corr_BA.size() < 0.4 || (double)corr_BA.size() / (double)corr_AB.size() < 0.4)
				{
					continue;
				}
				if (corr_AB.size() >= corr_BA.size())
				{
					corr = corr_AB;
					flag = 0;
					if ((double)corr.size() / (double)node_list[j]->m_cloud_->size() < 0.4)
					{
						continue;
					}
				}
				else
				{
					corr = corr_BA;
					flag = 1;
					if ((double)corr.size() / (double)node_list[k]->m_cloud_->size() < 0.4)
					{
						continue;
					}
				}
				m_search_->setInputCloud(m_original_cloud_);
				std::vector<int> pointIdxRadiusSearch;
				std::vector<float> pointRadiusSquaredDistance;
				pcl::PointXYZ midPoint;
				int number_of_pairs = corr.size();
				int total_have_neiborghtor = 0;
				float sum_thickness = 0;
				float avg_thickness;
				for (auto onecorr : corr)
				{
					pcl::PointXYZ p1, p2;
					p1 = onecorr.first;
					p2 = onecorr.second;
					midPoint.x = (double)(p1.x + p2.x) / 2.0;
					midPoint.y = (double)(p1.y + p2.y) / 2.0;
					midPoint.z = (double)(p1.z + p2.z) / 2.0;
					double distance = pcl::euclideanDistance(p1, p2);
					sum_thickness += distance;
					pointIdxRadiusSearch.clear();
					pointRadiusSquaredDistance.clear();
					m_search_->radiusSearch(midPoint, distance * 0.5 * 0.5, pointIdxRadiusSearch, pointRadiusSquaredDistance);
					int size = pointIdxRadiusSearch.size();
					if (size > 0)
					{
						total_have_neiborghtor++;
						/*pcl::PointCloud<pcl::PointXYZ>::Ptr tt(new pcl::PointCloud<pcl::PointXYZ>);
						for (auto id : pointIdxRadiusSearch)
						{
							tt->push_back(m_original_cloud_->points[id]);
						}
						record++;
						pcl::io::savePCDFileASCII(std::to_string(record)+"tt.pcd", *tt);*/

					}
				}
				avg_thickness = sum_thickness / (double)number_of_pairs;
				double radio = (double)total_have_neiborghtor / (double)number_of_pairs;
				//cout <<avg_thickness<< endl;
				if (radio > 0.7 && avg_thickness < 0.02 && number_of_pairs > 5)//参数
				{
					temp_thicknesspair.thickness_pair.first = node_list[j];
					temp_thicknesspair.thickness_pair.second = node_list[k];
					if (flag == 0)
					{
						temp_thicknesspair.type.first = node_list[j]->m_type_;
						temp_thicknesspair.type.second = node_list[k]->m_type_;


					}
					else
					{
						temp_thicknesspair.type.first = node_list[k]->m_type_;
						temp_thicknesspair.type.second = node_list[j]->m_type_;
					}

					temp_thicknesspair.corr = corr;
					temp_thicknesspair.point_num = number_of_pairs;//点对的个数
					temp_thicknesspair.thickness_val = avg_thickness;
					pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_one(new pcl::PointCloud<pcl::PointXYZ>);//一个厚度
					for (auto point : corr)
					{
						pcl::PointXYZ tempa, tempb;
						tempa.x = point.first.x;
						tempa.y = point.first.y;
						tempa.z = point.first.z;
						tempb.x = point.second.x;
						tempb.y = point.second.y;
						tempb.z = point.second.z;
						cloud_one->push_back(tempa);
						cloud_one->push_back(tempb);

					}
					pcl::PointXYZ min_point_AABB;
					pcl::PointXYZ max_point_AABB;
					pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
					feature_extractor.setInputCloud(cloud_one);
					feature_extractor.compute();
					feature_extractor.getAABB(min_point_AABB, max_point_AABB);
					temp_thicknesspair.min_x = min_point_AABB.x;
					temp_thicknesspair.min_y = min_point_AABB.y;
					temp_thicknesspair.max_x = max_point_AABB.x;
					temp_thicknesspair.max_y = max_point_AABB.y;
					m_ThicknessPair_list_.push_back(temp_thicknesspair);
				}
			}

		}

	}


	///////删除质心z坐标大于一定阈值的node
	for (auto onenode = m_ThicknessPair_list_.begin(); onenode != m_ThicknessPair_list_.end();)
	{
		auto pairs = onenode->corr;
		Cloud::Ptr cloud(new Cloud);
		for (auto point : pairs)
		{
			cloud->push_back(point.first);
			cloud->push_back(point.second);
		}

		Eigen::Vector4f centroid;                    // 质心
		pcl::compute3DCentroid(*cloud, centroid);


		//cout << "" << centroid(2) << endl;
		if (centroid(2) > a)
		{
			onenode = m_ThicknessPair_list_.erase(onenode);
		}
		else
		{
			onenode++;
		}
	}
	//删除视角角度大于一定阈值的node
	vector<float> ang_results;
	for (auto jonenode = m_ThicknessPair_list_.begin(); jonenode != m_ThicknessPair_list_.end();)
	{
		auto pairs = jonenode->corr;
		float ang_res = calNodeAngle(pairs);
		ang_results.push_back(ang_res);
		if (ang_res > b)
		{
			jonenode = m_ThicknessPair_list_.erase(jonenode);
		}
		else
		{
			jonenode++;
		}
	}

	//写入特征
	/*ofstream OutFile("Test.txt");
	OutFile << "This is a Test12!";
	OutFile.close();*/


	//删掉长度小的厚度（厚度点个数的逻辑）
	//vector<int> arr;
	//for (auto onenode = m_ThicknessPair_list_.begin(); onenode != m_ThicknessPair_list_.end();)
	//{
	//	int a = onenode->point_num;
	//	arr.push_back(a);
	//	onenode++;
	//}
	//int asize = arr.size();
	////cout << "" << asize << endl;
	//sort(arr.begin(), arr.begin() + asize);
	//cout <<"asize=" << asize << endl;
	///*for (int i = 0; i < arr.size(); i++)
	//{
	//	cout << arr[i] << endl;
	//}*/
	//int decount = 0.2* asize;
	//cout << "decount=" << decount << endl;
	//
	//for (auto onenode = m_ThicknessPair_list_.begin(); onenode != m_ThicknessPair_list_.end();)
	//{
	//	int flag = 1;
	//	int a = onenode->point_num;
	//	for (int i = 0; i < decount; i++)
	//	{
	//		if (a == arr[i] && a < 20)
	//		{
	//			onenode = m_ThicknessPair_list_.erase(onenode);
	//			flag = 0;
	//		}
	//	}
	//	if (flag == 1)
	//	{
	//		onenode++;

	//	}
	//}



	//删除共享
	vector<int>share_thick_point_num;
	for (auto onenode = m_ThicknessPair_list_.begin(); onenode != m_ThicknessPair_list_.end();)
	{
		vector<float>otherxx;
		vector<float>otheryy;
		vector<float>otherzz;
		for (auto tonenode = m_ThicknessPair_list_.begin(); tonenode != m_ThicknessPair_list_.end();)
		{
			if (tonenode != onenode)
			{
				auto pairs = tonenode->corr;
				int flag = 0;
				for (auto point : pairs)
				{
					if (flag == 5)
					{
						continue;
					}
					float x = point.first.x;
					float x2 = point.second.x;
					float y = point.first.y;
					float y2 = point.second.y;
					float z = point.first.z;
					float z2 = point.second.z;
					otherxx.push_back(x);
					otherxx.push_back(x2);
					otheryy.push_back(y);
					otheryy.push_back(y2);
					otherzz.push_back(z);
					otherzz.push_back(z2);
					flag++;
				}
			}
			tonenode++;

		}
		auto pairs = onenode->corr;
		int a = onenode->point_num;
		int flaggg = 0;
		for (auto point : pairs)
		{
			for (int i = 0; i < otherxx.size(); i++)
			{
				if (flaggg == 1)
				{
					continue;
				}
				if (point.first.x == otherxx[i] || point.second.x == otherxx[i])
				{
					if (point.first.y == otheryy[i] || point.second.y == otheryy[i])
					{
						if (point.first.z == otherzz[i] || point.second.z == otherzz[i])
						{

							share_thick_point_num.push_back(a);
							flaggg = 1;
						}
					}
				}

			}
		}
		onenode++;
		vector<float>(otherxx).swap(otherxx);
		vector<float>(otheryy).swap(otheryy);
		vector<float>(otherzz).swap(otherzz);
	}
	int share_size = share_thick_point_num.size();
	sort(share_thick_point_num.begin(), share_thick_point_num.begin() + share_size);
	if (share_thick_point_num.size() == 0)
	{
		int a = 0;
		share_thick_point_num.push_back(a);
	}

	for (auto onenode = m_ThicknessPair_list_.begin(); onenode != m_ThicknessPair_list_.end();)
	{
		int a = onenode->point_num;
		for (int i = 0; i < share_thick_point_num.size(); i++)
		{
			if (a == share_thick_point_num[i])
			{
				onenode = m_ThicknessPair_list_.erase(onenode);
			}
			else
			{
				onenode++;
			}
		}
	}




	std::vector<ThicknessPair> ThicknessPair_list_temp;
	for (auto temp : m_ThicknessPair_list_)
	{
		if (temp.thickness_pair.first->m_type_ != temp.thickness_pair.second->m_type_)
		{
			ThicknessPair_list_temp.push_back(temp);
		}
		else
		{
			ThicknessPair_list_inclass.push_back(temp);
		}
	}
	if (ThicknessPair_list_temp.size() > 0)
	{
		std::vector<float> std_result;
		std::vector<std::pair<float, int>> thickresult;
		thicknessCal(ThicknessPair_list_temp, ThicknessPair_list_difclass, thickresult, std_result);//方差滤除
	}
	m_ThicknessPair_list_.clear();
	for (auto temp1 : ThicknessPair_list_difclass)
	{
		m_ThicknessPair_list_.push_back(temp1);

	}
	for (auto temp1 : ThicknessPair_list_inclass)
	{
		m_ThicknessPair_list_.push_back(temp1);
	}


	//数据集特征
	//int record4 = 0;
	//for (auto onenode = m_ThicknessPair_list_.begin(); onenode != m_ThicknessPair_list_.end();)
	//{
	//	vector<double> son_node_density;
	//	Node* temp_node1 = new Node();
	//	Node* temp_node2 = new Node();
	//	temp_node1 = onenode->thickness_pair.first;
	//	temp_node2 = onenode->thickness_pair.second;
	//	son_node_length.clear();
	//	son_node_roughness.clear();
	//	son_node_density.clear();
	//	son_node_mindis.clear();
	//	son_node_fitlinedis.clear();
	//	find_son_node(temp_node1);
	//	find_son_node(temp_node2);
	//	record4++;
	//	fstream f;
	//	f.open(std::to_string(record4) + "数据集特征.txt", ios::out);
	//	f << record4 << endl;
	//	//均值、方差
	//	double son_node_length_sum = std::accumulate(std::begin(son_node_length), std::end(son_node_length), 0.0);
	//	double son_node_length_mean = son_node_length_sum / son_node_length.size();
	//	double son_node_length_variance = 0.0;
	//	for (int i = 0; i < son_node_length.size(); i++)
	//	{
	//		son_node_length_variance = son_node_length_variance + pow(son_node_length[i] - son_node_length_mean, 2);
	//	}
	//	son_node_length_variance = son_node_length_variance / son_node_length.size();
	//	f << son_node_length_mean << endl << son_node_length_variance << endl;

	//	double son_node_roughness_sum = std::accumulate(std::begin(son_node_roughness), std::end(son_node_roughness), 0.0);
	//	double son_node_roughness_mean = son_node_roughness_sum / son_node_roughness.size();
	//	double son_node_roughness_variance = 0.0;
	//	for (int i = 0; i < son_node_roughness.size(); i++)
	//	{
	//		son_node_roughness_variance = son_node_roughness_variance + pow(son_node_roughness[i] - son_node_roughness_mean, 2);
	//	}
	//	son_node_roughness_variance = son_node_roughness_variance / son_node_roughness.size();
	//	f << son_node_roughness_mean << endl << son_node_roughness_variance << endl;

	//	m_search_->setInputCloud(m_original_cloud_);
	//	std::vector<int> pointIdxRadiusSearch;
	//	std::vector<float> pointRadiusSquaredDistance;
	//	pcl::PointXYZ midPoint;
	//	auto corrs = onenode->corr;
	//	for (auto onecorr : corrs)
	//	{
	//		vector<int> nearpoint;
	//		pcl::PointXYZ p1, p2;
	//		p1 = onecorr.first;
	//		p2 = onecorr.second;
	//		midPoint.x = (double)(p1.x + p2.x) / 2.0;
	//		midPoint.y = (double)(p1.y + p2.y) / 2.0;
	//		midPoint.z = (double)(p1.z + p2.z) / 2.0;
	//		double distance = pcl::euclideanDistance(p1, p2);
	//		pointIdxRadiusSearch.clear();
	//		pointRadiusSquaredDistance.clear();
	//		m_search_->radiusSearch(midPoint, 0.4 * distance, pointIdxRadiusSearch, pointRadiusSquaredDistance);
	//		//得到近邻点ID
	//		for (auto id : pointIdxRadiusSearch)
	//		{
	//			nearpoint.push_back(id);
	//		}
	//		pcl::PointCloud<pcl::PointXYZ>::Ptr tt(new pcl::PointCloud<pcl::PointXYZ>);
	//		for (auto id : nearpoint)
	//		{
	//			tt->push_back(m_original_cloud_->points[id]);
	//		}
	//		son_node_density.push_back(tt->size());

	//	}
	//	double son_node_density_sum = std::accumulate(std::begin(son_node_density), std::end(son_node_density), 0.0);
	//	double son_node_density_mean = son_node_density_sum / son_node_density.size();
	//	double son_node_density_variance = 0.0;
	//	for (int i = 0; i < son_node_density.size(); i++)
	//	{
	//		son_node_density_variance = son_node_density_variance + pow(son_node_density[i] - son_node_density_mean, 2);
	//	}
	//	son_node_density_variance = son_node_density_variance / son_node_roughness.size();
	//	f << son_node_density_mean << endl << son_node_density_variance << endl;

	//	double son_node_mindis_sum = std::accumulate(std::begin(son_node_mindis), std::end(son_node_mindis), 0.0);
	//	double son_node_mindis_mean = son_node_mindis_sum / son_node_mindis.size();
	//	double son_node_mindis_variance = 0.0;
	//	for (int i = 0; i < son_node_mindis.size(); i++)
	//	{
	//		son_node_mindis_variance = son_node_mindis_variance + pow(son_node_mindis[i] - son_node_mindis_mean, 2);
	//	}
	//	son_node_mindis_variance = son_node_mindis_variance / son_node_mindis.size();
	//	f << son_node_mindis_mean << endl << son_node_mindis_variance << endl;

	//	double son_node_fitlinedis_sum = std::accumulate(std::begin(son_node_fitlinedis), std::end(son_node_fitlinedis), 0.0);
	//	double son_node_fitlinedis_mean = son_node_fitlinedis_sum / son_node_fitlinedis.size();
	//	double son_node_fitlinedis_variance = 0.0;
	//	for (int i = 0; i < son_node_fitlinedis.size(); i++)
	//	{
	//		son_node_fitlinedis_variance = son_node_fitlinedis_variance + pow(son_node_fitlinedis[i] - son_node_fitlinedis_mean, 2);
	//	}
	//	son_node_fitlinedis_variance = son_node_fitlinedis_variance / son_node_fitlinedis.size();
	//	f << son_node_fitlinedis_mean << endl << son_node_fitlinedis_variance << endl;

	//
	//	
	//
	//	/*cout << "平面方程为：\n"
	//		<< coefficient[0] << "x + "
	//		<< coefficient[1] << "y + "
	//		<< coefficient[2] << "z + "
	//		<< coefficient[3] << " = 0"
	//		<< endl;*/
	//	//算厚度点到拟合平面的距离以及方差
	//	
	//	

	//	//最终特征
	//	//厚度值方差
	//	vector<double> thickness_value;
	//	for (auto point : corrs)
	//	{
	//		pcl::PointXYZ p1, p2;
	//		p1 = point.first;
	//		p2 = point.second;
	//		double distance = pcl::euclideanDistance(p1, p2);
	//		thickness_value.push_back(distance);
	//	}
	//	double thickness_value_sum = std::accumulate(std::begin(thickness_value), std::end(thickness_value), 0.0);
	//	double thickness_value_mean = thickness_value_sum / thickness_value.size();
	//	double thickness_value_variance = 0.0;
	//	for (int i = 0; i < thickness_value.size(); i++)
	//	{
	//		thickness_value_variance = thickness_value_variance + pow(thickness_value[i] - thickness_value_mean, 2);
	//	}
	//	thickness_value_variance = thickness_value_variance / thickness_value.size();


	//	//质心
	//	Cloud::Ptr cloud(new Cloud);
	//	for (auto point : corrs)
	//	{
	//		cloud->push_back(point.first);
	//		cloud->push_back(point.second);
	//	}
	//	Eigen::Vector4f centroid;
	//	pcl::compute3DCentroid(*cloud, centroid);
	//	double cent_z = centroid(2);//质心z坐标
	//	f << thickness_value_variance << endl << pointtoplanedis_variance << endl << pointtoplanedis_mean << endl << cent_z << endl;
	//	f.close();
	//	onenode++;
	//	
	//}

	//新点密度、厚度值方差、点到拟合面
	//int record6 = 0;
	//for (auto onenode = m_ThicknessPair_list_.begin(); onenode != m_ThicknessPair_list_.end();)
	//{
	//	record6++;
	//	fstream f;
	//	f.open(std::to_string(record6) + "点密度、厚度方差、点到拟合面.txt", ios::out);
	//	f << record6<< endl;
	//	vector<double> son_node_density;
	//	son_node_density.clear();
	//	m_search_->setInputCloud(m_original_cloud_);
	//	std::vector<int> pointIdxRadiusSearch;
	//	std::vector<float> pointRadiusSquaredDistance;
	//	pcl::PointXYZ midPoint;
	//	auto corrs = onenode->corr;
	//	vector<double> pointtoplanedis;
	//	pointtoplanedis.clear();
	//	for (auto onecorr : corrs)
	//	{
	//		vector<int> nearpoint;
	//		pcl::PointXYZ p1, p2;
	//		p1 = onecorr.first;
	//		p2 = onecorr.second;
	//		midPoint.x = (double)(p1.x + p2.x) / 2.0;
	//		midPoint.y = (double)(p1.y + p2.y) / 2.0;
	//		midPoint.z = (double)(p1.z + p2.z) / 2.0;
	//		double distance = pcl::euclideanDistance(p1, p2);
	//		pointIdxRadiusSearch.clear();
	//		pointRadiusSquaredDistance.clear();
	//		m_search_->radiusSearch(midPoint, 0.5 * distance, pointIdxRadiusSearch, pointRadiusSquaredDistance);
	//		//得到近邻点ID
	//		for (auto id : pointIdxRadiusSearch)
	//		{
	//			nearpoint.push_back(id);
	//		}
	//		pcl::PointCloud<pcl::PointXYZ>::Ptr tt(new pcl::PointCloud<pcl::PointXYZ>);
	//		for (auto id : nearpoint)
	//		{
	//			tt->push_back(m_original_cloud_->points[id]);
	//		}
	//		son_node_density.push_back(tt->size());
	//		if (tt->size() > 4)
	//		{
	//			pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_plane(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(tt));
	//			pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_plane);
	//			ransac.setDistanceThreshold(0.01);
	//			ransac.computeModel();
	//			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);//平面点云
	//			vector<int> inliers;
	//			ransac.getInliers(inliers);
	//			pcl::copyPointCloud<pcl::PointXYZ>(*tt, inliers, *cloud_plane);
	//			Eigen::VectorXf coefficient;
	//			ransac.getModelCoefficients(coefficient);
	//			double A = coefficient[0];
	//			double B = coefficient[1];
	//			double C = coefficient[2];
	//			double D = coefficient[3];
	//			auto dis1 = pcl::pointToPlaneDistance(p1, A, B, C, D);
	//			auto dis2 = pcl::pointToPlaneDistance(p2, A, B, C, D);
	//			pointtoplanedis.push_back(dis1);
	//			pointtoplanedis.push_back(dis2);
	//		}

	//	}
	//	double pointtoplanedis_sum = std::accumulate(std::begin(pointtoplanedis), std::end(pointtoplanedis), 0.0);
	//	double pointtoplanedis_mean = pointtoplanedis_sum / pointtoplanedis.size();
	//	double pointtoplanedis_variance = 0.0;
	//	for (int i = 0; i < pointtoplanedis.size(); i++)
	//	{
	//		pointtoplanedis_variance = pointtoplanedis_variance + pow(pointtoplanedis[i] - pointtoplanedis_mean, 2);
	//	}
	//	pointtoplanedis_variance = pointtoplanedis_variance / pointtoplanedis.size();
	//	f << pointtoplanedis_variance << endl << pointtoplanedis_mean << endl;

	//	double son_node_density_sum = std::accumulate(std::begin(son_node_density), std::end(son_node_density), 0.0);
	//	double son_node_density_mean = son_node_density_sum / son_node_density.size();
	//	double son_node_density_variance = 0.0;
	//	for (int i = 0; i < son_node_density.size(); i++)
	//	{
	//		son_node_density_variance = son_node_density_variance + pow(son_node_density[i] - son_node_density_mean, 2);
	//	}
	//	son_node_density_variance = son_node_density_variance / son_node_density.size();
	//	f << son_node_density_mean << endl << son_node_density_variance << endl;

	//	//厚度值方差
	//	vector<double> thickness_value;
	//	for (auto point : corrs)
	//	{
	//		pcl::PointXYZ p1, p2;
	//		p1 = point.first;
	//		p2 = point.second;
	//		double distance = pcl::euclideanDistance(p1, p2);
	//		thickness_value.push_back(distance);
	//	}
	//	double thickness_value_sum = std::accumulate(std::begin(thickness_value), std::end(thickness_value), 0.0);
	//	double thickness_value_mean = thickness_value_sum / thickness_value.size();
	//	double thickness_value_variance = 0.0;
	//	for (int i = 0; i < thickness_value.size(); i++)
	//	{
	//		thickness_value_variance = thickness_value_variance + pow(thickness_value[i] - thickness_value_mean, 2);
	//	}
	//	thickness_value_variance = thickness_value_variance / thickness_value.size();
	//	f << thickness_value_variance << endl;
	//	f.close();
	//	onenode++;

	//}





	//过程特征
	//int record3 = 0;
	//for (auto onenode = m_ThicknessPair_list_.begin(); onenode != m_ThicknessPair_list_.end();)
	//{
	//	vector<double> son_node_density;
	//	Node* temp_node1 = new Node();
	//	Node* temp_node2 = new Node();
	//	temp_node1 = onenode->thickness_pair.first;
	//	temp_node2 = onenode->thickness_pair.second;
	//	son_node_length.clear();
	//	son_node_roughness.clear();
	//	son_node_density.clear();
	//	son_node_mindis.clear();
	//	son_node_fitlinedis.clear();
	//	find_son_node(temp_node1);
	//	find_son_node(temp_node2);
	//	record3++;
	//	fstream f;
	//	f.open(std::to_string(record3) + "过程特征.txt", ios::out);

	//	for (int i = 0; i < son_node_length.size(); i++)
	//	{
	//		float features = 0;
	//		features = son_node_length[i];
	//		f << "长度 " << features << endl;

	//	}
	//	for (int i = 0; i < son_node_roughness.size(); i++)
	//	{
	//		float features = 0;
	//		features = son_node_roughness[i];
	//		f << "粗度 " << features << endl;
	//	}

	//	m_search_->setInputCloud(m_original_cloud_);
	//	std::vector<int> pointIdxRadiusSearch;
	//	std::vector<float> pointRadiusSquaredDistance;
	//	pcl::PointXYZ midPoint;
	//	auto corrs = onenode->corr;
	//	for (auto onecorr : corrs)
	//	{
	//		vector<int> nearpoint;
	//		pcl::PointXYZ p1, p2;
	//		p1 = onecorr.first;
	//		p2 = onecorr.second;
	//		midPoint.x = (double)(p1.x + p2.x) / 2.0;
	//		midPoint.y = (double)(p1.y + p2.y) / 2.0;
	//		midPoint.z = (double)(p1.z + p2.z) / 2.0;
	//		double distance = pcl::euclideanDistance(p1, p2);
	//		pointIdxRadiusSearch.clear();
	//		pointRadiusSquaredDistance.clear();
	//		m_search_->radiusSearch(midPoint, 0.4 * distance, pointIdxRadiusSearch, pointRadiusSquaredDistance);
	//		//得到近邻点ID
	//		for (auto id : pointIdxRadiusSearch)
	//		{
	//			nearpoint.push_back(id);
	//		}
	//		pcl::PointCloud<pcl::PointXYZ>::Ptr tt(new pcl::PointCloud<pcl::PointXYZ>);
	//		for (auto id : nearpoint)
	//		{
	//			tt->push_back(m_original_cloud_->points[id]);
	//		}
	//		son_node_density.push_back(tt->size());

	//	}
	//	for (int i = 0; i < son_node_density.size(); i++)
	//	{
	//		float features = 0;
	//		features = son_node_density[i];
	//		f << "点密度 " << features << endl;
	//	}
	//	for (int i = 0; i < son_node_mindis.size(); i++)
	//	{
	//		float features = 0;
	//		features = son_node_mindis[i];
	//		f << "子node端点最短距离 " << features << endl;
	//	}
	//	for (int i = 0; i < son_node_fitlinedis.size(); i++)
	//	{
	//		float features = 0;
	//		features = son_node_fitlinedis[i];
	//		f << "子node共线距离 " << features << endl;
	//	}
	//	onenode++;
	//	
	//}



	//最终特征
	//int record = 0;
	//for (auto onenode = m_ThicknessPair_list_.begin(); onenode != m_ThicknessPair_list_.end();)
	//{
	//	//求取一个厚度中,厚度点到拟合面的距离
	//	m_search_->setInputCloud(m_original_cloud_);
	//	std::vector<int> pointIdxRadiusSearch;
	//	std::vector<float> pointRadiusSquaredDistance;
	//	pcl::PointXYZ midPoint;
	//	auto corrs = onenode->corr;
	//	vector<double> pointtoplanedis;
	//	for (auto onecorr : corrs)
	//	{
	//		vector<int> nearpoint;
	//		pcl::PointXYZ p1, p2;
	//		p1 = onecorr.first;
	//		p2 = onecorr.second;
	//		midPoint.x = (double)(p1.x + p2.x) / 2.0;
	//		midPoint.y = (double)(p1.y + p2.y) / 2.0;
	//		midPoint.z = (double)(p1.z + p2.z) / 2.0;
	//		double distance = pcl::euclideanDistance(p1, p2);
	//		pointIdxRadiusSearch.clear();
	//		pointRadiusSquaredDistance.clear();
	//		m_search_->radiusSearch(midPoint, 0.4 * distance, pointIdxRadiusSearch, pointRadiusSquaredDistance);
	//		//得到近邻点ID
	//		for (auto id : pointIdxRadiusSearch)
	//		{
	//			nearpoint.push_back(id);
	//		}
	//		pcl::PointCloud<pcl::PointXYZ>::Ptr tt(new pcl::PointCloud<pcl::PointXYZ>);
	//		for (auto id : nearpoint)
	//		{
	//			tt->push_back(m_original_cloud_->points[id]);
	//		}
	//		//pcl::io::savePCDFileASCII(std::to_string(record) + "tt.pcd", *tt);
	//		//用近邻点拟合平面
	//		pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_plane(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(tt));
	//		pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_plane);
	//		ransac.setDistanceThreshold(0.01);
	//		ransac.computeModel();
	//		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);//平面点云
	//		vector<int> inliers;
	//		ransac.getInliers(inliers);
	//		pcl::copyPointCloud<pcl::PointXYZ>(*tt, inliers, *cloud_plane);
	//		Eigen::VectorXf coefficient;
	//		ransac.getModelCoefficients(coefficient);
	//		double A = coefficient[0];
	//		double B = coefficient[1];
	//		double C = coefficient[2];
	//		double D = coefficient[3];
	//		auto dis1 = pcl::pointToPlaneDistance(p1, A, B, C, D);
	//		auto dis2 = pcl::pointToPlaneDistance(p2, A, B, C, D);
	//		pointtoplanedis.push_back(dis1);
	//		pointtoplanedis.push_back(dis2);

	//	}

	//	//写入特征
	//	record++;
	//	fstream f;
	//	f.open(std::to_string(record) + "最终特征.txt", ios::out);
	//	for (int i = 0; i < pointtoplanedis.size(); i++)
	//	{
	//		f << "点到拟合面的距离为 " << pointtoplanedis[i] << endl;
	//	}
	//	int num_corr = 0;
	//	for (auto onecorr : corrs)
	//	{
	//		num_corr++;
	//		pcl::PointXYZ p1, p2;
	//		p1 = onecorr.first;
	//		p2 = onecorr.second;
	//		double distance = pcl::euclideanDistance(p1, p2);
	//		f << std::to_string(num_corr) + "点对厚度值 " << distance << endl;
	//	}
	//	f.close();
	//	onenode++;
	//}

	//单独厚度
	/*int individual_record = 0;
	for (auto onenode : m_ThicknessPair_list_)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr individual_thickness(new pcl::PointCloud<pcl::PointXYZRGB>);
		int Random_color_r, Random_color_g, Random_color_b;
		Random_color_r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		auto pairs = onenode.corr;
		int a = onenode.point_num;
		for (auto point : pairs)
		{
			pcl::PointXYZRGB tempa, tempb;
			tempa.x = point.first.x;
			tempa.y = point.first.y;
			tempa.z = point.first.z;
			tempb.x = point.second.x;
			tempb.y = point.second.y;
			tempb.z = point.second.z;
			tempa.r = Random_color_r;
			tempa.g = Random_color_g;
			tempa.b = Random_color_b;
			tempb.r = Random_color_r;
			tempb.g = Random_color_g;
			tempb.b = Random_color_b;
			individual_thickness->push_back(tempa);
			individual_thickness->push_back(tempb);
		}
		individual_thickness->width = individual_thickness->points.size();
		individual_thickness->height = 1;
		individual_thickness->is_dense = true;
		individual_record++;
		stringstream ss;
		ss << std::to_string(individual_record)<< ".pcd";
		pcl::PCDWriter writer;
		writer.write<pcl::PointXYZRGB>(ss.str(), *individual_thickness, false);

	}*/

	//测试
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_after(new pcl::PointCloud<pcl::PointXYZRGB>);
	for (auto onenode : m_ThicknessPair_list_)
	{
		int Random_color_r, Random_color_g, Random_color_b;
		Random_color_r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		auto pairs = onenode.corr;
		int a = onenode.point_num;
		for (auto point : pairs)
		{
			pcl::PointXYZRGB tempa, tempb;
			tempa.x = point.first.x;
			tempa.y = point.first.y;
			tempa.z = point.first.z;
			tempb.x = point.second.x;
			tempb.y = point.second.y;
			tempb.z = point.second.z;
			tempa.r = Random_color_r;
			tempa.g = Random_color_g;
			tempa.b = Random_color_b;
			tempb.r = Random_color_r;
			tempb.g = Random_color_g;
			tempb.b = Random_color_b;
			cloud_after->push_back(tempa);
			cloud_after->push_back(tempb);

		}
	}
	cloud_after->width = cloud_after->points.size();
	cloud_after->height = 1;
	cloud_after->is_dense = true;
	stringstream ss;
	ss << "法2所有厚度" << ".pcd";
	pcl::PCDWriter writer;
	writer.write<pcl::PointXYZRGB>(ss.str(), *cloud_after, false);
	
	//可视化
	/*boost::shared_ptr<visualization::PCLVisualizer> viewer(new visualization::PCLVisualizer("3D viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addPointCloud<PointXYZRGB>(cloud_after, "sample cloud");
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
	}*/
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////







template<typename PointT> std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>>
	thickness::Thickness<PointT>::cal_node_corr(Node* A, Node* B)
	{
		std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>> corr_list;
		pcl::PointCloud<pcl::PointXYZ>::Ptr a_cloud;
		a_cloud = A->m_cloud_;
		pcl::PointCloud<pcl::PointXYZ>::Ptr b_cloud;
		b_cloud = B->m_cloud_;
		auto pcadir = B->m_pca_dir;
		auto pcapoint = B->m_pca_point;
		pcl::KdTreeFLANN<pcl::PointXYZ> kd;
		kd.setInputCloud(b_cloud);
		//对每个a中的点求到b拟合直线的投影点。之后判断是否为对应点
		for (auto point : (*a_cloud))
		{
			float t = (pcadir[0] * (point.x - pcapoint[0]) + pcadir[1] * (point.y - pcapoint[1]) + pcadir[2] * (point.z - pcapoint[2])) / (pcadir[0] * pcadir[0] + pcadir[1] * pcadir[1] + pcadir[2] * pcadir[2]);
			pcl::PointXYZ project_p;
			project_p.x = pcadir[0] * t + pcapoint[0];
			project_p.y = pcadir[1] * t + pcapoint[1];
			project_p.z = pcadir[2] * t + pcapoint[2];
			std::vector<int> pointIdxKNNSearch(1);
			std::vector<float> pointKNNSquaredDistance(1);
			if (kd.nearestKSearch(project_p, 1, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
			{
				if (sqrt(pointKNNSquaredDistance[0]) < 0.0015)
				{
					std::pair<pcl::PointXYZ, pcl::PointXYZ> one_corr;
					one_corr.first = point;
					one_corr.second = b_cloud->points[pointIdxKNNSearch[0]];
					corr_list.push_back(one_corr);
				}
			}
		}

		return corr_list;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template<typename PointT> double
		thickness::Thickness<PointT>::fake_test_angle(int num_cluster, pcl::PointCloud<pcl::PointXYZ>::Ptr all_thickness_one, pcl::PointCloud<pcl::PointXYZ>::Ptr all_thickness_two, pcl::PointXYZRGB midPoint, pcl::PointCloud<pcl::PointXYZ>::Ptr curvature_i, pcl::PCA<pcl::PointXYZ> pca, Eigen::Vector2f b, Eigen::RowVector3f V2, float dis, std::vector<int> pointIdxRadiusSearch, std::vector<float> pointRadiusSquaredDistance, std::vector<int> pointIdxRadiusSearch_new, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointXYZ depthPoint)
	{
		pcl::PointXYZ midPoint_XYZ;
		midPoint_XYZ.x = midPoint.x;
		midPoint_XYZ.y = midPoint.y;
		midPoint_XYZ.z = midPoint.z;
		pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
		kdtree.setInputCloud(curvature_i);
		int K_search = 8;
		std::vector<int> pointIdxKNNSearch(K_search);
		std::vector<float> pointKNNSquaredDistance(K_search);
		double x1, x2, y1, y2, z1, z2;//求垂直于主方向直线用的（暂时没有用到）
		pcl::PointXYZ Point2;
		if (kdtree.nearestKSearch(midPoint_XYZ, K_search, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster_pca(new pcl::PointCloud<pcl::PointXYZ>);//八个点的点云为了求主方向
			for (std::size_t i = 0; i < pointIdxKNNSearch.size(); ++i)
			{
				pcl::PointXYZ temp;
				temp.x = (*curvature_i)[pointIdxKNNSearch[i]].x;
				temp.y = (*curvature_i)[pointIdxKNNSearch[i]].y;
				temp.z = (*curvature_i)[pointIdxKNNSearch[i]].z;
				cloud_cluster_pca->push_back(temp);
			}
			cloud_cluster_pca->width = cloud_cluster_pca->points.size();
			cloud_cluster_pca->height = 1;
			cloud_cluster_pca->is_dense = true;
			Eigen::Vector4f pcaCentroid_pca;
			pcl::compute3DCentroid(*cloud_cluster_pca, pcaCentroid_pca);
			Eigen::Matrix3f covariance_pca;
			pcl::computeCovarianceMatrix(*cloud_cluster_pca, pcaCentroid_pca, covariance_pca);
			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver_pca(covariance_pca, Eigen::ComputeEigenvectors);
			Eigen::Matrix3f eigenVectorsPCA_pca = eigen_solver_pca.eigenvectors();
			Eigen::Vector3f eigenValuesPCA_pca = eigen_solver_pca.eigenvalues();
			float t1 = eigenValuesPCA_pca(0);
			int ii = 0;
			if (t1 < eigenValuesPCA_pca(1))
			{
				ii = 1;
				t1 = eigenValuesPCA_pca(1);
			}
			if (t1 < eigenValuesPCA_pca(2))
			{
				ii = 2;
				t1 = eigenValuesPCA_pca(2);
			}
			Eigen::Vector3f v_pca(eigenVectorsPCA_pca(0, ii), eigenVectorsPCA_pca(1, ii), eigenVectorsPCA_pca(2, ii));
			v_pca /= v_pca.norm();
			x2 = midPoint.x + v_pca(0);
			y2 = midPoint.y + v_pca(1);
			z2 = midPoint.z + v_pca(2);
			x1 = midPoint.x;
			y1 = midPoint.y;
			z1 = midPoint.z;
			float x = 0.1;
			float y = 0.1;
			float z = (0 - x * v_pca(0) - y * v_pca(1)) / v_pca(2);
			Point2.x = midPoint.x + x;
			Point2.y = midPoint.y + y;
			Point2.z = midPoint.z + z;
		}
		if (kdtree.nearestKSearch(midPoint_XYZ, K_search, pointIdxKNNSearch, pointKNNSquaredDistance) == 0)//如果没有8个点就用之前求得全局的主方向
		{
			x2 = pca.getMean()[0];
			y2 = pca.getMean()[1];
			z2 = pca.getMean()[2];
			x1 = midPoint.x;
			y1 = midPoint.y;
			z1 = midPoint.z;
			//double z2 = (b[0] - V2[0] * midPoint.x - V2[1] * midPoint.y) / V2[2];
			Eigen::Vector4f pcaCentroid_first;
			pcl::compute3DCentroid(*curvature_i, pcaCentroid_first);
			Eigen::Matrix3f covariance_first;
			pcl::computeCovarianceMatrix(*curvature_i, pcaCentroid_first, covariance_first);
			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver_first(covariance_first, Eigen::ComputeEigenvectors);
			Eigen::Matrix3f eigenVectorsPCA_first = eigen_solver_first.eigenvectors();
			Eigen::Vector3f eigenValuesPCA_first = eigen_solver_first.eigenvalues();
			float t1 = eigenValuesPCA_first(0);
			int ii = 0;
			if (t1 < eigenValuesPCA_first(1))
			{
				ii = 1;
				t1 = eigenValuesPCA_first(1);
			}
			if (t1 < eigenValuesPCA_first(2))
			{
				ii = 2;
				t1 = eigenValuesPCA_first(2);
			}
			Eigen::Vector3f v_first(eigenVectorsPCA_first(0, ii), eigenVectorsPCA_first(1, ii), eigenVectorsPCA_first(2, ii));
			v_first /= v_first.norm();
			float x = 1.0;
			float y = 1.0;
			float z = (0 - x * v_first(0) - y * v_first(1)) / v_first(2);
			Point2.x = midPoint.x + x;
			Point2.y = midPoint.y + y;
			Point2.z = midPoint.z + z;
		}
		double normal01_x = x2 - x1;
		double normal01_y = y2 - y1;
		double normal01_z = z2 - z1;
		for (size_t t = 1; t < pointIdxRadiusSearch.size(); ++t)//求空间点到直线的距离为了求圆柱
		{
			double normal02_x = cloud->points[pointIdxRadiusSearch[t]].x - x1;
			double normal02_y = cloud->points[pointIdxRadiusSearch[t]].y - y1;
			double normal02_z = cloud->points[pointIdxRadiusSearch[t]].z - z1;
			double fenzi = normal01_x * normal02_x + normal01_y * normal02_y + normal01_z * normal02_z;
			double lengthN1 = sqrt(normal01_x * normal01_x + normal01_y * normal01_y + normal01_z * normal01_z);
			double lengthN2 = sqrt(normal02_x * normal02_x + normal02_y * normal02_y + normal02_z * normal02_z);
			double hudu = acos(fenzi / (lengthN1 * lengthN2));
			double ds = abs(lengthN2 * sin(hudu));
			double threshold = dis * 0.7;//如果点到直线的距离大于0.7倍的平均厚度就放到cloud_filtered
			if (ds > threshold)
			{
				pointIdxRadiusSearch_new.push_back(pointIdxRadiusSearch[t]);
			}
		}
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);//去除圆柱之后的点云，用来找平面的
		for (std::size_t i = 0; i < pointIdxRadiusSearch_new.size(); ++i)
		{
			pcl::PointXYZ temp;
			temp.x = cloud->points[pointIdxRadiusSearch_new[i]].x;
			temp.y = cloud->points[pointIdxRadiusSearch_new[i]].y;
			temp.z = cloud->points[pointIdxRadiusSearch_new[i]].z;
			cloud_filtered->push_back(temp);
		}
		cloud_filtered->width = cloud_filtered->points.size();
		cloud_filtered->height = 1;
		cloud_filtered->is_dense = true;
		//cout << cloud_filtered->width << "\n";
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_euclideanCluster(new pcl::search::KdTree<pcl::PointXYZ>);//用cloud_filtered欧式聚类
		tree_euclideanCluster->setInputCloud(cloud_filtered);
		vector<pcl::PointIndices> cluster_indices;
		pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
		ec.setClusterTolerance(dis * 0.2);
		ec.setMinClusterSize(1);
		ec.setMaxClusterSize(cloud_filtered->width);
		ec.setSearchMethod(tree_euclideanCluster);
		ec.setInputCloud(cloud_filtered);
		ec.extract(cluster_indices);
		//cout << cluster_indices.size() << "\n";
		if (cluster_indices.size() == 1)//如果欧式聚类结果只有一个就把当前点去除
		{
			return 10000;
		}
		vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_filtered_all;//把聚类的全部结果放进去
		for (vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster(new pcl::PointCloud<pcl::PointXYZ>);
			for (vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
			{
				cloud_cluster->points.push_back(cloud_filtered->points[*pit]);
			}
			cloud_cluster->width = cloud_cluster->points.size();
			cloud_cluster->height = 1;
			cloud_cluster->is_dense = true;
			cloud_filtered_all.push_back(cloud_cluster);
		}
		int standard = cloud_filtered_all[0]->width;
		int max_num = 0;
		for (int num = 0; num < cloud_filtered_all.size(); num++)
		{
			if (cloud_filtered_all[num]->width > standard)
			{
				standard = cloud_filtered_all[num]->width;
				max_num = num;
			}
		}
		vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_filtered_ex_first;
		for (int num = 0; num < cloud_filtered_all.size(); num++)
		{
			if (num == max_num)
			{
				continue;
			}
			else
			{
				cloud_filtered_ex_first.push_back(cloud_filtered_all[num]);
			}
		}
		int standard_ex_first = cloud_filtered_ex_first[0]->width;
		int second_num = 0;
		for (int num = 0; num < cloud_filtered_ex_first.size(); num++)
		{
			if (cloud_filtered_ex_first[num]->width > standard_ex_first)
			{
				standard_ex_first = cloud_filtered_ex_first[num]->width;
				second_num = num;
			}
		}
		//ransac
		pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_plane(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud_filtered_all[max_num]));

		pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_plane);
		ransac.setDistanceThreshold(0.001);
		ransac.setMaxIterations(500);
		ransac.setProbability(0.99);
		ransac.computeModel();
		vector<int> inliers;
		ransac.getInliers(inliers);

		Eigen::VectorXf coeff;
		ransac.getModelCoefficients(coeff);

		//cout << "平面模型系数coeff(a,b,c,d): " << coeff[0] << " \t" << coeff[1] << "\t " << coeff[2] << "\t " << coeff[3] << endl;
		if (cloud_filtered_ex_first[second_num]->width < 3)//如果第二大点云点个数小于3就返回第一个点到平面距离
		{
			double depthtoplane = 0.0;
			depthtoplane = fabs(coeff[0] * depthPoint.x + coeff[1] * depthPoint.y + coeff[2] * depthPoint.z + coeff[3]) / sqrt(coeff[0] * coeff[0] + coeff[1] * coeff[1] + coeff[2] * coeff[2]);
			return depthtoplane;
		}
		pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_plane_2(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(cloud_filtered_ex_first[second_num]));

		pcl::RandomSampleConsensus<pcl::PointXYZ> ransac_2(model_plane_2);
		ransac_2.setDistanceThreshold(0.001);
		ransac_2.setMaxIterations(500);
		ransac_2.setProbability(0.99);
		ransac_2.computeModel();
		vector<int> inliers_2;
		ransac_2.getInliers(inliers_2);

		Eigen::VectorXf coeff_2;
		ransac_2.getModelCoefficients(coeff_2);

		//cout << "平面模型系数coeff(a,b,c,d): " << coeff_2[0] << " \t" << coeff_2[1] << "\t " << coeff_2[2] << "\t " << coeff_2[3] << endl;
		double depthtoplane = 0.0;
		double dis_one = 0.0;
		dis_one = fabs(coeff[0] * depthPoint.x + coeff[1] * depthPoint.y + coeff[2] * depthPoint.z + coeff[3]) / sqrt(coeff[0] * coeff[0] + coeff[1] * coeff[1] + coeff[2] * coeff[2]);
		double dis_two = 0.0;
		dis_two = fabs(coeff_2[0] * depthPoint.x + coeff_2[1] * depthPoint.y + coeff_2[2] * depthPoint.z + coeff_2[3]) / sqrt(coeff_2[0] * coeff_2[0] + coeff_2[1] * coeff_2[1] + coeff_2[2] * coeff_2[2]);
		if (dis_two >= dis_one)
		{
			depthtoplane = dis_two;
			//cout << "dis_two: " << dis_two << "\n";
		}
		else
		{
			depthtoplane = dis_one;
			//cout << "dis_one: " << dis_one << "\n";
		}


		return depthtoplane;
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template<typename PointT> void
		thickness::Thickness<PointT>::validation_fackAngle(std::vector<ThicknessPair> ThicknessPair_list, std::vector<std::pair<float, int>>& thickresult, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
	{
		thickresult.clear();
		std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_cluster_curvature_all;
		std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_cluster_depth_all;
		std::vector<float> distance_all;
		for (auto thick : ThicknessPair_list)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster_curvature(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster_depth(new pcl::PointCloud<pcl::PointXYZ>);
			float distance_mean = 0.0;
			float distance_sum = 0.0;
			for (auto pair : thick.corr)
			{
				pcl::PointXYZ tempa, tempb;
				tempb.x = pair.second.x;
				tempb.y = pair.second.y;
				tempb.z = pair.second.z;
				cloud_cluster_curvature->push_back(tempb);
				tempa.x = pair.first.x;
				tempa.y = pair.first.y;
				tempa.z = pair.first.z;
				cloud_cluster_depth->push_back(tempa);
				distance_sum += pcl::euclideanDistance(tempa, tempb);
			}
			if (thick.type.first == 0)
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr tempptr(new pcl::PointCloud<pcl::PointXYZ>);
				tempptr = cloud_cluster_depth;
				cloud_cluster_depth = cloud_cluster_curvature;
				cloud_cluster_curvature = tempptr;

			}
			distance_mean = distance_sum / float(thick.point_num);
			distance_all.push_back(distance_mean);
			cloud_cluster_curvature->width = cloud_cluster_curvature->points.size();
			cloud_cluster_curvature->height = 1;
			cloud_cluster_curvature->is_dense = true;
			cloud_cluster_curvature_all.push_back(cloud_cluster_curvature);
			cloud_cluster_depth->width = cloud_cluster_depth->points.size();
			cloud_cluster_depth->height = 1;
			cloud_cluster_depth->is_dense = true;
			cloud_cluster_depth_all.push_back(cloud_cluster_depth);

		}
		pcl::search::OrganizedNeighbor<PointXYZRGB>::Ptr tree(new search::OrganizedNeighbor<PointXYZRGB>());
		tree->setInputCloud(cloud);
		std::vector<int> pointIdxRadiusSearch;//当前点搜索大球厚度2倍
		std::vector<float> pointRadiusSquaredDistance;
		std::vector<int> pointIdxRadiusSearch_new;
		std::vector<double> depthtoplane_all_avg;
		std::vector<double> depthtoplane_all;
		std::vector<std::vector<double>> depthtoplane_all_set;//全部的点到平面距离的
		pcl::PointCloud<pcl::PointXYZ>::Ptr all_thickness_one(new pcl::PointCloud<pcl::PointXYZ>);//中间变量，欧式聚类中点数最多的点
		pcl::PointCloud<pcl::PointXYZ>::Ptr all_thickness_two(new pcl::PointCloud<pcl::PointXYZ>);
		for (int i = 0; i < cloud_cluster_curvature_all.size(); i++)
		{
			pointIdxRadiusSearch.clear();
			pointRadiusSquaredDistance.clear();
			pointIdxRadiusSearch_new.clear();
			depthtoplane_all.clear();
			pcl::PCA<pcl::PointXYZ> pca;
			pca.setInputCloud(cloud_cluster_curvature_all[i]);
			Eigen::RowVector3f V1 = pca.getEigenVectors().col(0);
			Eigen::RowVector3f V2 = pca.getEigenVectors().col(1);
			Eigen::RowVector3f V3 = pca.getEigenVectors().col(2);
			Eigen::Matrix<float, 2, 3>A;
			A.row(0) = V2;
			A.row(1) = V3;
			Eigen::Vector3f sigma = pca.getMean().head<3>();
			Eigen::Vector2f b = A * sigma;
			int j = 0;
			int number = 0;
			int num_cluster = 0;
			while (j < cloud_cluster_curvature_all[i]->points.size())
			{
				pcl::PointXYZRGB midPoint;//其中的每一个点
				int order = j;
				midPoint.x = cloud_cluster_curvature_all[i]->points[order].x;
				midPoint.y = cloud_cluster_curvature_all[i]->points[order].y;
				midPoint.z = cloud_cluster_curvature_all[i]->points[order].z;
				midPoint.r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
				midPoint.g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
				midPoint.b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
				tree->radiusSearch(midPoint, distance_all[i] * 2.0, pointIdxRadiusSearch, pointRadiusSquaredDistance);//两倍平均距离搜索邻域点
				pcl::PointXYZ depthPoint;
				depthPoint.x = cloud_cluster_depth_all[i]->points[order].x;
				depthPoint.y = cloud_cluster_depth_all[i]->points[order].y;
				depthPoint.z = cloud_cluster_depth_all[i]->points[order].z;
				double depthtoplane = 0.0;
				depthtoplane = fake_test_angle(num_cluster, all_thickness_one, all_thickness_two, midPoint, cloud_cluster_curvature_all[i], pca, b, V2, distance_all[i],
					pointIdxRadiusSearch, pointRadiusSquaredDistance, pointIdxRadiusSearch_new, cloud, depthPoint);
				if ((int)depthtoplane == 10000)
				{
					j = j + 1;
					num_cluster = num_cluster + 1;
					continue;
				}
				if (depthtoplane < distance_all[i])//depthtoplane必须大于点到点距离
				{
					j = j + 1;
					num_cluster = num_cluster + 1;
					continue;
				}
				depthtoplane_all.push_back(depthtoplane);
				number = number + 1;
				j = j + 1;
				num_cluster = num_cluster + 1;
			}
			double sum_depthtoplane = 0.0;
			double avg_depthtoplane = 0.0;
			sort(depthtoplane_all.begin(), depthtoplane_all.end());
			int num_count = 0;
			int size_end = depthtoplane_all.size();
			if (depthtoplane_all.size() == 0)
			{
				avg_depthtoplane = 0;
			}
			if (depthtoplane_all.size() != 0)
			{
				for (auto one_dis : depthtoplane_all)
				{
					sum_depthtoplane += one_dis;
				}
				avg_depthtoplane = sum_depthtoplane / static_cast<float>(depthtoplane_all.size());
			}
			depthtoplane_all_avg.push_back(avg_depthtoplane);
		}
		for (int thickness_index = 0; thickness_index < static_cast<int>(ThicknessPair_list.size()); thickness_index++)
		{
			int num = ThicknessPair_list[thickness_index].point_num;
			float thickness_res_sum = 0;
			ThicknessPair one_thick = ThicknessPair_list[thickness_index];
			for (int index = 0; index < static_cast<int>(one_thick.corr.size()); index++)
			{
				pair<pcl::PointXYZ, pcl::PointXYZ> point_pair = one_thick.corr[index];
				float cal_res_one = pcl::euclideanDistance(point_pair.first, point_pair.second);
				thickness_res_sum += cal_res_one;
			}
			float avg_thickness_res = thickness_res_sum / static_cast<float>(num);
			double Truthvalue = 0.0;
			double Difference = abs(depthtoplane_all_avg[thickness_index] - avg_thickness_res);
			Truthvalue = avg_thickness_res + Difference * 2;//这个difference*2是不是有点大
			thickresult.push_back(make_pair(Truthvalue, num));
		}
	}
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template<typename PointT> void
		thickness::Thickness<PointT>::thicknessCal_1(std::vector<ThicknessPair> ThicknessPair_list, std::vector<std::pair<float, int>>& thickresult, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
	{
		pcl::search::OrganizedNeighbor<PointXYZRGB>::Ptr tree(new search::OrganizedNeighbor<PointXYZRGB>());
		tree->setInputCloud(cloud);
		std::vector<int> pointIdxRadiusSearch;
		std::vector<float> pointRadiusSquaredDistance;
		for (int thickness_index = 0; thickness_index < static_cast<int>(ThicknessPair_list.size()); thickness_index++)
		{
			int num = ThicknessPair_list[thickness_index].point_num;
			float thickness_res_sum = 0;
			float avg_thickness_res;//初始平均厚度
			float real_res_sum = 0;
			float avg_real_res;//真实平均厚度
			float sum_p1_plane = 0.0;
			float sum_p2_plane = 0.0;
			float avg_p1_plane, avg_p2_plane;
			ThicknessPair one_thickness = ThicknessPair_list[thickness_index];
			for (int index = 0; index < static_cast<int>(one_thickness.corr.size()); index++)
			{
				pcl::PointXYZ p1, p2;
				p1 = one_thickness.corr[index].first;
				p2 = one_thickness.corr[index].second;
				float cal_res_one = pcl::euclideanDistance(p1, p2);
				thickness_res_sum += cal_res_one;
			}
			avg_thickness_res = thickness_res_sum / static_cast<float>(num);//初始平均厚度
			for (int index = 0; index < static_cast<int>(one_thickness.corr.size()); index++)
			{
				pointIdxRadiusSearch.clear();
				pointRadiusSquaredDistance.clear();
				pcl::PointXYZ p1, p2;
				pcl::PointXYZRGB mid_point;
				float real_dis;
				p1.x = one_thickness.corr[index].first.x;
				p1.y = one_thickness.corr[index].first.y;
				p1.z = one_thickness.corr[index].first.z;
				p2.x = one_thickness.corr[index].second.x;
				p2.y = one_thickness.corr[index].second.y;
				p2.z = one_thickness.corr[index].second.z;
				mid_point.x = (p1.x + p2.x) / 2.0;
				mid_point.y = (p1.y + p2.y) / 2.0;
				mid_point.z = (p1.z + p2.z) / 2.0;
				mid_point.r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
				mid_point.g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
				mid_point.b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
				float temp_dis = pcl::euclideanDistance(p1, p2);
				Eigen::Vector4f centroid;                    // 质心
				Eigen::Matrix3f covariance_matrix;           // 协方差矩阵
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZRGB>);
				cloud_in->clear();
				tree->radiusSearch(mid_point, avg_thickness_res * 0.4, pointIdxRadiusSearch, pointRadiusSquaredDistance);
				for (auto i : pointIdxRadiusSearch)
				{
					cloud_in->push_back(cloud->points[i]);
				}
				// 计算归一化协方差矩阵和质心
				pcl::computeMeanAndCovarianceMatrix(*cloud_in, covariance_matrix, centroid);
				// 计算协方差矩阵的特征值与特征向量
				Eigen::Matrix3f eigenVectors;
				Eigen::Vector3f eigenValues;
				pcl::eigen33(covariance_matrix, eigenVectors, eigenValues);
				// 查找最小特征值的位置
				Eigen::Vector3f::Index minRow, minCol;
				eigenValues.minCoeff(&minRow, &minCol);
				Eigen::Vector3f normal = eigenVectors.col(minCol);
				double a = normal[0];
				double b = normal[1];
				double c = normal[2];
				double d = -normal.dot(centroid.head<3>());
				double dis_p1_plane = pcl::pointToPlaneDistance(p1, a, b, c, d);
				double dis_p2_plane = pcl::pointToPlaneDistance(p2, a, b, c, d);
				sum_p1_plane += dis_p1_plane;
				sum_p2_plane += dis_p2_plane;
			}
			avg_p1_plane = sum_p1_plane / static_cast<float>(num);
			avg_p2_plane = sum_p2_plane / static_cast<float>(num);
			if (avg_p1_plane > avg_p2_plane)
			{
				if (avg_p1_plane > avg_thickness_res * 0.2)
				{
					for (int index = 0; index < static_cast<int>(one_thickness.corr.size()); index++)
					{
						pointIdxRadiusSearch.clear();
						pointRadiusSquaredDistance.clear();
						pcl::PointXYZRGB p1, p2;
						float real_dis;
						p1.x = one_thickness.corr[index].first.x;
						p1.y = one_thickness.corr[index].first.y;
						p1.z = one_thickness.corr[index].first.z;
						p2.x = one_thickness.corr[index].second.x;
						p2.y = one_thickness.corr[index].second.y;
						p2.z = one_thickness.corr[index].second.z;
						float temp_dis = pcl::euclideanDistance(p1, p2);
						Eigen::Vector4f centroid;                    // 质心
						Eigen::Matrix3f covariance_matrix;           // 协方差矩阵
						pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZRGB>);
						cloud_in->clear();
						tree->radiusSearch(p2, temp_dis * 0.4, pointIdxRadiusSearch, pointRadiusSquaredDistance);
						for (auto i : pointIdxRadiusSearch)
						{
							cloud_in->push_back(cloud->points[i]);
						}
						// 计算归一化协方差矩阵和质心
						pcl::computeMeanAndCovarianceMatrix(*cloud_in, covariance_matrix, centroid);
						// 计算协方差矩阵的特征值与特征向量
						Eigen::Matrix3f eigenVectors;
						Eigen::Vector3f eigenValues;
						pcl::eigen33(covariance_matrix, eigenVectors, eigenValues);
						// 查找最小特征值的位置
						Eigen::Vector3f::Index minRow, minCol;
						eigenValues.minCoeff(&minRow, &minCol);
						// 获取平面方程：AX+BY+CZ+D = 0的系数
						Eigen::Vector3f normal = eigenVectors.col(minCol);
						double a = normal[0];
						double b = normal[1];
						double c = normal[2];
						double d = -normal.dot(centroid.head<3>());
						float x0 = p1.x;
						float y0 = p1.y;
						float z0 = p1.z;
						float xp = ((b * b + c * c) * x0 - a * (b * y0 + c * z0 + d)) / (a * a + b * b + c * c);
						float yp = ((a * a + c * c) * y0 - b * (a * x0 + c * z0 + d)) / (a * a + b * b + c * c);
						float zp = ((a * a + b * b) * z0 - c * (a * x0 + b * y0 + d)) / (a * a + b * b + c * c);
						p1.x = xp;
						p1.y = yp;
						p1.z = zp;
						real_dis = pcl::euclideanDistance(p1, p2);
						real_res_sum += real_dis;

					}
					avg_real_res = real_res_sum / static_cast<float>(num);
				}
				else
				{
					avg_real_res = avg_thickness_res;
				}
			}
			else if (avg_p1_plane < avg_p2_plane)
			{
				if (avg_p2_plane > avg_thickness_res * 0.2)
				{
					for (int index = 0; index < static_cast<int>(one_thickness.corr.size()); index++)
					{
						pointIdxRadiusSearch.clear();
						pointRadiusSquaredDistance.clear();
						pcl::PointXYZRGB p1, p2;
						float real_dis;
						p1.x = one_thickness.corr[index].first.x;
						p1.y = one_thickness.corr[index].first.y;
						p1.z = one_thickness.corr[index].first.z;
						p2.x = one_thickness.corr[index].second.x;
						p2.y = one_thickness.corr[index].second.y;
						p2.z = one_thickness.corr[index].second.z;
						float temp_dis = pcl::euclideanDistance(p1, p2);
						Eigen::Vector4f centroid;                    // 质心
						Eigen::Matrix3f covariance_matrix;           // 协方差矩阵
						pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZRGB>);
						cloud_in->clear();
						tree->radiusSearch(p1, temp_dis * 0.4, pointIdxRadiusSearch, pointRadiusSquaredDistance);
						for (auto i : pointIdxRadiusSearch)
						{
							cloud_in->push_back(cloud->points[i]);
						}
						// 计算归一化协方差矩阵和质心
						pcl::computeMeanAndCovarianceMatrix(*cloud_in, covariance_matrix, centroid);
						// 计算协方差矩阵的特征值与特征向量
						Eigen::Matrix3f eigenVectors;
						Eigen::Vector3f eigenValues;
						pcl::eigen33(covariance_matrix, eigenVectors, eigenValues);
						// 查找最小特征值的位置
						Eigen::Vector3f::Index minRow, minCol;
						eigenValues.minCoeff(&minRow, &minCol);
						// 获取平面方程：AX+BY+CZ+D = 0的系数
						Eigen::Vector3f normal = eigenVectors.col(minCol);
						double a = normal[0];
						double b = normal[1];
						double c = normal[2];
						double d = -normal.dot(centroid.head<3>());
						float x0 = p2.x;
						float y0 = p2.y;
						float z0 = p2.z;
						float xp = ((b * b + c * c) * x0 - a * (b * y0 + c * z0 + d)) / (a * a + b * b + c * c);
						float yp = ((a * a + c * c) * y0 - b * (a * x0 + c * z0 + d)) / (a * a + b * b + c * c);
						float zp = ((a * a + b * b) * z0 - c * (a * x0 + b * y0 + d)) / (a * a + b * b + c * c);
						p2.x = xp;
						p2.y = yp;
						p2.z = zp;
						real_dis = pcl::euclideanDistance(p1, p2);
						real_res_sum += real_dis;
					}
					avg_real_res = real_res_sum / static_cast<float>(num);
				}
				else
				{
					avg_real_res = avg_thickness_res;
				}
			}
			else { avg_real_res = avg_thickness_res; }
			thickresult.push_back(make_pair(avg_real_res, num));
		}
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	std::vector<double> son_node_length;
	std::vector<double> son_node_roughness;
	std::vector<double> son_node_mindis;
	std::vector<double> son_node_fitlinedis;
	template<typename PointT> void
		thickness::Thickness<PointT>::find_son_node(Node* a)
	{

		if (a->m_left_ && a->m_right_)
		{
			find_son_node(a->m_left_);
			find_son_node(a->m_right_);
		}
		else
		{
			/*if (!a->m_left_ && !a->m_right_)
			{

			}*/
			son_node_length.push_back(a->m_length_);
			son_node_roughness.push_back(a->m_roughness_);

			/*node* temp_node= new node();
			temp_node = a->m_father_;*/
			double b = 0;
			b = cal_min_dis(a->m_father_->m_left_, a->m_father_->m_right_);
			son_node_mindis.push_back(b);
			double c = 0;
			c = cal_fit_line_dis(a->m_father_->m_left_, a->m_father_->m_right_);
			son_node_fitlinedis.push_back(c);
		}
	}



	template<typename PointT>
	void thickness::Thickness<PointT>::feature_select()
	{
		//数据集特征
		int record4 = 0;
		ofstream f("feature.txt");
		/*fstream f;*/
		//f.open(/*std::to_string(record4) +*/ "feature.txt", /*ios::out, */std::ios::app);
		for (auto onenode = m_ThicknessPair_list_.begin(); onenode != m_ThicknessPair_list_.end();)
		{
			
			vector<double> son_node_density;
			Node* temp_node1 = new Node();
			Node* temp_node2 = new Node();
			temp_node1 = onenode->thickness_pair.first;
			temp_node2 = onenode->thickness_pair.second;
			son_node_length.clear();
			son_node_roughness.clear();
			son_node_density.clear();
			son_node_mindis.clear();
			son_node_fitlinedis.clear();
			find_son_node(temp_node1);
			find_son_node(temp_node2);
			record4++;
			/*f << record4 << endl;*/
			//均值、方差
			double son_node_length_sum = std::accumulate(std::begin(son_node_length), std::end(son_node_length), 0.0);
			double son_node_length_mean = son_node_length_sum / son_node_length.size();
			double son_node_length_variance = 0.0;
			for (int i = 0; i < son_node_length.size(); i++)
			{
				son_node_length_variance = son_node_length_variance + pow(son_node_length[i] - son_node_length_mean, 2);
			}
			son_node_length_variance = son_node_length_variance / son_node_length.size();
			f << son_node_length_mean << " " << son_node_length_variance << " ";

			double son_node_roughness_sum = std::accumulate(std::begin(son_node_roughness), std::end(son_node_roughness), 0.0);
			double son_node_roughness_mean = son_node_roughness_sum / son_node_roughness.size();
			double son_node_roughness_variance = 0.0;
			for (int i = 0; i < son_node_roughness.size(); i++)
			{
				son_node_roughness_variance = son_node_roughness_variance + pow(son_node_roughness[i] - son_node_roughness_mean, 2);
			}
			son_node_roughness_variance = son_node_roughness_variance / son_node_roughness.size();
			f << son_node_roughness_mean << " " << son_node_roughness_variance << " ";

			//点密度、厚度值方差、点到拟合面
			m_search_->setInputCloud(m_original_cloud_);
			std::vector<int> pointIdxRadiusSearch;
			std::vector<float> pointRadiusSquaredDistance;
			pcl::PointXYZ midPoint;
			auto corrs = onenode->corr;
			vector<double> pointtoplanedis;
			pointtoplanedis.clear();
			for (auto onecorr : corrs)
			{
				vector<int> nearpoint;
				pcl::PointXYZ p1, p2;
				p1 = onecorr.first;
				p2 = onecorr.second;
				midPoint.x = (double)(p1.x + p2.x) / 2.0;
				midPoint.y = (double)(p1.y + p2.y) / 2.0;
				midPoint.z = (double)(p1.z + p2.z) / 2.0;
				double distance = pcl::euclideanDistance(p1, p2);
				pointIdxRadiusSearch.clear();
				pointRadiusSquaredDistance.clear();
				m_search_->radiusSearch(midPoint, 0.5 * distance, pointIdxRadiusSearch, pointRadiusSquaredDistance);
				//得到近邻点ID
				for (auto id : pointIdxRadiusSearch)
				{
					nearpoint.push_back(id);
				}
				pcl::PointCloud<pcl::PointXYZ>::Ptr tt(new pcl::PointCloud<pcl::PointXYZ>);
				for (auto id : nearpoint)
				{
					tt->push_back(m_original_cloud_->points[id]);
				}
				son_node_density.push_back(tt->size());
				if (tt->size() > 4)
				{
					pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr model_plane(new pcl::SampleConsensusModelPlane<pcl::PointXYZ>(tt));
					pcl::RandomSampleConsensus<pcl::PointXYZ> ransac(model_plane);
					ransac.setDistanceThreshold(0.01);
					ransac.computeModel();
					pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_plane(new pcl::PointCloud<pcl::PointXYZ>);//平面点云
					vector<int> inliers;
					ransac.getInliers(inliers);
					pcl::copyPointCloud<pcl::PointXYZ>(*tt, inliers, *cloud_plane);
					Eigen::VectorXf coefficient;
					ransac.getModelCoefficients(coefficient);
					double A = coefficient[0];
					double B = coefficient[1];
					double C = coefficient[2];
					double D = coefficient[3];
					auto dis1 = pcl::pointToPlaneDistance(p1, A, B, C, D);
					auto dis2 = pcl::pointToPlaneDistance(p2, A, B, C, D);
					pointtoplanedis.push_back(dis1);
					pointtoplanedis.push_back(dis2);
				}

			}
			double pointtoplanedis_sum = std::accumulate(std::begin(pointtoplanedis), std::end(pointtoplanedis), 0.0);
			double pointtoplanedis_mean = pointtoplanedis_sum / pointtoplanedis.size();
			double pointtoplanedis_variance = 0.0;
			for (int i = 0; i < pointtoplanedis.size(); i++)
			{
				pointtoplanedis_variance = pointtoplanedis_variance + pow(pointtoplanedis[i] - pointtoplanedis_mean, 2);
			}
			pointtoplanedis_variance = pointtoplanedis_variance / pointtoplanedis.size();
			f << pointtoplanedis_variance << " " << pointtoplanedis_mean << " ";

			double son_node_density_sum = std::accumulate(std::begin(son_node_density), std::end(son_node_density), 0.0);
			double son_node_density_mean = son_node_density_sum / son_node_density.size();
			double son_node_density_variance = 0.0;
			for (int i = 0; i < son_node_density.size(); i++)
			{
				son_node_density_variance = son_node_density_variance + pow(son_node_density[i] - son_node_density_mean, 2);
			}
			son_node_density_variance = son_node_density_variance / son_node_density.size();
			f << son_node_density_mean << " " << son_node_density_variance << " ";

			//厚度值方差
			vector<double> thickness_value;
			for (auto point : corrs)
			{
				pcl::PointXYZ p1, p2;
				p1 = point.first;
				p2 = point.second;
				double distance = pcl::euclideanDistance(p1, p2);
				thickness_value.push_back(distance);
			}
			double thickness_value_sum = std::accumulate(std::begin(thickness_value), std::end(thickness_value), 0.0);
			double thickness_value_mean = thickness_value_sum / thickness_value.size();
			double thickness_value_variance = 0.0;
			for (int i = 0; i < thickness_value.size(); i++)
			{
				thickness_value_variance = thickness_value_variance + pow(thickness_value[i] - thickness_value_mean, 2);
			}
			thickness_value_variance = thickness_value_variance / thickness_value.size();
			f << thickness_value_variance << " ";


			double son_node_mindis_sum = std::accumulate(std::begin(son_node_mindis), std::end(son_node_mindis), 0.0);
			double son_node_mindis_mean = son_node_mindis_sum / son_node_mindis.size();
			double son_node_mindis_variance = 0.0;
			for (int i = 0; i < son_node_mindis.size(); i++)
			{
				son_node_mindis_variance = son_node_mindis_variance + pow(son_node_mindis[i] - son_node_mindis_mean, 2);
			}
			son_node_mindis_variance = son_node_mindis_variance / son_node_mindis.size();
			f << son_node_mindis_mean << " " << son_node_mindis_variance << " ";

			double son_node_fitlinedis_sum = std::accumulate(std::begin(son_node_fitlinedis), std::end(son_node_fitlinedis), 0.0);
			double son_node_fitlinedis_mean = son_node_fitlinedis_sum / son_node_fitlinedis.size();
			double son_node_fitlinedis_variance = 0.0;
			for (int i = 0; i < son_node_fitlinedis.size(); i++)
			{
				son_node_fitlinedis_variance = son_node_fitlinedis_variance + pow(son_node_fitlinedis[i] - son_node_fitlinedis_mean, 2);
			}
			son_node_fitlinedis_variance = son_node_fitlinedis_variance / son_node_fitlinedis.size();
			f << son_node_fitlinedis_mean << " " << son_node_fitlinedis_variance<<endl;

			//质心
			//Cloud::Ptr cloud(new Cloud);
			//for (auto point : corrs)
			//{
			//	cloud->push_back(point.first);
			//	cloud->push_back(point.second);
			//}
			//Eigen::Vector4f centroid;
			//pcl::compute3DCentroid(*cloud, centroid);
			//std::cout << "zhixin=" << centroid[2] << endl;
			
			onenode++;
		}
		f.close();
	}


	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template<typename PointT> void
	thickness::Thickness<PointT>::statistics(std::vector<std::pair<float, int>>& S, std::vector<std::vector<float>>& aabb_xy, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
	{
		/*for (auto t : m_ThicknessPair_list_)
		{
			S.push_back(make_pair(t.thickness_val, t.point_num));
			std::vector<float> temp;
			temp.push_back(t.min_x);
			temp.push_back(t.min_y);
			temp.push_back(t.max_x);
			temp.push_back(t.max_y);
			aabb_xy.push_back(temp);
			temp.clear();
		}*/
		/*std::vector<ThicknessPair> inclass;
		std::vector<ThicknessPair> difclass;
		for (auto temp : m_ThicknessPair_list_)
		{
			if (temp.thickness_pair.first->m_type_ != temp.thickness_pair.second->m_type_)
			{
				difclass.push_back(temp);
			}
			else
			{
				inclass.push_back(temp);
			}
		}*/

		if (ThicknessPair_list_difclass.size() > 0)
		{
			//std::vector<ThicknessPair> difclass_std;
			//std::vector<float> std_result;
			//std::vector<std::pair<float, int>> thickresult;
			//thicknessCal(difclass, difclass_std, thickresult, std_result);
			validation_fackAngle(ThicknessPair_list_difclass, S, cloud);
			for (auto dift:ThicknessPair_list_difclass)
			{
				/*auto corrs = dift.corr;
				Cloud::Ptr cloud(new Cloud);
				for (auto point : corrs)
				{
					cloud->push_back(point.first);
					cloud->push_back(point.second);
				}
				Eigen::Vector4f centroid;
				pcl::compute3DCentroid(*cloud, centroid);
				std::cout << "difclass=" << centroid[2] << endl;*/

				std::vector<float> temp1;
				temp1.push_back(dift.min_x);
				temp1.push_back(dift.min_y);
				temp1.push_back(dift.max_x);
				temp1.push_back(dift.max_y);
				aabb_xy.push_back(temp1);
				temp1.clear();
			}
		}
		if (ThicknessPair_list_inclass.size() > 0)
		{
			thicknessCal_1(ThicknessPair_list_inclass, S, cloud);
			for (auto t:ThicknessPair_list_inclass)
			{
				/*auto corrs = t.corr;
				Cloud::Ptr cloud(new Cloud);
				for (auto point : corrs)
				{
					cloud->push_back(point.first);
					cloud->push_back(point.second);
				}
				Eigen::Vector4f centroid;
				pcl::compute3DCentroid(*cloud, centroid);
				std::cout <<"inclass=" << centroid[2] << endl;*/

				std::vector<float> temp2;
				temp2.push_back(t.min_x);
				temp2.push_back(t.min_y);
				temp2.push_back(t.max_x);
				temp2.push_back(t.max_y);
				aabb_xy.push_back(temp2);
				temp2.clear();
			}
		}
	}

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template class thickness::Thickness<PointXYZ>;