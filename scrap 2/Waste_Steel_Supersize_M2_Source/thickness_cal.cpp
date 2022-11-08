#pragma once
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
using namespace std;
using namespace pcl;
using namespace Eigen;

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

		if (cloud_new_a->size() >=3)
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
			temp_node->m_pca_point =line_point;
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

	//测试用,用于保存聚类后的点云
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_after(new pcl::PointCloud<pcl::PointXYZRGB>);
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
		writer.write<pcl::PointXYZRGB>(ss.str(), *cloud_after, false);

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> bool
thickness::Thickness<PointT>::prepare()
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
	for (int i = 0; i < static_cast<int>(m_A_cloud_->size());i++)
	{
		Node* temp_node = new Node();
		Eigen::Vector4f temp_p, temp_dir;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		cloud->push_back(m_A_cloud_->points[i]);
		temp_node->m_pmin_ = m_A_cloud_->points[i];
		temp_node->m_pmax_ = m_A_cloud_->points[i];
		temp_node->m_cloud_ = cloud;
		temp_node->m_type_=0;
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
		if ((*onenode)->m_cloud_ ->size()<3)
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
		pcl::computeMeanAndCovarianceMatrix(*p1_cloud, covariance_matrix, centroid);
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
			if (i==j||is_merge[j] != 0)
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
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_after(new pcl::PointCloud<pcl::PointXYZRGB>);
	for (auto onenode : m_plane_list_)
	{
		int Random_color_r, Random_color_g, Random_color_b;
		Random_color_r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		temp_cloud = onenode.plane_cloud;
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
	ss << "平面检测后" << ".pcd";
	pcl::PCDWriter writer;
	writer.write<pcl::PointXYZRGB>(ss.str(), *cloud_after, false);

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
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_after(new pcl::PointCloud<pcl::PointXYZRGB>);
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
	writer.write<pcl::PointXYZRGB>(ss.str(), *cloud_after, false);

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::density_detection()
{
	for (int i = 0; i < static_cast<int>(m_parallel_list_.size()); i++)
	{
		auto node_list = m_parallel_list_[i].node_list;
		for (int j = 0; j < static_cast<int>(m_parallel_list_[i].node_list.size()); j++)
		{
			ThicknessPair temp_thicknesspair;
			for (int k = 0; k < static_cast<int>(m_parallel_list_[i].node_list.size()); k++)
			{
				if (j <=k )
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
					if ((double)corr.size() / (double)node_list[j]->m_cloud_->size() < 0.4)
					{
						continue;
					}
				}
				else
				{
					corr = corr_BA;
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
					m_search_->radiusSearch(midPoint, distance*0.5 * 0.5, pointIdxRadiusSearch, pointRadiusSquaredDistance);
					int size = pointIdxRadiusSearch.size();
					if (size > 0)
					{
						total_have_neiborghtor++;
						pcl::PointCloud<pcl::PointXYZ>::Ptr tt(new pcl::PointCloud<pcl::PointXYZ>);
						for (auto id : pointIdxRadiusSearch)
						{
							tt->push_back(m_original_cloud_->points[id]);
						}
						pcl::io::savePCDFileASCII("tt.pcd", *tt);
					}
				}
				avg_thickness=sum_thickness/ (double)number_of_pairs;
				double radio = (double)total_have_neiborghtor / (double)number_of_pairs;
				cout <<avg_thickness<< endl;
				if (radio > 0.7&&avg_thickness<0.02)//参数
				{
					temp_thicknesspair.thickness_pair.first = node_list[j];
					temp_thicknesspair.thickness_pair.second = node_list[k];
					temp_thicknesspair.corr = corr;
					m_ThicknessPair_list_.push_back(temp_thicknesspair);
				}
			}	
			
		}
		
	}
	//测试
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_after(new pcl::PointCloud<pcl::PointXYZRGB>);
	for (auto onenode : m_ThicknessPair_list_)
	{
		int Random_color_r, Random_color_g, Random_color_b;
		Random_color_r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		auto pairs = onenode.corr;
		for (auto point : pairs)
		{
			pcl::PointXYZRGB tempa,tempb;
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
	for (auto point:(*a_cloud))
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
template class thickness::Thickness<PointXYZ>;



