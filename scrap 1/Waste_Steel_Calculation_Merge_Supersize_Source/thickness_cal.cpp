#pragma once
#include "thickness_cal.h"
#include "kMeans.cpp"
//#include "fakeangle.cpp"
#include "fakeboundary.cpp"
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
#include <pcl/search/organized.h>
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
	min_pts_per_cluster_(1),
	max_pts_per_cluster_(std::numeric_limits<int>::max()),
	point_direction_a_(),
	point_direction_b_(),
	point_direction_c_(),
	margin_a(),
	margin_b(),
	margin_c(),
	search_(),
	radius_1_(0.001f),
	radius_2_(0.015f),//类内寻找时默认最大的厚度
	neighbour_number_(10),
	point_neighbours_a_(0),
	point_neighbours_b_(0),
	point_neighbours_c_(0),
	point_neighbours_aWithb_(0),
	point_neighbours_dis_a_(0),
	point_neighbours_dis_b_(0),
	point_neighbours_dis_c_(0),
	point_neighbours_aWithb_dis_(0),
	point_labels_a_(0),
	point_labels_b_(0),
	point_labels_c_(0),
	angel_(30.0f / 180.0f * static_cast<float> (M_PI)),
	angle_with_direction_(30.0f / 180.0f * static_cast<float> (M_PI)),
	dis_threshold_(0.0008),
	geodesic_distances_threshold_(0.015),
	detection_type_(true),
	thickness_index_(0),
	thickness_index_validation(0),
	num_pts_in_thickness_(0),
	thickness_result_(0),
	std_result_(0),
	number_of_thicknesses_(0)
{
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
thickness::Thickness<PointT>::~Thickness()
{
	point_neighbours_a_.clear();
	point_neighbours_b_.clear();
	point_neighbours_c_.clear();
	point_neighbours_aWithb_.clear();
	point_neighbours_dis_a_.clear();
	point_neighbours_dis_b_.clear();
	point_neighbours_dis_c_.clear();
	point_neighbours_aWithb_.clear();
	point_labels_a_.clear();
	point_labels_b_.clear();
	point_labels_c_.clear();
	thickness_index_.clear();
	std_result_.clear();
	thickness_index_validation.clear();
	num_pts_in_thickness_.clear();
	thickness_result_.clear();
	
	
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> int
thickness::Thickness<PointT>::getMinClusterSize()
{
	return (min_pts_per_cluster_);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> int
thickness::Thickness<PointT>::getMaxClusterSize()
{
	return (max_pts_per_cluster_);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::setMinClusterSize(int min_cluster_size)
{
	min_pts_per_cluster_ = min_cluster_size;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::setMaxClusterSize(int max_cluster_size)
{
	max_pts_per_cluster_ = max_cluster_size;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> Eigen::Vector3f
thickness::Thickness<PointT>::getDirection(int type, int point_index) const
{
	if (type == 0)
		return((*point_direction_a_)[point_index]);
	if (type == 1)
		return((*point_direction_b_)[point_index]);
	if (type == 2)
		return((*point_direction_c_)[point_index]);
	else
		std::cout << "输入错误,请重新输入边界类型" << std::endl;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> typename thickness::Thickness<PointT>::KdTreePtr
thickness::Thickness<PointT>::getSearchMethod() const
{
	return(search_);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::setSearchMethod(const KdTreePtr& tree)
{
	search_ = tree;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> float
thickness::Thickness<PointT>::getRadius1() const
{
	return(radius_1_);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> float
thickness::Thickness<PointT>::getRadius2() const
{
	return(radius_2_);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::setRadius1(float search_radius1)
{
	radius_1_ = search_radius1;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::setRadius2(float search_radius2)
{
	radius_2_ = search_radius2;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::setGeodesicDis(float geodesic_distance)
{
	geodesic_distances_threshold_ = geodesic_distance;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::setDistanceToPlane(float dis)
{
	dis_threshold_ = dis;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> unsigned int
thickness::Thickness<PointT>::getNumberOfNeighbours() const
{
	return (neighbour_number_);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::setAngelThreshold(float angel)
{
	angel_ = angel;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::setAngelThresholdWithDirection(float angle_with_direction)
{
	angle_with_direction_ = angle_with_direction;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::inputCloud(const PointCloudConstPtr& cloudA, const PointCloudConstPtr& cloudB)
{
	margin_a = cloudA;
	margin_b = cloudB;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::inputCloud(const PointCloudConstPtr& cloud)
{
	margin_c = cloud;
}

template<typename PointT> void
thickness::Thickness<PointT>::inputOriginalCloud(const PointCloudConstPtr& cloud)
{
	original_cloud = cloud;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::inputCloudDirection(const PointDirectionPtr& CloudADirectionPtr, const PointDirectionPtr& CloudBDirectionPtr)
{
	point_direction_a_ = CloudADirectionPtr;
	point_direction_b_ = CloudBDirectionPtr;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::inputCloudDirection(const PointDirectionPtr& CloudDirectionPtr)
{
	point_direction_c_ = CloudDirectionPtr;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> bool
thickness::Thickness<PointT>::prepareWithSegmentation()
{
	if (margin_a == nullptr || margin_b == nullptr)
		return(false);
	if (point_direction_a_ == nullptr || point_direction_b_ == nullptr)
		return(false);
	if (!search_)
	{
		search_.reset(new pcl::search::KdTree<PointT>);
		return(true);
	}
	else
		return(true);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> bool
thickness::Thickness<PointT>::prepareWithSegmentationSelf()
{
	if (margin_c == nullptr)
		return(false);
	if (point_direction_c_ == nullptr)
		return(false);
	if (!search_)
	{
		search_.reset(new pcl::search::KdTree<PointT>);
		return(true);
	}
	else
		return(true);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::findNeighbours(int type)
{
	if (type == 0)
	{
		search_->setInputCloud(margin_a);
		int point_number = static_cast<int>(margin_a->size());
		std::vector<int> neighbours;
		std::vector<float> distances;
		point_neighbours_a_.resize(margin_a->size());
		point_neighbours_dis_a_.resize(margin_a->size());
		for (int i_point = 0; i_point < point_number; i_point++)
		{
			neighbours.clear();
			distances.clear();
			search_->radiusSearch(i_point, radius_1_, neighbours, distances);
			//search_->nearestKSearch(i_point, 50, neighbours, distances);
			point_neighbours_a_[i_point].swap(neighbours);
			point_neighbours_dis_a_[i_point].swap(distances);

		}
		return;
	}
	else if (type == 1)
	{
		search_->setInputCloud(margin_b);
		int point_number = static_cast<int>(margin_b->size());
		std::vector<int> neighbours;
		std::vector<float> distances;
		point_neighbours_b_.resize(margin_b->size());
		point_neighbours_dis_b_.resize(margin_b->size());
		for (int i_point = 0; i_point < point_number; i_point++)
		{
			neighbours.clear();
			distances.clear();
			search_->radiusSearch(i_point, radius_1_, neighbours, distances);
			//search_->nearestKSearch(i_point, 50, neighbours, distances);
			point_neighbours_b_[i_point].swap(neighbours);
			point_neighbours_dis_b_[i_point].swap(distances);
		}
		return;
	}
	else if (type == 2)
	{
		search_->setInputCloud(margin_c);
		int point_number = static_cast<int>(margin_c->size());
		std::vector<int> neighbours;
		std::vector<float> distances;
		point_neighbours_c_.resize(margin_c->size());
		point_neighbours_dis_c_.resize(margin_c->size());
		for (int i_point = 0; i_point < point_number; i_point++)
		{
			neighbours.clear();
			distances.clear();
			search_->radiusSearch(i_point, radius_2_, neighbours, distances);
			//search_->nearestKSearch(i_point, 50, neighbours, distances);
			point_neighbours_c_[i_point].swap(neighbours);
			point_neighbours_dis_c_[i_point].swap(distances);
		}
		return;
	}
	else
	{
		std::cout << "输入参数错误,请重新输入进行邻域查找的边界类型" << std::endl;
	}

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::findNeighboursAWithB()
{
	if (static_cast<int>(margin_a->size()) != 0 && static_cast<int>(margin_b->size()) != 0)
	{
		search_->setInputCloud(margin_b);
	}
	int point_number = static_cast<int>(margin_a->size());
	pcl::PointXYZ searchPoint;
	std::vector<int> neighbours;
	std::vector<float> distances;
	point_neighbours_aWithb_.resize(margin_a->size());
	point_neighbours_aWithb_dis_.resize(margin_a->size());
	for (int i_point = 0; i_point < point_number; i_point++)
	{
		neighbours.clear();
		distances.clear();
		searchPoint.x = margin_a->points[i_point].x;
		searchPoint.y = margin_a->points[i_point].y;
		searchPoint.z = margin_a->points[i_point].z;
		search_->radiusSearch(searchPoint, 0.015, neighbours, distances);//可变的参数
		//search_->nearestKSearch(searchPoint, 50, neighbours, distances);
		point_neighbours_aWithb_[i_point].swap(neighbours);
		point_neighbours_aWithb_dis_[i_point].swap(distances);
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Thickness<PointT>::thicknessDetection(std::vector<thicknessIndex>& all_thickness_pair_index, int& detection_type)
{
	if (detection_type == 1)
	{
		thickness_index_.clear();
		all_thickness_pair_index.clear();
		point_neighbours_a_.clear();
		point_neighbours_b_.clear();
		point_neighbours_c_.clear();
		point_neighbours_aWithb_.clear();
		point_neighbours_dis_a_.clear();
		point_neighbours_dis_b_.clear();
		point_neighbours_dis_c_.clear();
		point_neighbours_aWithb_.clear();
		num_pts_in_thickness_.clear();
		number_of_thicknesses_ = 0;
		point_labels_a_.clear();
		point_labels_b_.clear();
		point_labels_c_.clear();
		bool segmentation_is_possible = this->prepareWithSegmentation();
		if (!segmentation_is_possible)
		{
			std::cout << "请查看输入的类间数据" << std::endl;
			return;
		}
		findNeighbours(0);
		findNeighbours(1);
		findNeighboursAWithB();
		int counter = 0;
		int segment_label = 0;//厚度标签类型
		int num_of_pts_a = static_cast<int>(margin_a->size());
		int num_of_pts_b = static_cast<int>(margin_b->size());
		point_labels_a_.resize(num_of_pts_a, -1);
		point_labels_b_.resize(num_of_pts_b, -1);
		int segmented_pair_number = 0;
		int number_of_segment = 0;
		while (1)
		{
			//初始化第一个种子厚度点对
			pair<int, int> first_seed;
			bool finded_first_seed = false;
			for (int seed_point_p1 = 0; seed_point_p1 < num_of_pts_a; seed_point_p1++)
			{
				if (point_labels_a_[seed_point_p1] != -1)
				{
					continue;
				}
				if (point_neighbours_aWithb_[seed_point_p1].size() == 0)
				{
					continue;
				}
				float data[3];
				std::vector<int> after_planedis_cal;//符合平面距离阈值的点的索引
				data[0] = margin_a->points[seed_point_p1].x;
				data[1] = margin_a->points[seed_point_p1].y;
				data[2] = margin_a->points[seed_point_p1].z;
				pcl::PointXYZ p1;
				p1.x = data[0];
				p1.y = data[1];
				p1.z = data[2];
				Eigen::Vector4f p1_direction;//平面一般方程的系数
				p1_direction(0) = (*point_direction_a_)[seed_point_p1](0);
				p1_direction(1) = (*point_direction_a_)[seed_point_p1](1);
				p1_direction(2) = (*point_direction_a_)[seed_point_p1](2);
				p1_direction(3) = -(data[0] * p1_direction(0) + data[1] * p1_direction(1) + data[2] * p1_direction(2));
				Vector3f direction_p1, direction_q1;//方向向量（为了求角度）
				direction_p1(0) = p1_direction(0);
				direction_p1(1) = p1_direction(1);
				direction_p1(2) = p1_direction(2);
				for (int selected_i = 0; selected_i < point_neighbours_aWithb_[seed_point_p1].size(); selected_i++)
				{
					if (point_labels_b_[point_neighbours_aWithb_[seed_point_p1][selected_i]] != -1)
					{
						continue;
					}
					float data1[3];
					data1[0] = margin_b->points[point_neighbours_aWithb_[seed_point_p1][selected_i]].x;
					data1[1] = margin_b->points[point_neighbours_aWithb_[seed_point_p1][selected_i]].y;
					data1[2] = margin_b->points[point_neighbours_aWithb_[seed_point_p1][selected_i]].z;
					float dis_to_plane = abs(p1_direction(0) * data1[0] + p1_direction(1) * data1[1] + p1_direction(2) * data1[2] + p1_direction(3)) / sqrt(p1_direction(0)\
						* p1_direction(0) + p1_direction(1) * p1_direction(1) + p1_direction(2) * p1_direction(2));
					if (dis_to_plane < dis_threshold_)
					{
						after_planedis_cal.push_back(point_neighbours_aWithb_[seed_point_p1][selected_i]);//从一个点的全部索引点中取到平面距离小于阈值的
					}
				}
				if (after_planedis_cal.size() == 0)
				{
					continue;
				}
				std::vector<std::pair<float, int>> cal_dis_result;
				for (int selected_dis = 0; selected_dis < after_planedis_cal.size(); selected_dis++)
				{
					pcl::PointXYZ q1;
					q1.x = margin_b->points[after_planedis_cal[selected_dis]].x;
					q1.y = margin_b->points[after_planedis_cal[selected_dis]].y;
					q1.z = margin_b->points[after_planedis_cal[selected_dis]].z;
					float dis = pcl::euclideanDistance(p1, q1);
					cal_dis_result.push_back(make_pair(dis, after_planedis_cal[selected_dis]));
				}
				sort(cal_dis_result.begin(), cal_dis_result.end());//对符合距离条件的点计算与p1的距离然后排序
				for (int selected_ang = 0; selected_ang < cal_dis_result.size(); selected_ang++)
				{
					direction_q1(0) = (*point_direction_b_)[cal_dis_result[selected_ang].second](0);
					direction_q1(1) = (*point_direction_b_)[cal_dis_result[selected_ang].second](1);
					direction_q1(2) = (*point_direction_b_)[cal_dis_result[selected_ang].second](2);
					double angle_p1q1 = pcl::getAngle3D(direction_p1, direction_q1, false);
					if (angle_p1q1 > M_PI_2)
					{
						angle_p1q1 = angle_p1q1 - M_PI_2;
					}
					if (angle_p1q1 < angel_)
					{
						first_seed = make_pair(seed_point_p1, cal_dis_result[selected_ang].second);
						finded_first_seed = true;
						/*cout << margin_a->points[seed_point_p1].x <<"aa"<<margin_a->points[seed_point_p1].y << " " << margin_a->points[seed_point_p1].z << endl;
						cout << margin_b->points[cal_dis_result[selected_ang].second].x <<" " << margin_b->points[cal_dis_result[selected_ang].second].y <<" " << margin_b->points[cal_dis_result[selected_ang].second].z << endl;*/
						break;
					}
					else
					{
						continue;
					}
				}
				if (finded_first_seed)
				{
					break;
				}
				else
				{
					continue;
				}
			}
			if (!finded_first_seed)
			{
				return;
			}
			int pair_in_segment;
			vector<pair<int, int>> pairs;
			pair_in_segment = growRegion(first_seed, number_of_segment, pairs);
			if (pairs.size() < 4)
			{
				continue;
			}
			all_thickness_pair_index.push_back(pairs);
			thickness_index_.push_back(pairs);
			segmented_pair_number += pair_in_segment;
			num_pairs_in_segment.push_back(pair_in_segment);
			number_of_segment++;
		}

	}
	else if (detection_type == 2)
	{
		thickness_index_.clear();
		all_thickness_pair_index.clear();
		point_neighbours_a_.clear();
		point_neighbours_b_.clear();
		point_neighbours_c_.clear();
		point_neighbours_aWithb_.clear();
		point_neighbours_dis_a_.clear();
		point_neighbours_dis_b_.clear();
		point_neighbours_dis_c_.clear();
		point_neighbours_aWithb_.clear();
		num_pts_in_thickness_.clear();
		number_of_thicknesses_ = 0;
		point_labels_a_.clear();
		point_labels_b_.clear();
		point_labels_c_.clear();
		bool segmentation_is_possible = this->prepareWithSegmentationSelf();
		if (!segmentation_is_possible)
		{
			std::cout << "请查看输入类内数据" << std::endl;
			return;
		}
		findNeighbours(2);
		int segment_label = 0;//厚度标签类型
		int num_of_pts_c = static_cast<int>(margin_c->size());
		point_labels_c_.resize(num_of_pts_c, -1);
		int segmented_pair_number = 0;
		int number_of_segment = 0;
		//构造图结构
		//pcl::KdTreeFLANN<PointT> kdtree;
		//kdtree.setInputCloud(margin_c);
		//using namespace boost;
		//typedef property <edge_weight_t, float> Weight;
		//typedef adjacency_list<vecS, vecS, undirectedS, no_property, Weight> Graph;
		//Graph cloud_graph;
		//std::vector<std::vector<float> > geodesic_distances_;
		//for (size_t point_i = 0; point_i < margin_c->points.size(); ++point_i)
		//{
		//	std::vector<int> k_indices(neighbour_number_);
		//	std::vector<float> k_distances(neighbour_number_);
		//	kdtree.nearestKSearch(static_cast<int> (point_i), neighbour_number_, k_indices, k_distances);
		//	for (int k_i = 0; k_i < static_cast<int> (k_indices.size()); ++k_i)
		//		add_edge(point_i, k_indices[k_i], Weight(sqrtf(k_distances[k_i])), cloud_graph);
		//}
		//const size_t E = num_edges(cloud_graph),
		//	V = num_vertices(cloud_graph);
		////PCL_INFO("The graph has %lu vertices and %lu edges.\n", V, E);
		//geodesic_distances_.clear();
		//for (size_t i = 0; i < V; ++i)
		//{
		//	std::vector<float> aux(V);
		//	geodesic_distances_.push_back(aux);
		//}
		//johnson_all_pairs_shortest_paths(cloud_graph, geodesic_distances_);

		//寻找第一个种子对
		while (1)
		{
			//初始化第一个种子厚度点对
			pair<int, int> first_seed;
			bool finded_first_seed = false;
			for (int seed_point_p1 = 0; seed_point_p1 < num_of_pts_c; seed_point_p1++)
			{
				if (point_labels_c_[seed_point_p1] != -1)
				{
					continue;
				}
				if (point_neighbours_c_[seed_point_p1].size() == 0)
				{
					continue;
				}
				float data[3];
				std::vector<int> after_planedis_cal;//符合平面距离阈值的点的索引
				data[0] = margin_c->points[seed_point_p1].x;
				data[1] = margin_c->points[seed_point_p1].y;
				data[2] = margin_c->points[seed_point_p1].z;
				pcl::PointXYZ p1;
				p1.x = data[0];
				p1.y = data[1];
				p1.z = data[2];
				Eigen::Vector4f p1_direction;
				p1_direction(0) = (*point_direction_c_)[seed_point_p1](0);
				p1_direction(1) = (*point_direction_c_)[seed_point_p1](1);
				p1_direction(2) = (*point_direction_c_)[seed_point_p1](2);
				p1_direction(3) = -(data[0] * p1_direction(0) + data[1] * p1_direction(1) + data[2] * p1_direction(2));
				Vector3f direction_p1, direction_q1, direction_p1q1_candidate,direction_p1_2d;
				direction_p1(0) = p1_direction(0);
				direction_p1(1) = p1_direction(1);
				direction_p1(2) = p1_direction(2);
				for (int selected_i = 0; selected_i < point_neighbours_c_[seed_point_p1].size(); selected_i++)
				{
					if (point_labels_c_[point_neighbours_c_[seed_point_p1][selected_i]] != -1)
					{
						continue;
					}
					float data1[3];
					data1[0] = margin_c->points[point_neighbours_c_[seed_point_p1][selected_i]].x;
					data1[1] = margin_c->points[point_neighbours_c_[seed_point_p1][selected_i]].y;
					data1[2] = margin_c->points[point_neighbours_c_[seed_point_p1][selected_i]].z;
					float dis_to_plane = abs(p1_direction(0) * data1[0] + p1_direction(1) * data1[1] + p1_direction(2) * data1[2] + p1_direction(3)) / sqrt(p1_direction(0)\
						* p1_direction(0) + p1_direction(1) * p1_direction(1) + p1_direction(2) * p1_direction(2));

					//求p1q1和p1的向量夹角
					direction_p1q1_candidate(0) = data1[0] - data[0];
					direction_p1q1_candidate(1) = data1[1] - data[1];
					direction_p1q1_candidate(2) = 0;
					direction_p1_2d(0) = direction_p1(0);
					direction_p1_2d(1) = direction_p1(1);
					direction_p1_2d(2) = 0;
					float ang_p1q1_p1 = getAngle3D(direction_p1q1_candidate, direction_p1_2d, false);
					if (dis_to_plane < dis_threshold_)
					{
						if (ang_p1q1_p1 > 20.0f / 180.0f * static_cast<float> (M_PI) && ang_p1q1_p1 < 160.0f / 180.0f * static_cast<float> (M_PI))
						{
							after_planedis_cal.push_back(point_neighbours_c_[seed_point_p1][selected_i]);
						}
						
					}
					/*float geodesic_distances_p1q1 = geodesic_distances_[seed_point_p1][point_neighbours_c_[seed_point_p1][selected_i]];
					if (dis_to_plane<dis_threshold_ && geodesic_distances_p1q1>geodesic_distances_threshold_)
					{
						after_planedis_cal.push_back(point_neighbours_c_[seed_point_p1][selected_i]);
					}*/
				}
				if (after_planedis_cal.size() == 0)
				{
					continue;
				}
				std::vector<std::pair<float, int>> cal_dis_result;
				for (int selected_dis = 0; selected_dis < after_planedis_cal.size(); selected_dis++)
				{
					pcl::PointXYZ q1;
					q1.x = margin_c->points[after_planedis_cal[selected_dis]].x;
					q1.y = margin_c->points[after_planedis_cal[selected_dis]].y;
					q1.z = margin_c->points[after_planedis_cal[selected_dis]].z;
					float dis = pcl::euclideanDistance(p1, q1);
					cal_dis_result.push_back(make_pair(dis, after_planedis_cal[selected_dis]));
				}
				sort(cal_dis_result.begin(), cal_dis_result.end(), greater<std::pair<float, int>>());//找最大距离
				for (int selected_ang = 0; selected_ang < cal_dis_result.size(); selected_ang++)
				{
					direction_q1(0) = (*point_direction_c_)[cal_dis_result[selected_ang].second](0);
					direction_q1(1) = (*point_direction_c_)[cal_dis_result[selected_ang].second](1);
					direction_q1(2) = (*point_direction_c_)[cal_dis_result[selected_ang].second](2);
					double angle_p1q1 = pcl::getAngle3D(direction_p1, direction_q1, false);
					if (angle_p1q1 > M_PI_2)
					{
						angle_p1q1 = angle_p1q1 - M_PI_2;
					}
					if (angle_p1q1 < angel_)
					{
						first_seed = make_pair(seed_point_p1, cal_dis_result[selected_ang].second);
						finded_first_seed = true;
						//cout << margin_c->points[seed_point_p1].x << " " << margin_c->points[seed_point_p1].y << " " << margin_c->points[seed_point_p1].z << endl;
						//cout << margin_c->points[cal_dis_result[selected_ang].second].x << " " << margin_c->points[cal_dis_result[selected_ang].second].y << " " << margin_c->points[cal_dis_result[selected_ang].second].z << endl; 
						break;
					}
					else
					{
						continue;
					}
				}
				if (finded_first_seed)
				{
					break;
				}
				else
				{
					continue;
				}
			}
			if (!finded_first_seed)
			{
				return;
			}
			int pair_in_segment;
			vector<pair<int, int>> pairs;
			//pair_in_segment = growRegionSelf(first_seed, number_of_segment, pairs, geodesic_distances_);
			pair_in_segment = growRegionSelf1(first_seed, number_of_segment, pairs);
			if (pairs.size() < 4)
			{
				continue;
			}
			all_thickness_pair_index.push_back(pairs);
			thickness_index_.push_back(pairs);
			segmented_pair_number += pair_in_segment;
			num_pairs_in_segment.push_back(pair_in_segment);
			number_of_segment++;
		}

	}
	else
	{
		cout << "重新输入检测类型" << endl;
	}


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> int
thickness::Thickness<PointT>::growRegion(std::pair<int, int> seed, int segment_number, vector<std::pair<int, int>>& segment_pairs)
{
	std::queue<pair<int, int>> seeds;
	seeds.push(seed);
	point_labels_a_[seed.first] = segment_number;
	point_labels_b_[seed.second] = segment_number;
	int num_pts_in_segment = 1;
	segment_pairs.push_back(seed);
	while (!seeds.empty())
	{
		std::pair<int, int> curr_seed;
		curr_seed = seeds.front();
		seeds.pop();
		//用seed对构成的两个平面来过滤掉一些点
		Vector3f point_p1, point_q1, vector_p1q1, vector_q1p1, p1_direction;
		point_p1(0) = margin_a->points[curr_seed.first].x;
		point_p1(1) = margin_a->points[curr_seed.first].y;
		point_p1(2) = margin_a->points[curr_seed.first].z;
		point_q1(0) = margin_b->points[curr_seed.second].x;
		point_q1(1) = margin_b->points[curr_seed.second].y;
		point_q1(2) = margin_b->points[curr_seed.second].z;
		vector_p1q1 = point_q1 - point_p1;
		vector_q1p1 = point_p1 - point_q1;
		Vector4f plane_p1q1 = getPlane(point_p1, vector_p1q1);
		Vector4f plane_q1p1 = getPlane(point_q1, vector_q1p1);
		p1_direction(0) = (*point_direction_a_)[curr_seed.first](0);
		p1_direction(1) = (*point_direction_a_)[curr_seed.first](1);
		p1_direction(2) = (*point_direction_a_)[curr_seed.first](2);
		vector <int> plane_cal_w1, plane_cal_w2;//经过到初始pair组成平面距离过滤的点的索引
		for (int select_with_planep1 = 0; select_with_planep1 < point_neighbours_a_[curr_seed.first].size(); select_with_planep1++)
		{
			if (point_labels_a_[point_neighbours_a_[curr_seed.first][select_with_planep1]] != -1)
			{
				continue;
			}
			float data[3];
			data[0] = margin_a->points[point_neighbours_a_[curr_seed.first][select_with_planep1]].x;
			data[1] = margin_a->points[point_neighbours_a_[curr_seed.first][select_with_planep1]].y;
			data[2] = margin_a->points[point_neighbours_a_[curr_seed.first][select_with_planep1]].z;
			float dis_to_plane = abs(plane_p1q1(0) * data[0] + plane_p1q1(1) * data[1] + plane_p1q1(2) * data[2] + plane_p1q1(3)) / sqrt(plane_p1q1(0)\
				* plane_p1q1(0) + plane_p1q1(1) * plane_p1q1(1) + plane_p1q1(2) * plane_p1q1(2));
			if (dis_to_plane < dis_threshold_)
			{
				plane_cal_w1.push_back(point_neighbours_a_[curr_seed.first][select_with_planep1]);
			}
		}
		if (plane_cal_w1.size() == 0)
		{
			continue;
		}
		for (int select_with_planep2 = 0; select_with_planep2 < point_neighbours_b_[curr_seed.second].size(); select_with_planep2++)
		{
			if (point_labels_b_[point_neighbours_b_[curr_seed.second][select_with_planep2]] != -1)
			{
				continue;
			}
			float data1[3];
			data1[0] = margin_b->points[point_neighbours_b_[curr_seed.second][select_with_planep2]].x;
			data1[1] = margin_b->points[point_neighbours_b_[curr_seed.second][select_with_planep2]].y;
			data1[2] = margin_b->points[point_neighbours_b_[curr_seed.second][select_with_planep2]].z;
			float dis_to_plane = abs(plane_q1p1(0) * data1[0] + plane_q1p1(1) * data1[1] + plane_q1p1(2) * data1[2] + plane_q1p1(3)) / sqrt(plane_q1p1(0)\
				* plane_q1p1(0) + plane_q1p1(1) * plane_q1p1(1) + plane_q1p1(2) * plane_q1p1(2));
			if (dis_to_plane < dis_threshold_)
			{
				plane_cal_w2.push_back(point_neighbours_b_[curr_seed.second][select_with_planep2]);
			}
		}
		if (plane_cal_w2.size() == 0)
		{
			continue;
		}
		//进行w1w2子集点中的厚度点对搜索
		for (int select_p2 = 0; select_p2 < plane_cal_w1.size(); select_p2++)
		{
			Vector3f select_p2_direction;
			select_p2_direction(0) = (*point_direction_a_)[plane_cal_w1[select_p2]](0);
			select_p2_direction(1) = (*point_direction_a_)[plane_cal_w1[select_p2]](1);
			select_p2_direction(2) = (*point_direction_a_)[plane_cal_w1[select_p2]](2);
			float ang_p1p2 = getAngle3D(p1_direction, select_p2_direction, false);
			if (ang_p1p2 > M_PI_2)
			{
				ang_p1p2 = ang_p1p2 - M_PI_2;
			}
			vector <int> select_q2_index;
			if (ang_p1p2 < angle_with_direction_)//如果p2的选择点的方向与p1的方向相似
			{
				Vector3f select_p2_xyz;
				select_p2_xyz(0) = margin_a->points[plane_cal_w1[select_p2]].x;
				select_p2_xyz(1) = margin_a->points[plane_cal_w1[select_p2]].y;
				select_p2_xyz(2) = margin_a->points[plane_cal_w1[select_p2]].z;
				Vector4f select_p2_plane = getPlane(select_p2_xyz, select_p2_direction);
				for (int select_q2 = 0; select_q2 < plane_cal_w2.size(); select_q2++)
				{
					if (point_labels_b_[plane_cal_w2[select_q2]] != -1)
					{
						continue;
					}
					float data[3];
					data[0] = margin_b->points[plane_cal_w2[select_q2]].x;
					data[1] = margin_b->points[plane_cal_w2[select_q2]].y;
					data[2] = margin_b->points[plane_cal_w2[select_q2]].z;
					float dis_to_plane = abs(select_p2_plane(0) * data[0] + select_p2_plane(1) * data[1] + select_p2_plane(2) * data[2] + select_p2_plane(3))\
						/ sqrt(select_p2_plane(0) * select_p2_plane(0) + select_p2_plane(1) * select_p2_plane(1) + select_p2_plane(2) * select_p2_plane(2));
					if (dis_to_plane < dis_threshold_)
					{
						select_q2_index.push_back(plane_cal_w2[select_q2]);
						/*cout << margin_b->points[plane_cal_w2[select_q2]].x <<" " << margin_b->points[plane_cal_w2[select_q2]].y <<" "<< margin_b->points[plane_cal_w2[select_q2]].z << endl;*/
					}
				}
				if (select_q2_index.size() == 0)
				{
					continue;
				}
				//角度与距离差的判断用于鉴别是否为种子对
				vector<pair<float, pair<int, int>>> disdiff_with_pair;
				for (int selected_ang = 0; selected_ang < select_q2_index.size(); selected_ang++)
				{
					float data[3];
					data[0] = margin_b->points[select_q2_index[selected_ang]].x;
					data[1] = margin_b->points[select_q2_index[selected_ang]].y;
					data[2] = margin_b->points[select_q2_index[selected_ang]].z;
					Vector3f p2q2_direction;
					p2q2_direction(0) = data[0] - select_p2_xyz(0);
					p2q2_direction(1) = data[1] - select_p2_xyz(1);
					p2q2_direction(2) = data[2] - select_p2_xyz(2);
					float ang_p1q1_p2q2 = getAngle3D(vector_p1q1, p2q2_direction, false);
					if (ang_p1q1_p2q2 > M_PI_2)
					{
						ang_p1q1_p2q2 = ang_p1q1_p2q2 - M_PI_2;
					}
					if (ang_p1q1_p2q2 < angel_)
					{
						pair<int, int> p2q2;
						p2q2.first = plane_cal_w1[select_p2];
						p2q2.second = select_q2_index[selected_ang];
						float dis_p1q1 = getPairDis(curr_seed);
						float dis_p2q2 = getPairDis(p2q2);
						float dis_diff = abs(dis_p2q2 - dis_p1q1);
						disdiff_with_pair.push_back(make_pair(dis_diff, p2q2));
					}
				}
				if (disdiff_with_pair.size() == 0)
				{
					continue;
				}
				sort(disdiff_with_pair.begin(), disdiff_with_pair.end());
				if (disdiff_with_pair[0].first > 0.0004)
				{
					continue;
				}
				seeds.push(disdiff_with_pair[0].second);
				point_labels_a_[disdiff_with_pair[0].second.first] = segment_number;
				point_labels_b_[disdiff_with_pair[0].second.second] = segment_number;
				num_pts_in_segment++;
				segment_pairs.push_back(disdiff_with_pair[0].second);
			}
			if (ang_p1p2 <= angle_with_direction_)
			{
				continue;
			}
		}

	}
	return (num_pts_in_segment);

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> int
thickness::Thickness<PointT>::growRegionSelf(std::pair<int, int> seed, int segment_number, std::vector<std::pair<int, int>>& segment_pairs, std::vector<std::vector<float> >& geodesic_distances)
{
	std::queue<pair<int, int>> seeds;
	seeds.push(seed);
	point_labels_c_[seed.first] = segment_number;
	point_labels_c_[seed.second] = segment_number;
	int num_pts_in_segment = 1;
	segment_pairs.push_back(seed);
	while (!seeds.empty())
	{
		std::pair<int, int> curr_seed;
		curr_seed = seeds.front();
		seeds.pop();
		Vector3f point_p1, point_q1, vector_p1q1, vector_q1p1, p1_direction;
		point_p1(0) = margin_c->points[curr_seed.first].x;
		point_p1(1) = margin_c->points[curr_seed.first].y;
		point_p1(2) = margin_c->points[curr_seed.first].z;
		point_q1(0) = margin_c->points[curr_seed.second].x;
		point_q1(1) = margin_c->points[curr_seed.second].y;
		point_q1(2) = margin_c->points[curr_seed.second].z;
		vector_p1q1 = point_q1 - point_p1;
		vector_q1p1 = point_p1 - point_q1;
		Vector4f plane_p1q1 = getPlane(point_p1, vector_p1q1);
		Vector4f plane_q1p1 = getPlane(point_q1, vector_q1p1);
		p1_direction(0) = (*point_direction_c_)[curr_seed.first](0);
		p1_direction(1) = (*point_direction_c_)[curr_seed.first](1);
		p1_direction(2) = (*point_direction_c_)[curr_seed.first](2);
		vector <int> plane_cal_w1, plane_cal_w2;//经过到初始pair组成平面距离过滤的点的索引
		for (int select_with_planep1 = 0; select_with_planep1 < point_neighbours_c_[curr_seed.first].size(); select_with_planep1++)
		{
			if (point_labels_c_[point_neighbours_c_[curr_seed.first][select_with_planep1]] != -1)
			{
				continue;
			}
			float data[3];
			data[0] = margin_c->points[point_neighbours_c_[curr_seed.first][select_with_planep1]].x;
			data[1] = margin_c->points[point_neighbours_c_[curr_seed.first][select_with_planep1]].y;
			data[2] = margin_c->points[point_neighbours_c_[curr_seed.first][select_with_planep1]].z;
			float dis_to_plane = abs(plane_p1q1(0) * data[0] + plane_p1q1(1) * data[1] + plane_p1q1(2) * data[2] + plane_p1q1(3)) / sqrt(plane_p1q1(0)\
				* plane_p1q1(0) + plane_p1q1(1) * plane_p1q1(1) + plane_p1q1(2) * plane_p1q1(2));
			if (dis_to_plane < dis_threshold_)
			{
				plane_cal_w1.push_back(point_neighbours_c_[curr_seed.first][select_with_planep1]);
			}
		}
		if (plane_cal_w1.size() == 0)
		{
			continue;
		}
		for (int select_with_planep2 = 0; select_with_planep2 < point_neighbours_c_[curr_seed.second].size(); select_with_planep2++)
		{
			if (point_labels_c_[point_neighbours_c_[curr_seed.second][select_with_planep2]] != -1)
			{
				continue;
			}
			float data1[3];
			data1[0] = margin_c->points[point_neighbours_c_[curr_seed.second][select_with_planep2]].x;
			data1[1] = margin_c->points[point_neighbours_c_[curr_seed.second][select_with_planep2]].y;
			data1[2] = margin_c->points[point_neighbours_c_[curr_seed.second][select_with_planep2]].z;
			float dis_to_plane = abs(plane_q1p1(0) * data1[0] + plane_q1p1(1) * data1[1] + plane_q1p1(2) * data1[2] + plane_q1p1(3)) / sqrt(plane_q1p1(0)\
				* plane_q1p1(0) + plane_q1p1(1) * plane_q1p1(1) + plane_q1p1(2) * plane_q1p1(2));
			if (dis_to_plane < dis_threshold_)
			{
				plane_cal_w2.push_back(point_neighbours_c_[curr_seed.second][select_with_planep2]);
			}
		}
		if (plane_cal_w2.size() == 0)
		{
			continue;
		}
		//进行w1w2子集点中的厚度点对搜索
		for (int select_p2 = 0; select_p2 < plane_cal_w1.size(); select_p2++)
		{
			Vector3f select_p2_direction;
			select_p2_direction(0) = (*point_direction_c_)[plane_cal_w1[select_p2]](0);
			select_p2_direction(1) = (*point_direction_c_)[plane_cal_w1[select_p2]](1);
			select_p2_direction(2) = (*point_direction_c_)[plane_cal_w1[select_p2]](2);
			float ang_p1p2 = getAngle3D(p1_direction, select_p2_direction, false);
			if (ang_p1p2 > M_PI_2)
			{
				ang_p1p2 = ang_p1p2 - M_PI_2;
			}
			vector <int> select_q2_index;
			if (ang_p1p2 < angle_with_direction_)//如果p2的选择点的方向与p1的方向相似
			{
				Vector3f select_p2_xyz;
				select_p2_xyz(0) = margin_c->points[plane_cal_w1[select_p2]].x;
				select_p2_xyz(1) = margin_c->points[plane_cal_w1[select_p2]].y;
				select_p2_xyz(2) = margin_c->points[plane_cal_w1[select_p2]].z;
				Vector4f select_p2_plane = getPlane(select_p2_xyz, select_p2_direction);
				for (int select_q2 = 0; select_q2 < plane_cal_w2.size(); select_q2++)
				{
					if (point_labels_c_[plane_cal_w2[select_q2]] != -1)
					{
						continue;
					}
					float data[3];
					data[0] = margin_c->points[plane_cal_w2[select_q2]].x;
					data[1] = margin_c->points[plane_cal_w2[select_q2]].y;
					data[2] = margin_c->points[plane_cal_w2[select_q2]].z;
					float dis_to_plane = abs(select_p2_plane(0) * data[0] + select_p2_plane(1) * data[1] + select_p2_plane(2) * data[2] + select_p2_plane(3))\
						/ sqrt(select_p2_plane(0) * select_p2_plane(0) + select_p2_plane(1) * select_p2_plane(1) + select_p2_plane(2) * select_p2_plane(2));
					float geodesic_distances_p2q2 = geodesic_distances[plane_cal_w1[select_p2]][plane_cal_w2[select_q2]];
					if (dis_to_plane < dis_threshold_ && geodesic_distances_p2q2>geodesic_distances_threshold_)
					{
						select_q2_index.push_back(plane_cal_w2[select_q2]);
					}
				}
				if (select_q2_index.size() == 0)
				{
					continue;
				}
				//角度与距离差的判断用于鉴别是否为种子对
				vector<pair<float, pair<int, int>>> disdiff_with_pair;
				for (int selected_ang = 0; selected_ang < select_q2_index.size(); selected_ang++)
				{
					float data[3];
					data[0] = margin_c->points[select_q2_index[selected_ang]].x;
					data[1] = margin_c->points[select_q2_index[selected_ang]].y;
					data[2] = margin_c->points[select_q2_index[selected_ang]].z;
					Vector3f p2q2_direction;
					p2q2_direction(0) = data[0] - select_p2_xyz(0);
					p2q2_direction(1) = data[1] - select_p2_xyz(1);
					p2q2_direction(2) = data[2] - select_p2_xyz(2);
					float ang_p1q1_p2q2 = getAngle3D(vector_p1q1, p2q2_direction, false);
					if (ang_p1q1_p2q2 > M_PI_2)
					{
						ang_p1q1_p2q2 = ang_p1q1_p2q2 - M_PI_2;
					}
					if (ang_p1q1_p2q2 < angel_)
					{
						pair<int, int> p2q2;
						p2q2.first = plane_cal_w1[select_p2];
						p2q2.second = select_q2_index[selected_ang];
						float dis_p1q1 = getPairDisSelf(curr_seed);
						float dis_p2q2 = getPairDisSelf(p2q2);
						float dis_diff = abs(dis_p2q2 - dis_p1q1);
						disdiff_with_pair.push_back(make_pair(dis_diff, p2q2));
					}
				}
				if (disdiff_with_pair.size() == 0)
				{
					continue;
				}
				sort(disdiff_with_pair.begin(), disdiff_with_pair.end());
				if (disdiff_with_pair[0].first > 0.0004)
				{
					continue;
				}
				seeds.push(disdiff_with_pair[0].second);
				point_labels_c_[disdiff_with_pair[0].second.first] = segment_number;
				point_labels_c_[disdiff_with_pair[0].second.second] = segment_number;
				num_pts_in_segment++;
				segment_pairs.push_back(disdiff_with_pair[0].second);
			}
			if (ang_p1p2 <= angle_with_direction_)
			{
				continue;
			}
		}
	}
	return (num_pts_in_segment);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> int
thickness::Thickness<PointT>::growRegionSelf1(std::pair<int, int> seed, int segment_number, std::vector<std::pair<int, int>>& segment_pairs)
{
	std::queue<pair<int, int>> seeds;
	seeds.push(seed);
	point_labels_c_[seed.first] = segment_number;
	point_labels_c_[seed.second] = segment_number;
	int num_pts_in_segment = 1;
	segment_pairs.push_back(seed);
	while (!seeds.empty())
	{
		std::pair<int, int> curr_seed;
		curr_seed = seeds.front();
		seeds.pop();
		Vector3f point_p1, point_q1, vector_p1q1, vector_q1p1, p1_direction;
		point_p1(0) = margin_c->points[curr_seed.first].x;
		point_p1(1) = margin_c->points[curr_seed.first].y;
		point_p1(2) = margin_c->points[curr_seed.first].z;
		point_q1(0) = margin_c->points[curr_seed.second].x;
		point_q1(1) = margin_c->points[curr_seed.second].y;
		point_q1(2) = margin_c->points[curr_seed.second].z;
		vector_p1q1 = point_q1 - point_p1;
		vector_q1p1 = point_p1 - point_q1;
		Vector4f plane_p1q1 = getPlane(point_p1, vector_p1q1);
		Vector4f plane_q1p1 = getPlane(point_q1, vector_q1p1);
		p1_direction(0) = (*point_direction_c_)[curr_seed.first](0);
		p1_direction(1) = (*point_direction_c_)[curr_seed.first](1);
		p1_direction(2) = (*point_direction_c_)[curr_seed.first](2);
		vector <int> plane_cal_w1, plane_cal_w2;//经过到初始pair组成平面距离过滤的点的索引
		for (int select_with_planep1 = 0; select_with_planep1 < point_neighbours_c_[curr_seed.first].size(); select_with_planep1++)
		{
			if (point_labels_c_[point_neighbours_c_[curr_seed.first][select_with_planep1]] != -1)
			{
				continue;
			}
			float data[3];
			data[0] = margin_c->points[point_neighbours_c_[curr_seed.first][select_with_planep1]].x;
			data[1] = margin_c->points[point_neighbours_c_[curr_seed.first][select_with_planep1]].y;
			data[2] = margin_c->points[point_neighbours_c_[curr_seed.first][select_with_planep1]].z;
			float dis_to_plane = abs(plane_p1q1(0) * data[0] + plane_p1q1(1) * data[1] + plane_p1q1(2) * data[2] + plane_p1q1(3)) / sqrt(plane_p1q1(0)\
				* plane_p1q1(0) + plane_p1q1(1) * plane_p1q1(1) + plane_p1q1(2) * plane_p1q1(2));
			if (dis_to_plane < dis_threshold_)
			{
				plane_cal_w1.push_back(point_neighbours_c_[curr_seed.first][select_with_planep1]);
			}
		}
		if (plane_cal_w1.size() == 0)
		{
			continue;
		}
		for (int select_with_planep2 = 0; select_with_planep2 < point_neighbours_c_[curr_seed.second].size(); select_with_planep2++)
		{
			if (point_labels_c_[point_neighbours_c_[curr_seed.second][select_with_planep2]] != -1)
			{
				continue;
			}
			float data1[3];
			data1[0] = margin_c->points[point_neighbours_c_[curr_seed.second][select_with_planep2]].x;
			data1[1] = margin_c->points[point_neighbours_c_[curr_seed.second][select_with_planep2]].y;
			data1[2] = margin_c->points[point_neighbours_c_[curr_seed.second][select_with_planep2]].z;
			float dis_to_plane = abs(plane_q1p1(0) * data1[0] + plane_q1p1(1) * data1[1] + plane_q1p1(2) * data1[2] + plane_q1p1(3)) / sqrt(plane_q1p1(0)\
				* plane_q1p1(0) + plane_q1p1(1) * plane_q1p1(1) + plane_q1p1(2) * plane_q1p1(2));
			if (dis_to_plane < dis_threshold_)
			{
				plane_cal_w2.push_back(point_neighbours_c_[curr_seed.second][select_with_planep2]);
			}
		}
		if (plane_cal_w2.size() == 0)
		{
			continue;
		}
		//进行w1w2子集点中的厚度点对搜索
		for (int select_p2 = 0; select_p2 < plane_cal_w1.size(); select_p2++)
		{
			Vector3f select_p2_direction;
			select_p2_direction(0) = (*point_direction_c_)[plane_cal_w1[select_p2]](0);
			select_p2_direction(1) = (*point_direction_c_)[plane_cal_w1[select_p2]](1);
			select_p2_direction(2) = (*point_direction_c_)[plane_cal_w1[select_p2]](2);
			float ang_p1p2 = getAngle3D(p1_direction, select_p2_direction, false);
			if (ang_p1p2 > M_PI_2)
			{
				ang_p1p2 = ang_p1p2 - M_PI_2;
			}
			vector <int> select_q2_index;
			if (ang_p1p2 < angle_with_direction_)//如果p2的选择点的方向与p1的方向相似
			{
				Vector3f select_p2_xyz, direction_p2q2_candidate;
				select_p2_xyz(0) = margin_c->points[plane_cal_w1[select_p2]].x;
				select_p2_xyz(1) = margin_c->points[plane_cal_w1[select_p2]].y;
				select_p2_xyz(2) = margin_c->points[plane_cal_w1[select_p2]].z;
				Vector4f select_p2_plane = getPlane(select_p2_xyz, select_p2_direction);
				for (int select_q2 = 0; select_q2 < plane_cal_w2.size(); select_q2++)
				{
					if (point_labels_c_[plane_cal_w2[select_q2]] != -1)
					{
						continue;
					}
					float data[3];
					data[0] = margin_c->points[plane_cal_w2[select_q2]].x;
					data[1] = margin_c->points[plane_cal_w2[select_q2]].y;
					data[2] = margin_c->points[plane_cal_w2[select_q2]].z;
					float dis_to_plane = abs(select_p2_plane(0) * data[0] + select_p2_plane(1) * data[1] + select_p2_plane(2) * data[2] + select_p2_plane(3))\
						/ sqrt(select_p2_plane(0) * select_p2_plane(0) + select_p2_plane(1) * select_p2_plane(1) + select_p2_plane(2) * select_p2_plane(2));

					//求p1q1和p1的向量夹角
					Vector3f select_p2_direction_2d;
					direction_p2q2_candidate(0) = select_p2_xyz(0) - data[0];
					direction_p2q2_candidate(1) = select_p2_xyz(1) - data[1];
					direction_p2q2_candidate(2) = 0;
					select_p2_direction_2d(0) = select_p2_direction(0);
					select_p2_direction_2d(1) = select_p2_direction(1);
					select_p2_direction_2d(2) = 0;
					float ang_p2q2_p2 = getAngle3D(direction_p2q2_candidate, select_p2_direction_2d, false);
					if (dis_to_plane < dis_threshold_)
					{
						if (ang_p2q2_p2 > 20.0f / 180.0f * static_cast<float> (M_PI) && ang_p2q2_p2 < 160.0f / 180.0f * static_cast<float> (M_PI))
						{
							select_q2_index.push_back(plane_cal_w2[select_q2]);
						}
						
					}
					/*float geodesic_distances_p2q2 = geodesic_distances[plane_cal_w1[select_p2]][plane_cal_w2[select_q2]];
					if (dis_to_plane < dis_threshold_ && geodesic_distances_p2q2>geodesic_distances_threshold_)
					{
						select_q2_index.push_back(plane_cal_w2[select_q2]);
					}*/
				}
				if (select_q2_index.size() == 0)
				{
					continue;
				}
				//角度与距离差的判断用于鉴别是否为种子对
				vector<pair<float, pair<int, int>>> disdiff_with_pair;
				for (int selected_ang = 0; selected_ang < select_q2_index.size(); selected_ang++)
				{
					float data[3];
					data[0] = margin_c->points[select_q2_index[selected_ang]].x;
					data[1] = margin_c->points[select_q2_index[selected_ang]].y;
					data[2] = margin_c->points[select_q2_index[selected_ang]].z;
					Vector3f p2q2_direction;
					p2q2_direction(0) = data[0] - select_p2_xyz(0);
					p2q2_direction(1) = data[1] - select_p2_xyz(1);
					p2q2_direction(2) = data[2] - select_p2_xyz(2);
					float ang_p1q1_p2q2 = getAngle3D(vector_p1q1, p2q2_direction, false);
					if (ang_p1q1_p2q2 > M_PI_2)
					{
						ang_p1q1_p2q2 = ang_p1q1_p2q2 - M_PI_2;
					}
					if (ang_p1q1_p2q2 < angel_)
					{
						pair<int, int> p2q2;
						p2q2.first = plane_cal_w1[select_p2];
						p2q2.second = select_q2_index[selected_ang];
						float dis_p1q1 = getPairDisSelf(curr_seed);
						float dis_p2q2 = getPairDisSelf(p2q2);
						float dis_diff = abs(dis_p2q2 - dis_p1q1);
						disdiff_with_pair.push_back(make_pair(dis_diff, p2q2));
					}
				}
				if (disdiff_with_pair.size() == 0)
				{
					continue;
				}
				sort(disdiff_with_pair.begin(), disdiff_with_pair.end());
				if (disdiff_with_pair[0].first > 0.0004)
				{
					continue;
				}
				seeds.push(disdiff_with_pair[0].second);
				point_labels_c_[disdiff_with_pair[0].second.first] = segment_number;
				point_labels_c_[disdiff_with_pair[0].second.second] = segment_number;
				num_pts_in_segment++;
				segment_pairs.push_back(disdiff_with_pair[0].second);
			}
			if (ang_p1p2 <= angle_with_direction_)
			{
				continue;
			}
		}
	}
	return (num_pts_in_segment);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PoinT> void
thickness::Thickness<PoinT>::thicknessCal_1(std::vector<thicknessIndex>& thickness_index_arg, std::vector<std::pair<float, int>>& thickness_result, int& detection_type, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
	if (detection_type == 1)
	{
		for (int thickness_index = 0; thickness_index < static_cast<int>(thickness_index_arg.size()); thickness_index++)
		{
			int num = thickness_index_arg[thickness_index].size();
			float thickness_res_sum = 0;
			thicknessIndex one_thickness = thickness_index_arg[thickness_index];
			for (int index = 0; index < static_cast<int>(one_thickness.size()); index++)
			{
				pair<int, int> one_pair;
				one_pair.first = one_thickness[index].first;
				one_pair.second = one_thickness[index].second;
				float cal_res_one = getPairDis(one_pair);
				thickness_res_sum += cal_res_one;
			}
			float avg_thickness_res = thickness_res_sum / static_cast<float>(num);
			thickness_result_.push_back(make_pair(avg_thickness_res, 2 * num));
		}
		thickness_result = this->thickness_result_;
	}
	else if (detection_type == 2)
	{
		for (int thickness_index = 0; thickness_index < static_cast<int>(thickness_index_arg.size()); thickness_index++)
		{
			int num = thickness_index_arg[thickness_index].size();
			float thickness_res_sum = 0;
			thicknessIndex one_thickness = thickness_index_arg[thickness_index];
			for (int index = 0; index < static_cast<int>(one_thickness.size()); index++)
			{
				pair<int, int> one_pair;
				one_pair.first = one_thickness[index].first;
				one_pair.second = one_thickness[index].second;
				float cal_res_one = getPairDisSelf(one_pair);
				thickness_res_sum += cal_res_one;
			}
			float avg_thickness_res = thickness_res_sum / static_cast<float>(num);
			thickness_result_.push_back(make_pair(avg_thickness_res, 2 * num));
		}
		thickness_result = this->thickness_result_;
	}
	else if (detection_type == 3)//3表示对都是深度突变的厚度进行投影平面后的计算
	{
		pcl::search::OrganizedNeighbor<PointXYZRGB>::Ptr tree(new search::OrganizedNeighbor<PointXYZRGB>());
		tree->setInputCloud(cloud);
		std::vector<int> pointIdxRadiusSearch;
		std::vector<float> pointRadiusSquaredDistance;
		for (int thickness_index = 0; thickness_index < static_cast<int>(thickness_index_arg.size()); thickness_index++)
		{
			int num = thickness_index_arg[thickness_index].size();
			float thickness_res_sum = 0;
			thicknessIndex one_thickness = thickness_index_arg[thickness_index];
			for (int index = 0; index < static_cast<int>(one_thickness.size()); index++)
			{
				pointIdxRadiusSearch.clear();
				pointRadiusSquaredDistance.clear();
				pcl::PointXYZ p1, p2;
				pcl::PointXYZRGB mid_point;
				float real_dis;
				p1.x = margin_c->points[one_thickness[index].first].x;
				p1.y = margin_c->points[one_thickness[index].first].y;
				p1.z = margin_c->points[one_thickness[index].first].z;
				p2.x = margin_c->points[one_thickness[index].second].x;
				p2.y = margin_c->points[one_thickness[index].second].y;
				p2.z = margin_c->points[one_thickness[index].second].z;
				mid_point.x = (p1.x + p2.x) / 2.0;
				mid_point.y = (p1.y + p2.y) / 2.0;
				mid_point.z = (p1.z + p2.z) / 2.0;
				mid_point.r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
				mid_point.g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
				mid_point.b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
				float temp_dis= pcl::euclideanDistance(p1, p2);
				Eigen::Vector4f centroid;                    // 质心
				Eigen::Matrix3f covariance_matrix;           // 协方差矩阵
				pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZRGB>);
				cloud_in->clear();
				tree->radiusSearch(mid_point, temp_dis * 0.4, pointIdxRadiusSearch, pointRadiusSquaredDistance);
				for (auto i : pointIdxRadiusSearch)
				{
					cloud_in->push_back(cloud->points[i]);
				}

				stringstream ss1;
				ss1 << "测试" << ".pcd";
				pcl::PCDWriter writer;
				writer.write<pcl::PointXYZRGB>(ss1.str(), *cloud_in, false);

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
				double dis_p1_plane = pcl::pointToPlaneDistance(p1, a, b, c, d);
				double dis_p2_plane = pcl::pointToPlaneDistance(p2, a, b, c, d);
				if (dis_p1_plane > dis_p2_plane)//投影到平面
				{
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
				}
				else
				{
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
				}
				thickness_res_sum += real_dis;
			}
			float avg_thickness_res = thickness_res_sum / static_cast<float>(num);
			thickness_result_.push_back(make_pair(avg_thickness_res, 2 * num));
		}
		thickness_result = this->thickness_result_;
	}
	else if (detection_type == 4)//自己的改正方法
	{
	   pcl::search::OrganizedNeighbor<PointXYZRGB>::Ptr tree(new search::OrganizedNeighbor<PointXYZRGB>());
	   tree->setInputCloud(cloud);
	   std::vector<int> pointIdxRadiusSearch;
	   std::vector<float> pointRadiusSquaredDistance;
	   for (int thickness_index = 0; thickness_index < static_cast<int>(thickness_index_arg.size()); thickness_index++)
	   {
		   int num = thickness_index_arg[thickness_index].size();
		   float thickness_res_sum = 0;
		   thicknessIndex one_thickness = thickness_index_arg[thickness_index];
		   int num_p1_real=0;//统计p1z坐标大于p2的点个数，
		   for (int index = 0; index < static_cast<int>(one_thickness.size()); index++)
		   {
			   pcl::PointXYZ p1, p2;
			   p1.x = margin_c->points[one_thickness[index].first].x;
			   p1.y = margin_c->points[one_thickness[index].first].y;
			   p1.z = margin_c->points[one_thickness[index].first].z;
			   p2.x = margin_c->points[one_thickness[index].second].x;
			   p2.y = margin_c->points[one_thickness[index].second].y;
			   p2.z = margin_c->points[one_thickness[index].second].z;
			   if (p1.z < p2.z)
			   {
				   num_p1_real++;
			   }
		   }
		   if (num_p1_real > num * 0.6)//如果p1大部分都小于p2证明要改正全部p2点
		   {
			   for (int index = 0; index < static_cast<int>(one_thickness.size()); index++)
			   {
				   pointIdxRadiusSearch.clear();
				   pointRadiusSquaredDistance.clear();
				   pcl::PointXYZRGB p1, p2;
				   float real_dis;
				   p1.x = margin_c->points[one_thickness[index].first].x;
				   p1.y = margin_c->points[one_thickness[index].first].y;
				   p1.z = margin_c->points[one_thickness[index].first].z;
				   p2.x = margin_c->points[one_thickness[index].second].x;
				   p2.y = margin_c->points[one_thickness[index].second].y;
				   p2.z = margin_c->points[one_thickness[index].second].z;
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
				   stringstream ss1;
				   ss1 << "测试" << ".pcd";
				   pcl::PCDWriter writer;
				   writer.write<pcl::PointXYZRGB>(ss1.str(), *cloud_in, false);

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
				   thickness_res_sum += real_dis;
			   }
			   
		   }
		   else
		   {
			   for (int index = 0; index < static_cast<int>(one_thickness.size()); index++)
			   {
				   pointIdxRadiusSearch.clear();
				   pointRadiusSquaredDistance.clear();
				   pcl::PointXYZRGB p1, p2;
				   float real_dis;
				   p1.x = margin_c->points[one_thickness[index].first].x;
				   p1.y = margin_c->points[one_thickness[index].first].y;
				   p1.z = margin_c->points[one_thickness[index].first].z;
				   p2.x = margin_c->points[one_thickness[index].second].x;
				   p2.y = margin_c->points[one_thickness[index].second].y;
				   p2.z = margin_c->points[one_thickness[index].second].z;
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
				   stringstream ss1;
				   ss1 << "测试" << ".pcd";
				   pcl::PCDWriter writer;
				   writer.write<pcl::PointXYZRGB>(ss1.str(), *cloud_in, false);

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
				   thickness_res_sum += real_dis;
			   }
		   }
		   float avg_thickness_res = thickness_res_sum / static_cast<float>(num);
		   thickness_result_.push_back(make_pair(avg_thickness_res, 2 * num));
	   }
	   thickness_result = this->thickness_result_;
    }
	else if (detection_type == 5)
	{
	  pcl::search::OrganizedNeighbor<PointXYZRGB>::Ptr tree(new search::OrganizedNeighbor<PointXYZRGB>());
	  tree->setInputCloud(cloud);
	  std::vector<int> pointIdxRadiusSearch;
	  std::vector<float> pointRadiusSquaredDistance;
	  for (int thickness_index = 0; thickness_index < static_cast<int>(thickness_index_arg.size()); thickness_index++)
	  {
		  int num = thickness_index_arg[thickness_index].size();
		  float thickness_res_sum = 0;
		  float avg_thickness_res;//初始平均厚度
		  float real_res_sum = 0;
		  float avg_real_res;//真实平均厚度
		  float sum_p1_plane = 0.0;
		  float sum_p2_plane = 0.0;
		  float avg_p1_plane, avg_p2_plane;
		  thicknessIndex one_thickness = thickness_index_arg[thickness_index];
		  //先计算原始的平均厚度
		  for (int index = 0; index < static_cast<int>(one_thickness.size()); index++)
		  {
			  pair<int, int> one_pair;
			  one_pair.first = one_thickness[index].first;
			  one_pair.second = one_thickness[index].second;
			  float cal_res_one = getPairDisSelf(one_pair);
			  thickness_res_sum += cal_res_one;
		  }
		  avg_thickness_res = thickness_res_sum / static_cast<float>(num);//初始平均厚度
		  for (int index = 0; index < static_cast<int>(one_thickness.size()); index++)
		  {
			  pointIdxRadiusSearch.clear();
			  pointRadiusSquaredDistance.clear();
			  pcl::PointXYZ p1, p2;
			  pcl::PointXYZRGB mid_point;
			  float real_dis;
			  p1.x = margin_c->points[one_thickness[index].first].x;
			  p1.y = margin_c->points[one_thickness[index].first].y;
			  p1.z = margin_c->points[one_thickness[index].first].z;
			  p2.x = margin_c->points[one_thickness[index].second].x;
			  p2.y = margin_c->points[one_thickness[index].second].y;
			  p2.z = margin_c->points[one_thickness[index].second].z;
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
			  // 获取平面方程：AX+BY+CZ+D = 0的系数
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
			  if (avg_p1_plane > avg_thickness_res * 0.2)//参数
			  {
				  for (int index = 0; index < static_cast<int>(one_thickness.size()); index++)
				  {
					  pointIdxRadiusSearch.clear();
					  pointRadiusSquaredDistance.clear();
					  pcl::PointXYZRGB p1, p2;
					  float real_dis;
					  p1.x = margin_c->points[one_thickness[index].first].x;
					  p1.y = margin_c->points[one_thickness[index].first].y;
					  p1.z = margin_c->points[one_thickness[index].first].z;
					  p2.x = margin_c->points[one_thickness[index].second].x;
					  p2.y = margin_c->points[one_thickness[index].second].y;
					  p2.z = margin_c->points[one_thickness[index].second].z;
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
				  for (int index = 0; index < static_cast<int>(one_thickness.size()); index++)
				  {
					  pointIdxRadiusSearch.clear();
					  pointRadiusSquaredDistance.clear();
					  pcl::PointXYZRGB p1, p2;
					  float real_dis;
					  p1.x = margin_c->points[one_thickness[index].first].x;
					  p1.y = margin_c->points[one_thickness[index].first].y;
					  p1.z = margin_c->points[one_thickness[index].first].z;
					  p2.x = margin_c->points[one_thickness[index].second].x;
					  p2.y = margin_c->points[one_thickness[index].second].y;
					  p2.z = margin_c->points[one_thickness[index].second].z;
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
					  /*stringstream ss1;
					  ss1 << "测试" << ".pcd";
					  pcl::PCDWriter writer;
					  writer.write<pcl::PointXYZRGB>(ss1.str(), *cloud_in, false);*/

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
		  else{ avg_real_res = avg_thickness_res; }
		  thickness_result_.push_back(make_pair(avg_real_res, 2 * num));
	  }
	  thickness_result = this->thickness_result_;
    }
	else
	{
		cout << "重新输入检测类型" << endl;
	}

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PoinT> void
thickness::Thickness<PoinT>::thicknessCal(std::vector<thicknessIndex>& thickness_index_arg, std::vector<thicknessIndex>& thickness_index_std, std::vector<std::pair<float, int>>& thickness_result, std::vector<float>& std_result, int& detection_type)
{
	if (detection_type == 1)
	{
		for (int thickness_index = 0; thickness_index < static_cast<int>(thickness_index_arg.size()); thickness_index++)
		{
			int num = thickness_index_arg[thickness_index].size();
			float thickness_res_sum = 0;
			thicknessIndex one_thickness = thickness_index_arg[thickness_index];
			for (int index = 0; index < static_cast<int>(one_thickness.size()); index++)
			{
				pair<int, int> one_pair;
				one_pair.first = one_thickness[index].first;
				one_pair.second = one_thickness[index].second;
				float cal_res_one = getPairDis(one_pair);
				thickness_res_sum += cal_res_one;
			}
			float avg_thickness_res = thickness_res_sum / static_cast<float>(num);
			float thickness_res_sum_std = 0;
			for (int index = 0; index < static_cast<int>(one_thickness.size()); index++)
			{
				pair<int, int> one_pair;
				one_pair.first = one_thickness[index].first;
				one_pair.second = one_thickness[index].second;
				float cal_res_one = getPairDis(one_pair);
				thickness_res_sum_std += (cal_res_one - avg_thickness_res) * (cal_res_one - avg_thickness_res);
			}
			float avg_thickness_res_std = thickness_res_sum_std / static_cast<float>(num);
			std_result_.push_back(avg_thickness_res_std);
		}
		std_result = this->std_result_;
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
		for (int thickness_index = 0; thickness_index < static_cast<int>(thickness_index_arg.size()); thickness_index++)
		{
			int num = thickness_index_arg[thickness_index].size();
			float thickness_res_sum = 0;
			thicknessIndex one_thickness = thickness_index_arg[thickness_index];
			for (int index = 0; index < static_cast<int>(one_thickness.size()); index++)
			{
				pair<int, int> one_pair;
				one_pair.first = one_thickness[index].first;
				one_pair.second = one_thickness[index].second;
				float cal_res_one = getPairDis(one_pair);
				thickness_res_sum += cal_res_one;
			}
			float avg_thickness_res = thickness_res_sum / static_cast<float>(num);
			float thickness_res_sum_std = 0;
			for (int index = 0; index < static_cast<int>(one_thickness.size()); index++)
			{
				pair<int, int> one_pair;
				one_pair.first = one_thickness[index].first;
				one_pair.second = one_thickness[index].second;
				float cal_res_one = getPairDis(one_pair);
				thickness_res_sum_std += (cal_res_one - avg_thickness_res) * (cal_res_one - avg_thickness_res);
			}
			float avg_thickness_res_std = thickness_res_sum_std / static_cast<float>(num);
			if (avg_thickness_res_std < threshold)
			{
				//std_result_.push_back(avg_thickness_res_std);
				thickness_index_std.push_back(one_thickness);
				thickness_result_.push_back(make_pair(avg_thickness_res, 2 * num));
			}
		}
		thickness_result = this->thickness_result_;
		//std_result = this->std_result_;
	}
	else if (detection_type == 2)
	{
		for (int thickness_index = 0; thickness_index < static_cast<int>(thickness_index_arg.size()); thickness_index++)
		{
			int num = thickness_index_arg[thickness_index].size();
			float thickness_res_sum = 0;
			thicknessIndex one_thickness = thickness_index_arg[thickness_index];
			for (int index = 0; index < static_cast<int>(one_thickness.size()); index++)
			{
				pair<int, int> one_pair;
				one_pair.first = one_thickness[index].first;
				one_pair.second = one_thickness[index].second;
				float cal_res_one = getPairDisSelf(one_pair);
				thickness_res_sum += cal_res_one;
			}
			float avg_thickness_res = thickness_res_sum / static_cast<float>(num);
			float thickness_res_sum_std = 0;
			for (int index = 0; index < static_cast<int>(one_thickness.size()); index++)
			{
				pair<int, int> one_pair;
				one_pair.first = one_thickness[index].first;
				one_pair.second = one_thickness[index].second;
				float cal_res_one = getPairDisSelf(one_pair);
				thickness_res_sum_std += (cal_res_one - avg_thickness_res) * (cal_res_one - avg_thickness_res);
			}
			float avg_thickness_res_std = thickness_res_sum_std / static_cast<float>(num);
			std_result_.push_back(avg_thickness_res_std);
		}
		std_result = this->std_result_;
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
			threshold = std_result[floor(std_result.size()/2.0)];
		}
		for (int thickness_index = 0; thickness_index < static_cast<int>(thickness_index_arg.size()); thickness_index++)
		{
			int num = thickness_index_arg[thickness_index].size();
			float thickness_res_sum = 0;
			thicknessIndex one_thickness = thickness_index_arg[thickness_index];
			for (int index = 0; index < static_cast<int>(one_thickness.size()); index++)
			{
				pair<int, int> one_pair;
				one_pair.first = one_thickness[index].first;
				one_pair.second = one_thickness[index].second;
				float cal_res_one = getPairDisSelf(one_pair);
				thickness_res_sum += cal_res_one;
			}
			float avg_thickness_res = thickness_res_sum / static_cast<float>(num);
			float thickness_res_sum_std = 0;
			for (int index = 0; index < static_cast<int>(one_thickness.size()); index++)
			{
				pair<int, int> one_pair;
				one_pair.first = one_thickness[index].first;
				one_pair.second = one_thickness[index].second;
				float cal_res_one = getPairDisSelf(one_pair);
				thickness_res_sum_std += (cal_res_one - avg_thickness_res) * (cal_res_one - avg_thickness_res);
			}
			float avg_thickness_res_std = thickness_res_sum_std / static_cast<float>(num);
			if (avg_thickness_res_std < threshold)
			{
				//std_result_.push_back(avg_thickness_res_std);
				thickness_index_std.push_back(one_thickness);
				thickness_result_.push_back(make_pair(avg_thickness_res, 2 * num));
			}
		}
		thickness_result = this->thickness_result_;
		//std_result = this->std_result_;
	}
	else
	{
		cout << "重新输入检测类型" << endl;
	}

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
	// Running K-Means Clustering
	int iters = 100;
	KMeans kmeans(K, iters);
	double threshold = kmeans.run(all_points);
	return threshold;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> Eigen::Vector4f
thickness::Thickness<PointT>::getPlane(Eigen::Vector3f point, Eigen::Vector3f direction)
{
	Vector4f conf;
	conf(0) = direction(0);
	conf(1) = direction(1);
	conf(2) = direction(2);
	conf(3) = -(direction(0) * point(0) + direction(1) * point(1) + direction(2) * point(2));
	return conf;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> float
thickness::Thickness<PointT>::getPairDis(pair<int, int> seed_pair)
{
	pcl::PointXYZ p1, p2;
	p1.x = margin_a->points[seed_pair.first].x;
	p1.y = margin_a->points[seed_pair.first].y;
	p1.z = margin_a->points[seed_pair.first].z;
	p2.x = margin_b->points[seed_pair.second].x;
	p2.y = margin_b->points[seed_pair.second].y;
	p2.z = margin_b->points[seed_pair.second].z;
	float dis = pcl::euclideanDistance(p1, p2);
	return dis;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> float
thickness::Thickness<PointT>::getPairDisSelf(pair<int, int> seed_pair)
{
	pcl::PointXYZ p1, p2;
	p1.x = margin_c->points[seed_pair.first].x;
	p1.y = margin_c->points[seed_pair.first].y;
	p1.z = margin_c->points[seed_pair.first].z;
	p2.x = margin_c->points[seed_pair.second].x;
	p2.y = margin_c->points[seed_pair.second].y;
	p2.z = margin_c->points[seed_pair.second].z;
	float dis = pcl::euclideanDistance(p1, p2);
	return dis;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//进行厚度的合并

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//计算单个厚度的厚度值


template class thickness::Thickness<PointXYZ>;
template class thickness::Cluster<PointXYZ>;

template<typename PointT> std::vector<std::vector<std::pair<int, int>>>
thickness::Thickness<PointT>::validation()
{
	search_->setInputCloud(original_cloud);  
	std::vector<int> pointIdxRadiusSearch;
	std::vector<float> pointRadiusSquaredDistance;
	pcl::PointXYZ midPoint;
	for (std::vector<std::vector<std::pair<int, int>>>::const_iterator it = thickness_index_.begin(); it != thickness_index_.end(); ++it)
	{
		int number_of_pairs = it->size();
		int total_have_neiborghtor = 0;
		for (std::vector<std::pair<int, int>>::const_iterator pit = it->begin(); pit != it->end(); ++pit)
		{
			pcl::PointXYZ p1, p2;
			p1.x = margin_c->points[(*pit).first].x;
			p1.y = margin_c->points[(*pit).first].y;
			p1.z = margin_c->points[(*pit).first].z;
			p2.x = margin_c->points[(*pit).second].x;
			p2.y = margin_c->points[(*pit).second].y;
			p2.z = margin_c->points[(*pit).second].z;
			midPoint.x = (double)(p1.x + p2.x) / 2.0;
			midPoint.y = (double)(p1.y + p2.y) / 2.0;
			midPoint.z = (double)(p1.z + p2.z) / 2.0;
			double distance = pcl::euclideanDistance(p1, p2);
			pointIdxRadiusSearch.clear();
			pointRadiusSquaredDistance.clear();
			search_->radiusSearch(midPoint, distance * 0.3, pointIdxRadiusSearch, pointRadiusSquaredDistance);
			double size = pointIdxRadiusSearch.size();
			if (size > 0)
			{
				total_have_neiborghtor++;
			}
		}
		double radio = (double)total_have_neiborghtor / (double)number_of_pairs;
		if (radio > 0.2)
		{
			thickness_index_validation.push_back(*it);
		}
	}
	return thickness_index_validation;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> std::vector<std::vector<std::pair<int, int>>>
	thickness::Thickness<PointT>::validation_fackThickness(std::vector<thicknessIndex> thickness_pair, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_cluster_curvature_all;
	std::vector<float> distance_all;
	std::vector<thicknessIndex> thickness_pair_new;
	for (std::vector<std::vector<std::pair<int, int>>>::const_iterator it = thickness_pair.begin(); it != thickness_pair.end(); ++it)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster_curvature(new pcl::PointCloud<pcl::PointXYZ>);
		float distance_mean = 0.0;
		float distance_sum = 0.0;
		for (std::vector<std::pair<int, int>>::const_iterator pit = it->begin(); pit != it->end(); ++pit)
		{
			pcl::PointXYZ tempa, tempb;
			tempb.x = margin_b->points[(*pit).second].x;
			tempb.y = margin_b->points[(*pit).second].y;
			tempb.z = margin_b->points[(*pit).second].z;
			cloud_cluster_curvature->push_back(tempb);
			tempa.x = margin_a->points[(*pit).first].x;
			tempa.y = margin_a->points[(*pit).first].y;
			tempa.z = margin_a->points[(*pit).first].z;
			distance_sum += pcl::euclideanDistance(tempa, tempb);
		}
		distance_mean = distance_sum / float(it->size());
		distance_all.push_back(distance_mean);
		cloud_cluster_curvature->width = cloud_cluster_curvature->points.size();
		cloud_cluster_curvature->height = 1;
		cloud_cluster_curvature->is_dense = true;
		cloud_cluster_curvature_all.push_back(cloud_cluster_curvature);
	}
	pcl::search::OrganizedNeighbor<PointXYZRGB>::Ptr tree(new search::OrganizedNeighbor<PointXYZRGB>());
	tree->setInputCloud(cloud);
	std::vector<int> pointIdxRadiusSearch;
	std::vector<float> pointRadiusSquaredDistance;
	std::vector<int> pointIdxRadiusSearch_new;
	for (int i = 0; i < cloud_cluster_curvature_all.size(); i++)
	{
		pointIdxRadiusSearch.clear();
		pointRadiusSquaredDistance.clear();
		pointIdxRadiusSearch_new.clear();
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
		pcl::PointXYZRGB midPoint;
		int order = int(cloud_cluster_curvature_all[i]->points.size() / 2.0);
		midPoint.x = cloud_cluster_curvature_all[i]->points[order].x;
		midPoint.y = cloud_cluster_curvature_all[i]->points[order].y;
		midPoint.z = cloud_cluster_curvature_all[i]->points[order].z;
		midPoint.r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		midPoint.g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		midPoint.b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		tree->radiusSearch(midPoint, distance_all[i] * 0.9, pointIdxRadiusSearch, pointRadiusSquaredDistance);
		int num = 0;//进行三次，有一次满足就加1
		num = fake_test(midPoint, cloud_cluster_curvature_all[i], pca, b, V2, distance_all[i], pointIdxRadiusSearch, pointRadiusSquaredDistance, pointIdxRadiusSearch_new, cloud, num);
		pcl::PointXYZRGB Point_0;
		int order_2 = 0;
		Point_0.x = cloud_cluster_curvature_all[i]->points[order_2].x;
		Point_0.y = cloud_cluster_curvature_all[i]->points[order_2].y;
		Point_0.z = cloud_cluster_curvature_all[i]->points[order_2].z;
		Point_0.r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Point_0.g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Point_0.b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		pointIdxRadiusSearch.clear();
		pointRadiusSquaredDistance.clear();
		pointIdxRadiusSearch_new.clear();
		tree->radiusSearch(Point_0, distance_all[i] * 0.9, pointIdxRadiusSearch, pointRadiusSquaredDistance);
		num = fake_test(Point_0, cloud_cluster_curvature_all[i], pca, b, V2, distance_all[i], pointIdxRadiusSearch, pointRadiusSquaredDistance, pointIdxRadiusSearch_new, cloud, num);
		pcl::PointXYZRGB Point_allsize;
		int order_3 = cloud_cluster_curvature_all[i]->points.size()-1;
		Point_allsize.x = cloud_cluster_curvature_all[i]->points[order_3].x;
		Point_allsize.y = cloud_cluster_curvature_all[i]->points[order_3].y;
		Point_allsize.z = cloud_cluster_curvature_all[i]->points[order_3].z;
		Point_allsize.r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Point_allsize.g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Point_allsize.b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		pointIdxRadiusSearch.clear();
		pointRadiusSquaredDistance.clear();
		pointIdxRadiusSearch_new.clear();
		tree->radiusSearch(Point_allsize, distance_all[i] * 0.9, pointIdxRadiusSearch, pointRadiusSquaredDistance);
		num = fake_test(Point_allsize, cloud_cluster_curvature_all[i], pca, b, V2, distance_all[i], pointIdxRadiusSearch, pointRadiusSquaredDistance, pointIdxRadiusSearch_new, cloud, num);
		if (num >= 1)
		{
			thickness_pair_new.push_back(thickness_pair[i]);
		}
	}
	return thickness_pair_new;
}
template<typename PointT> std::vector<std::pair<float, int>>
thickness::Thickness<PointT>::validation_fackAngle(std::vector<thicknessIndex> thickness_pair, std::vector<std::pair<float, int>>& thickness_result, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud)
{
	thickness_result.clear();
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_cluster_curvature_all;
	std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_cluster_depth_all;
	std::vector<float> distance_all;
	for (std::vector<std::vector<std::pair<int, int>>>::const_iterator it = thickness_pair.begin(); it != thickness_pair.end(); ++it)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster_curvature(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster_depth(new pcl::PointCloud<pcl::PointXYZ>);
		float distance_mean = 0.0;
		float distance_sum = 0.0;
		for (std::vector<std::pair<int, int>>::const_iterator pit = it->begin(); pit != it->end(); ++pit)
		{
			pcl::PointXYZ tempa, tempb;
			tempb.x = margin_b->points[(*pit).second].x;
			tempb.y = margin_b->points[(*pit).second].y;
			tempb.z = margin_b->points[(*pit).second].z;
			cloud_cluster_curvature->push_back(tempb);
			tempa.x = margin_a->points[(*pit).first].x;
			tempa.y = margin_a->points[(*pit).first].y;
			tempa.z = margin_a->points[(*pit).first].z;
			cloud_cluster_depth->push_back(tempa);
			distance_sum += pcl::euclideanDistance(tempa, tempb);
		}
		distance_mean = distance_sum / float(it->size());
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
			depthtoplane = fake_test_angle(num_cluster, all_thickness_one, all_thickness_two, midPoint, cloud_cluster_curvature_all[i], pca, b, V2, distance_all[i], pointIdxRadiusSearch, pointRadiusSquaredDistance, pointIdxRadiusSearch_new, cloud, depthPoint);
			if ((int)depthtoplane == 10000)//证明当前点欧式聚类只有一个就不用了
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

			//if (number > (cloud_cluster_curvature_all[i]->points.size() * 0.4))//如果采样的数量大于全部曲率突变点的0.4倍就结束循环
			//{
			//	break;
			//}

			j = j + 1;
			num_cluster = num_cluster + 1;
		}
		double sum_depthtoplane = 0.0;
		double avg_depthtoplane = 0.0;
		sort(depthtoplane_all.begin(), depthtoplane_all.end());
		int num_count = 0;
		int size_end = depthtoplane_all.size();
		//for (int q = 0; q < depthtoplane_all.size() ; q++)
		//{
		//	cout << "depthtoplane_all[q]: " << depthtoplane_all[q] << "\n";
		//}
		//for (int q = depthtoplane_all.size() -2; q < depthtoplane_all.size() - 1; q++)
		//{
		//	//cout << "depthtoplane_all[q]: " << depthtoplane_all[q] << "\n";
		//	sum_depthtoplane += depthtoplane_all[q];
		//	num_count = num_count + 1;
		//}
		
		//if (depthtoplane_all.size() <= 4)
		//{
		//	for (int q = 0; q < 1; q++)
		//	{
		//		sum_depthtoplane += depthtoplane_all[q];
		//		num_count = num_count + 1;
		//	}
		//}
		//avg_depthtoplane = sum_depthtoplane / (double)num_count;
		//cout << "avg_depthtoplane" << avg_depthtoplane << "\n";
	

		//for (int i = 0; i < depthtoplane_all.size(); i++)
		//{
		//	cout << depthtoplane_all[i] << "\n";
		//}
		if (depthtoplane_all.size() == 0)
		{
			avg_depthtoplane = 0;
		}

		//法1
		/*if (depthtoplane_all.size() != 0)
		{
			avg_depthtoplane = depthtoplane_all[depthtoplane_all.size() - 1];
		}*/
		//法2
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
	for (int thickness_index = 0; thickness_index < static_cast<int>(thickness_pair.size()); thickness_index++)
	{
		int num = thickness_pair[thickness_index].size();
		float thickness_res_sum = 0;
		thicknessIndex one_thickness = thickness_pair[thickness_index];
		for (int index = 0; index < static_cast<int>(one_thickness.size()); index++)
		{
			pair<int, int> one_pair;
			one_pair.first = one_thickness[index].first;
			one_pair.second = one_thickness[index].second;
			float cal_res_one = getPairDis(one_pair);
			thickness_res_sum += cal_res_one;
		}
		float avg_thickness_res = thickness_res_sum / static_cast<float>(num);
		double Truthvalue = 0.0;
		double Difference = abs(depthtoplane_all_avg[thickness_index] - avg_thickness_res);
		Truthvalue = avg_thickness_res + Difference * 2;
		thickness_result.push_back(make_pair(Truthvalue, 2 * num));
	}
	return thickness_result;
}
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
	double x1,x2,y1,y2,z1,z2;//求垂直于主方向直线用的（暂时没有用到）
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

	/*stringstream ss;
	ss << num_cluster << "_" << "max_num" << ".pcd";
	pcl::PCDWriter writer;
	writer.write<pcl::PointXYZ>(ss.str(), *cloud_filtered_all[max_num], false);
	stringstream ss1;
	ss1 << num_cluster << "_" << "second_num" << ".pcd";
	pcl::PCDWriter writer1;
	writer1.write<pcl::PointXYZ>(ss1.str(), *cloud_filtered_ex_first[second_num], false);

	*all_thickness_one += *cloud_filtered_all[max_num];
	*all_thickness_two += *cloud_filtered_ex_first[second_num];

	pcl::io::savePCDFileBinary("max_num.pcd", *all_thickness_one);
	pcl::io::savePCDFileBinary("second_num.pcd", *all_thickness_two);*/
	
	/*Eigen::Vector4f pcaCentroid_first;
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
	pcl::PointXYZ Point2;
	Point2.x = midPoint.x + x;
	Point2.y = midPoint.y + y;
	Point2.z = midPoint.z + z;*/

	//Vector3dNew O(midPoint.x, midPoint.y, midPoint.z);
	//Vector3dNew E(Point2.x, Point2.y, Point2.z);
	//Vector3dNew Center(midPoint.x, midPoint.y, midPoint.z);
	//double R = dis * 2.0;
	//vector<Vector3dNew> points;
	//LineIntersectSphere(O, E, Center, R, points);
	//vector<string> line_kmeans;
	//pcl::PointXYZ p1_standard, p2, endpoint;
	//p1_standard.x = cloud->points[pointIdxRadiusSearch_new[0]].x;
	//p1_standard.y = cloud->points[pointIdxRadiusSearch_new[0]].y;
	//p1_standard.z = cloud->points[pointIdxRadiusSearch_new[0]].z;
	//endpoint.x = cloud->points[pointIdxRadiusSearch_new[0]].x;
	//endpoint.y = cloud->points[pointIdxRadiusSearch_new[0]].y;
	//endpoint.z = cloud->points[pointIdxRadiusSearch_new[0]].z;
	//p2.x = points[0].x;
	//p2.y = points[0].y;
	//p2.z = points[0].z;
	//double standard = pcl::euclideanDistance(p1_standard, p2);
	//for (size_t t = 0; t < pointIdxRadiusSearch_new.size(); ++t)
	//{
	//	double disPlane = 0.0;
	//	pcl::PointXYZ p1;
	//	p1.x = cloud->points[pointIdxRadiusSearch_new[t]].x;
	//	p1.y = cloud->points[pointIdxRadiusSearch_new[t]].y;
	//	p1.z = cloud->points[pointIdxRadiusSearch_new[t]].z;
	//	disPlane = pcl::euclideanDistance(p1, p2);
	//	if (disPlane < standard)
	//	{
	//		endpoint.x = cloud->points[pointIdxRadiusSearch_new[t]].x;
	//		endpoint.y = cloud->points[pointIdxRadiusSearch_new[t]].y;
	//		endpoint.z = cloud->points[pointIdxRadiusSearch_new[t]].z;
	//		standard = disPlane;
	//	}
	//}
	//pcl::PointXYZ p1_standard_2, p2_2, endpoint_2;
	//p1_standard_2.x = cloud->points[pointIdxRadiusSearch_new[0]].x;
	//p1_standard_2.y = cloud->points[pointIdxRadiusSearch_new[0]].y;
	//p1_standard_2.z = cloud->points[pointIdxRadiusSearch_new[0]].z;
	//endpoint_2.x = cloud->points[pointIdxRadiusSearch_new[0]].x;
	//endpoint_2.y = cloud->points[pointIdxRadiusSearch_new[0]].y;
	//endpoint_2.z = cloud->points[pointIdxRadiusSearch_new[0]].z;
	//p2_2.x = points[1].x;
	//p2_2.y = points[1].y;
	//p2_2.z = points[1].z;
	//double standard_2 = pcl::euclideanDistance(p1_standard_2, p2_2);
	//for (size_t t = 0; t < pointIdxRadiusSearch_new.size(); ++t)
	//{
	//	double disPlane = 0.0;
	//	pcl::PointXYZ p1_2;
	//	p1_2.x = cloud->points[pointIdxRadiusSearch_new[t]].x;
	//	p1_2.y = cloud->points[pointIdxRadiusSearch_new[t]].y;
	//	p1_2.z = cloud->points[pointIdxRadiusSearch_new[t]].z;
	//	disPlane = pcl::euclideanDistance(p1_2, p2_2);
	//	if (disPlane < standard_2)
	//	{
	//		endpoint_2.x = cloud->points[pointIdxRadiusSearch_new[t]].x;
	//		endpoint_2.y = cloud->points[pointIdxRadiusSearch_new[t]].y;
	//		endpoint_2.z = cloud->points[pointIdxRadiusSearch_new[t]].z;
	//		standard_2 = disPlane;
	//	}
	//}
	//vector<pcl::PointXYZ> points_new;
	//points_new.push_back(endpoint);
	//points_new.push_back(endpoint_2);
	//for (auto it : points_new)
	//{
	//	string line_t;
	//	line_t = to_string(it.x);
	//	line_t.append(" ");
	//	line_t.append(to_string(it.y));
	//	line_t.append(" ");
	//	line_t.append(to_string(it.z));
	//	line_kmeans.push_back(line_t);
	//}
	//for (size_t t = 0; t < pointIdxRadiusSearch_new.size(); t++)
	//{
	//	string line_t;
	//	line_t = to_string(cloud->points[pointIdxRadiusSearch_new[t]].x);
	//	line_t.append(" ");
	//	line_t.append(to_string(cloud->points[pointIdxRadiusSearch_new[t]].y));
	//	line_t.append(" ");
	//	line_t.append(to_string(cloud->points[pointIdxRadiusSearch_new[t]].z));
	//	line_kmeans.push_back(line_t);
	//}
	//int pointId = 1;
	//vector<PointAngle> all_points;
	//for (int i = 0; i < line_kmeans.size(); i++)
	//{
	//	PointAngle point(pointId, line_kmeans[i]);
	//	all_points.push_back(point);
	//	pointId++;
	//}
	//int K = 2;
	//if ((int)all_points.size() <= K)
	//{
	//	return dis;
	//}
	//int iters = 1;
	//KMeans_fake_angle kmeans(K, iters);
	//double depthtoplane = 0.0;
	//depthtoplane = kmeans.run(all_points, depthPoint);

	return depthtoplane;
}
template<typename PointT> int
thickness::Thickness<PointT>::fake_test(pcl::PointXYZRGB midPoint, pcl::PointCloud<pcl::PointXYZ>::Ptr curvature_i, pcl::PCA<pcl::PointXYZ> pca, Eigen::Vector2f b, Eigen::RowVector3f V2, float dis, std::vector<int> pointIdxRadiusSearch, std::vector<float> pointRadiusSquaredDistance, std::vector<int> pointIdxRadiusSearch_new, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int num)
{
	double x1 = pca.getMean()[0];
	double y1 = pca.getMean()[1];
	double z1 = pca.getMean()[2];
	double x2 = midPoint.x;
	double y2 = midPoint.y;
	double z2 = (b[0] - V2[0] * midPoint.x - V2[1] * midPoint.y) / V2[2];
	double normal01_x = x2 - x1;
	double normal01_y = y2 - y1;
	double normal01_z = z2 - z1;
	for (size_t t = 1; t < pointIdxRadiusSearch.size(); ++t)
	{
		double normal02_x = cloud->points[pointIdxRadiusSearch[t]].x - x1;
		double normal02_y = cloud->points[pointIdxRadiusSearch[t]].y - y1;
		double normal02_z = cloud->points[pointIdxRadiusSearch[t]].z - z1;
		double fenzi = normal01_x * normal02_x + normal01_y * normal02_y + normal01_z * normal02_z;
		double lengthN1 = sqrt(normal01_x * normal01_x + normal01_y * normal01_y + normal01_z * normal01_z);
		double lengthN2 = sqrt(normal02_x * normal02_x + normal02_y * normal02_y + normal02_z * normal02_z);
		double hudu = acos(fenzi / (lengthN1 * lengthN2));
		double ds = abs(lengthN2 * sin(hudu));
		double threshold = dis * 0.2;//点到直线距离的阈值
		if (ds > threshold)
		{
			pointIdxRadiusSearch_new.push_back(pointIdxRadiusSearch[t]);
		}
	}
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
	pcl::PointXYZ Point2;
	Point2.x = midPoint.x + x;
	Point2.y = midPoint.y + y;
	Point2.z = midPoint.z + z;
	//求直线与球相交的两个点
	Vector3dNew O(midPoint.x, midPoint.y, midPoint.z);
	Vector3dNew E(Point2.x, Point2.y, Point2.z);
	Vector3dNew Center(midPoint.x, midPoint.y, midPoint.z);
	double R = dis * 0.9;
	vector<Vector3dNew> points;
	LineIntersectSphere(O, E, Center, R, points);
	vector<string> line_kmeans;
	for (auto it : points)
	{
		string line_t;
		line_t = to_string(it.x);
		line_t.append(" ");
		line_t.append(to_string(it.y));
		line_t.append(" ");
		line_t.append(to_string(it.z));
		line_kmeans.push_back(line_t);
	}
	for (size_t t = 0; t < pointIdxRadiusSearch_new.size(); t++)
	{
		string line_t;
		line_t = to_string(cloud->points[pointIdxRadiusSearch_new[t]].x);
		line_t.append(" ");
		line_t.append(to_string(cloud->points[pointIdxRadiusSearch_new[t]].y));
		line_t.append(" ");
		line_t.append(to_string(cloud->points[pointIdxRadiusSearch_new[t]].z));
		line_kmeans.push_back(line_t);
	}
	int pointId = 1;
	vector<PointNew> all_points;
	for (int i = 0; i < line_kmeans.size(); i++)
	{
		PointNew point(pointId, line_kmeans[i]);
		all_points.push_back(point);
		pointId++;
	}
	int K = 2;//二分类
	if ((int)all_points.size() <= K)
	{
		return num;
	}
	int iters = 1;
	KMeans_fake kmeans(K, iters);
	double angle = kmeans.run(all_points);
	//angle是角度的cos值
	if (abs(angle - 1.0) > 0.036)//0.05越小保留越多
	{
		num++;
	}
	return num;
}
template<typename PointT> void
thickness::Thickness<PointT>::SolvingQuadratics(double a, double b, double c, std::vector<double>& t)
{
	double delta = b * b - 4 * a * c;
	if (delta < 0)
	{
		return;
	}

	if (abs(delta) < EPSILON)
	{
		t.push_back(-b / (2 * a));
	}
	else
	{
		t.push_back((-b + sqrt(delta)) / (2 * a));
		t.push_back((-b - sqrt(delta)) / (2 * a));
	}
}
template<typename PointT> void
thickness::Thickness<PointT>::LineIntersectSphere(Vector3dNew& O, Vector3dNew& E, Vector3dNew& Center, double R, std::vector<Vector3dNew>& points)
{
	Vector3dNew D = E - O;			//线段方向向量

	double a = (D.x * D.x) + (D.y * D.y) + (D.z * D.z);
	double b = (2 * D.x * (O.x - Center.x) + 2 * D.y * (O.y - Center.y) + 2 * D.z * (O.z - Center.z));
	double c = ((O.x - Center.x) * (O.x - Center.x) + (O.y - Center.y) * (O.y - Center.y) + (O.z - Center.z) * (O.z - Center.z)) - R * R;

	vector<double> t;
	SolvingQuadratics(a, b, c, t);

	for (auto it : t)
	{
		points.push_back(O + D.Scalar(it));
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> std::vector<std::vector<std::pair<int, int>>>
	thickness::Thickness<PointT>::validationDirection_classBetween()
	{
		std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> cloud_first;
		std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> cloud_second;
		for (std::vector<std::vector<std::pair<int, int>>>::const_iterator it = thickness_index_.begin(); it != thickness_index_.end(); ++it)
		{
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster_first(new pcl::PointCloud<pcl::PointXYZRGB>);
			pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster_second(new pcl::PointCloud<pcl::PointXYZRGB>);
			int Random_color_r, Random_color_g, Random_color_b;
			Random_color_r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
			Random_color_g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
			Random_color_b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
			for (std::vector<std::pair<int, int>>::const_iterator pit = it->begin(); pit != it->end(); ++pit)
			{
				pcl::PointXYZRGB tempa, tempb;
				tempa.x = margin_a->points[(*pit).first].x;
				tempa.y = margin_a->points[(*pit).first].y;
				tempa.z = margin_a->points[(*pit).first].z;
				tempa.r = Random_color_r;
				tempa.g = Random_color_g;
				tempa.b = Random_color_b;
				cloud_cluster_first->push_back(tempa);
				tempb.x = margin_b->points[(*pit).second].x;
				tempb.y = margin_b->points[(*pit).second].y;
				tempb.z = margin_b->points[(*pit).second].z;
				tempb.r = Random_color_r;
				tempb.g = Random_color_g;
				tempb.b = Random_color_b;
				cloud_cluster_second->push_back(tempb);
			}
			cloud_cluster_first->width = cloud_cluster_first->points.size();
			cloud_cluster_first->height = 1;
			cloud_cluster_first->is_dense = true;
			cloud_cluster_second->width = cloud_cluster_second->points.size();
			cloud_cluster_second->height = 1;
			cloud_cluster_second->is_dense = true;
			cloud_first.push_back(cloud_cluster_first);
			cloud_second.push_back(cloud_cluster_second);
		}
		for (int i = 0; i < cloud_first.size(); i++)
		{
			//cloud_first主方向
			Eigen::Vector4f pcaCentroid_first;
			pcl::compute3DCentroid(*cloud_first[i], pcaCentroid_first);
			Eigen::Matrix3f covariance_first;
			pcl::computeCovarianceMatrix(*cloud_first[i], pcaCentroid_first, covariance_first);
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

			//cloud_second主方向
			Eigen::Vector4f pcaCentroid_second;
			pcl::compute3DCentroid(*cloud_second[i], pcaCentroid_second);
			Eigen::Matrix3f covariance_second;
			pcl::computeCovarianceMatrix(*cloud_second[i], pcaCentroid_second, covariance_second);
			Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver_second(covariance_second, Eigen::ComputeEigenvectors);
			Eigen::Matrix3f eigenVectorsPCA_second = eigen_solver_second.eigenvectors();
			Eigen::Vector3f eigenValuesPCA_second = eigen_solver_second.eigenvalues();
			float t1_second = eigenValuesPCA_second(0);
			int ii_second = 0;
			if (t1_second < eigenValuesPCA_second(1))
			{
				ii_second = 1;
				t1_second = eigenValuesPCA_second(1);
			}
			if (t1_second < eigenValuesPCA_second(2))
			{
				ii_second = 2;
				t1_second = eigenValuesPCA_second(2);
			}
			Eigen::Vector3f v_second(eigenVectorsPCA_second(0, ii_second), eigenVectorsPCA_second(1, ii_second), eigenVectorsPCA_second(2, ii_second));
			v_second /= v_second.norm();
			float angle;
			angle = abs((v_first(0) * v_second(0) + v_first(1) * v_second(1) + v_first(2) * v_second(2)) / (sqrt(v_first(0) * v_first(0) + v_first(1) * v_first(1) + v_first(2) * v_first(2)) * sqrt(v_second(0) * v_second(0) + v_second(1) * v_second(1) + v_second(2) * v_second(2))));
			if (abs(angle - 1.0) < 0.05)
			{
				thickness_index_validation_direction.push_back(thickness_index_[i]);
			}
		}
		return thickness_index_validation_direction;
	}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> std::vector<std::vector<std::pair<int, int>>>
	thickness::Thickness<PointT>::validationDirection_classWithin()
{
	std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> cloud_first;
	std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> cloud_second;
	for (std::vector<std::vector<std::pair<int, int>>>::const_iterator it = thickness_index_validation.begin(); it != thickness_index_validation.end(); ++it)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster_first(new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster_second(new pcl::PointCloud<pcl::PointXYZRGB>);
		int Random_color_r, Random_color_g, Random_color_b;
		Random_color_r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		for (std::vector<std::pair<int, int>>::const_iterator pit = it->begin(); pit != it->end(); ++pit)
		{
			pcl::PointXYZRGB tempa, tempb;
			tempa.x = margin_c->points[(*pit).first].x;
			tempa.y = margin_c->points[(*pit).first].y;
			tempa.z = margin_c->points[(*pit).first].z;
			tempa.r = Random_color_r;
			tempa.g = Random_color_g;
			tempa.b = Random_color_b;
			cloud_cluster_first->push_back(tempa);
			tempb.x = margin_c->points[(*pit).second].x;
			tempb.y = margin_c->points[(*pit).second].y;
			tempb.z = margin_c->points[(*pit).second].z;
			tempb.r = Random_color_r;
			tempb.g = Random_color_g;
			tempb.b = Random_color_b;
			cloud_cluster_second->push_back(tempb);
		}
		cloud_cluster_first->width = cloud_cluster_first->points.size();
		cloud_cluster_first->height = 1;
		cloud_cluster_first->is_dense = true;
		cloud_cluster_second->width = cloud_cluster_second->points.size();
		cloud_cluster_second->height = 1;
		cloud_cluster_second->is_dense = true;
		cloud_first.push_back(cloud_cluster_first);
		cloud_second.push_back(cloud_cluster_second);
	}
	for (int i = 0; i < cloud_first.size(); i++)
	{
		//cloud_first主方向
		Eigen::Vector4f pcaCentroid_first;
		pcl::compute3DCentroid(*cloud_first[i], pcaCentroid_first);
		Eigen::Matrix3f covariance_first;
		pcl::computeCovarianceMatrix(*cloud_first[i], pcaCentroid_first, covariance_first);
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

		//cloud_second主方向
		Eigen::Vector4f pcaCentroid_second;
		pcl::compute3DCentroid(*cloud_second[i], pcaCentroid_second);
		Eigen::Matrix3f covariance_second;
		pcl::computeCovarianceMatrix(*cloud_second[i], pcaCentroid_second, covariance_second);
		Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver_second(covariance_second, Eigen::ComputeEigenvectors);
		Eigen::Matrix3f eigenVectorsPCA_second = eigen_solver_second.eigenvectors();
		Eigen::Vector3f eigenValuesPCA_second = eigen_solver_second.eigenvalues();
		float t1_second = eigenValuesPCA_second(0);
		int ii_second = 0;
		if (t1_second < eigenValuesPCA_second(1))
		{
			ii_second = 1;
			t1_second = eigenValuesPCA_second(1);
		}
		if (t1_second < eigenValuesPCA_second(2))
		{
			ii_second = 2;
			t1_second = eigenValuesPCA_second(2);
		}
		Eigen::Vector3f v_second(eigenVectorsPCA_second(0, ii_second), eigenVectorsPCA_second(1, ii_second), eigenVectorsPCA_second(2, ii_second));
		v_second /= v_second.norm();
		float angle;
		angle = abs((v_first(0) * v_second(0) + v_first(1) * v_second(1) + v_first(2) * v_second(2)) / (sqrt(v_first(0) * v_first(0) + v_first(1) * v_first(1) + v_first(2) * v_first(2)) * sqrt(v_second(0) * v_second(0) + v_second(1) * v_second(1) + v_second(2) * v_second(2))));
		if (abs(angle - 1.0) < 0.05)  
		{
			thickness_index_validation_direction.push_back(thickness_index_validation[i]);
		}
	}
	return thickness_index_validation_direction;
}






















//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////聚类中函数实现////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT>
thickness::Cluster<PointT>::Cluster() :
	margin_A_(),
	margin_B_(),
	margin_C_(),
	thickness_index_(),
	thickness_index_after_clustering_(),
	node_list_(),
	th_cen_(0.94),//越小会欠分割
	th_cri_(0.955),
	th_fit_dis(0.001),//这个暂时没有效果

	line_w1_(0.2),
	line_w2_(0.65),
	line_w3_(0.15),

	cricle_w1_(0.15),
	cricle_w2_(0.4),
	cricle_w3_(0.45),
	dis_matrix_(),

	z_th_(1000),
	view_ang_th_(50)

{
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT>
thickness::Cluster<PointT>::~Cluster()
{
	thickness_index_after_clustering_.clear();
	node_list_.clear();
		
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Cluster<PointT>::inputMargin(const PointCloudConstPtr& cloudC)
{
	margin_C_ = cloudC;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Cluster<PointT>::inputMargin(const PointCloudConstPtr& cloudA, const PointCloudConstPtr& cloudB)
{
	margin_A_ = cloudA;
	margin_B_ = cloudB;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Cluster<PointT>::inputThicknessIndex(const std::vector<thicknessIndex>& thickness_index, int type)
{
	thickness_index_ = thickness_index;
	Type_ = type;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Cluster<PointT>::setMaxZ(float max_z)
{
	z_th_ = max_z;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Cluster<PointT>::setMaxViewAng(float ang)
{
	view_ang_th_ = ang;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> float
thickness::Cluster<PointT>::calNodeAngle(Node* A)
{
	Eigen::Vector4f centroid;                    // 质心
	Eigen::Matrix3f covariance_matrix;           // 协方差矩阵
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
	cloud_in = A->cloud_;
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
	double D = -normal.dot(centroid.head<3>());
	Eigen::Vector3f vector_with_origin;
	vector_with_origin[0] = 0 - A->center_.x;
	vector_with_origin[1] = 0 - A->center_.y;
	vector_with_origin[2] = 0 - A->center_.z;
	float ang = pcl::getAngle3D(normal, vector_with_origin,true);
	if (ang > 90)
	{
		normal[0] = -normal[0];
		normal[1] = -normal[1];
		normal[2] = -normal[2];
		ang = pcl::getAngle3D(normal, vector_with_origin,true);
	}
	return abs(ang);
	
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> bool
thickness::Cluster<PointT>::prepareWithClustering()
{
	if (Type_ == 0)
	{
		if (margin_C_ == nullptr|| thickness_index_.size() == 0)
		{
			return(false);
		}
		else
			return(true);
	}
	if (Type_ == 1)
	{
		if (margin_A_ == nullptr || margin_B_ == nullptr|| thickness_index_.size() == 0)
		{
			return(false);
		}
		else
			return(true);
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> pcl::PointCloud<pcl::PointXYZ>::Ptr
thickness::Cluster<PointT>::getNodePointcloud(Node* A)
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (A->level_ == 0)
	{
		if (Type_ == 1)
		{
			pcl::PointXYZ temp_a, temp_b;
			std::vector<std::pair<int, int>> thickness_index = A->node_thickness_index_;
			for (auto one_thickness_it : thickness_index)
			{
				temp_a = margin_A_->points[one_thickness_it.first];
				temp_b = margin_B_->points[one_thickness_it.second];
				cloud->push_back(temp_a);
				cloud->push_back(temp_b);
			}
			return cloud;
		}
		if(Type_ == 0)
		{
			pcl::PointXYZ temp_a, temp_b;
			std::vector<std::pair<int, int>> thickness_index = A->node_thickness_index_;
			for (auto one_thickness_it : thickness_index)
			{
				temp_a = margin_C_->points[one_thickness_it.first];
				temp_b = margin_C_->points[one_thickness_it.second];
				cloud->push_back(temp_a);
				cloud->push_back(temp_b);
			}
			return cloud;
		}
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Cluster<PointT>::initialization()
{
	if (Type_ == 0)
	{
		int cluster_id = 0;//初始的聚类编号
		for (auto one_thickness_indexs_it = thickness_index_.cbegin(); one_thickness_indexs_it != thickness_index_.cend(); one_thickness_indexs_it++)
		{
			Node* temp_node = new Node();
			temp_node->node_thickness_index_ = (*one_thickness_indexs_it);
			temp_node->level_ = 0;//初始的聚类层数，0意味是在最底层
			temp_node->cluster_id_ = cluster_id++;
			temp_node->cluster_num = 1;
			node_list_.push_back(temp_node);
		}
		//求每个初始node的质心和点云
		for (auto one_node : node_list_)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
			Eigen::Vector4f centroid;
			pcl::PointXYZ temp_center;
			cloud = getNodePointcloud(one_node);
			pcl::compute3DCentroid(*cloud, centroid);
			temp_center.x = centroid[0];
			temp_center.y = centroid[1];
			temp_center.z = centroid[2];
			one_node->center_ = temp_center;
			one_node->cloud_ = cloud;
		}

		for (auto onenode = node_list_.begin(); onenode != node_list_.end(); onenode++)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr mid_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointXYZ pmin, pmax;
			for (int index = 0; index < static_cast<int>((*onenode)->node_thickness_index_.size()); index++)
			{
				int a = (*onenode)->node_thickness_index_[index].first;
				int b = (*onenode)->node_thickness_index_[index].second;
				pcl::PointXYZ point_mid;
				point_mid.x = (margin_C_->points[a].x + margin_C_->points[b].x) / 2;
				point_mid.y = (margin_C_->points[a].y + margin_C_->points[b].y) / 2;
				point_mid.z = (margin_C_->points[a].z + margin_C_->points[b].z) / 2;
				mid_cloud->push_back(point_mid);
			}
			float L = pcl::getMaxSegment(*mid_cloud, pmin, pmax);
			(*onenode)->pmin_ = pmin;
			(*onenode)->pmax_ = pmax;
			(*onenode)->l_ = L;
		}
		for (auto onenode = node_list_.cbegin(); onenode != node_list_.cend();)
		{
			if ((*onenode)->l_ < 0.005)
			{
				onenode = node_list_.erase(onenode);
			}
			else
			{
				onenode++;
			}
		}
	}

	if (Type_ == 1)
	{
		int cluster_id = 0;//初始的聚类编号
		for (auto one_thickness_indexs_it = thickness_index_.cbegin(); one_thickness_indexs_it != thickness_index_.cend(); one_thickness_indexs_it++)
		{
			Node* temp_node = new Node();
			temp_node->node_thickness_index_ = (*one_thickness_indexs_it);
			temp_node->level_ = 0;//初始的聚类层数，0意味是在最底层
			temp_node->cluster_id_ = cluster_id++;
			temp_node->cluster_num = 1;
			node_list_.push_back(temp_node);
		}
		//求每个初始node的质心和点云
		for (auto one_node : node_list_)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
			Eigen::Vector4f centroid;
			pcl::PointXYZ temp_center;

			cloud = getNodePointcloud(one_node);
			pcl::compute3DCentroid(*cloud, centroid);
			temp_center.x = centroid[0];
			temp_center.y = centroid[1];
			temp_center.z = centroid[2];
			one_node->center_ = temp_center;
			one_node->cloud_ = cloud;
		}
		for (auto onenode = node_list_.begin(); onenode != node_list_.end(); onenode++)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr mid_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointXYZ pmin, pmax;
			for (int index = 0; index < static_cast<int>((*onenode)->node_thickness_index_.size()); index++)
			{
				int a = (*onenode)->node_thickness_index_[index].first;
				int b = (*onenode)->node_thickness_index_[index].second;
				pcl::PointXYZ point_mid;
				point_mid.x = (margin_A_->points[a].x + margin_B_->points[b].x) / 2;
				point_mid.y = (margin_A_->points[a].y + margin_B_->points[b].y) / 2;
				point_mid.z = (margin_A_->points[a].z + margin_B_->points[b].z) / 2;
				mid_cloud->push_back(point_mid);
			}
			float L = pcl::getMaxSegment(*mid_cloud, pmin, pmax);
			(*onenode)->pmin_ = pmin;
			(*onenode)->pmax_ = pmax;
			(*onenode)->l_ = L;
		}
		for (auto onenode = node_list_.cbegin(); onenode != node_list_.cend();)
		{
			if ((*onenode)->l_ < 0.005)
			{
				onenode=node_list_.erase(onenode);
			}
			else
			{
				onenode++;
			}
		}
	}
	
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> void
thickness::Cluster<PointT>::clustering(std::vector<thicknessIndex>& thickness_index_arg)
{
	if (Type_ == 0)
	{
		bool cluster_is_possible = this->prepareWithClustering();
		if (!cluster_is_possible)
		{
			std::cout << "未检测出类内厚度特征！！" << std::endl;
			return;
		}
		//初始化
		initialization();
		if (node_list_.size() == 0)
		{
			cout << "未检测出类内厚度特征！！" << endl;
			return;
		}
		//第一次计算距离矩阵
		for (int i = 0; i < node_list_.size(); i++)
		{
			std::vector<float> temp;
			for (int j = 0; j < node_list_.size(); j++)
			{
				if (i < j)
				{
					Node* A = node_list_[i];
					Node* B = node_list_[j];
					float dis_curr_cen = calCenterDis(A, B);
					float dis_curr_fit = calFitLineDisSelf(A, B);
					float dis_curr_hou = calThicknessDisSelf(A, B);
					float dis_add_weight = calLineWeightingDis(dis_curr_cen, dis_curr_fit, dis_curr_hou);
					temp.push_back(dis_add_weight);

				}
				else
				{
					temp.push_back(0);
				}

			}
			dis_matrix_.push_back(temp);
		}
		while (true)
		{
			float MaxDst = 0;
			int find_i, find_j; //用于记录最小两簇的索引
			for (int i = 0; i < dis_matrix_.size(); i++) //遍历的方法找到距离最小的两簇
			{
				for (int j = i + 1; j < dis_matrix_[i].size(); j++)
				{
					if (dis_matrix_[i][j] > MaxDst)
					{
						find_i = i;
						find_j = j;
						MaxDst = dis_matrix_[i][j];
					}
				}
			}
			//进行聚类前判断加权间距是否大于阈值
			if (MaxDst < th_cen_)
			{
				break;
			}
			//更新nodeA,把nodeB加到A上，成为新的node
			Node* temp_a_node = node_list_[find_i];
			Node* temp_b_node = node_list_[find_j];
			Node* temp_node = new Node(temp_a_node, temp_b_node);
			auto node_index_a = temp_a_node->node_thickness_index_;
			auto node_index_b = temp_b_node->node_thickness_index_;
			for (auto i : node_index_b)
			{
				node_index_a.push_back(i);
			}
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_a(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_b(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_new_a(new pcl::PointCloud<pcl::PointXYZ>);
			cloud_a = temp_a_node->cloud_;
			cloud_b = temp_b_node->cloud_;
			*cloud_new_a = (*cloud_a) + (*cloud_b);

			//测试用
			/*cloud_new_a->width = cloud_new_a->points.size();
			cloud_new_a->height = 1;
			cloud_new_a->is_dense = true;
			stringstream ss;
			ss << "new" << ".pcd";
			pcl::PCDWriter writer;
			writer.write<pcl::PointXYZ>(ss.str(), *cloud_new_a, false);*/

			pcl::PointCloud<pcl::PointXYZ>::Ptr mid_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointXYZ pmin, pmax;
			for (int index = 0; index < static_cast<int>(temp_node->node_thickness_index_.size()); index++)
			{
				int a = temp_node->node_thickness_index_[index].first;
				int b = temp_node->node_thickness_index_[index].second;
				pcl::PointXYZ point_mid;
				point_mid.x = (margin_C_->points[a].x + margin_C_->points[b].x) / 2;
				point_mid.y = (margin_C_->points[a].y + margin_C_->points[b].y) / 2;
				point_mid.z = (margin_C_->points[a].z + margin_C_->points[b].z) / 2;
				mid_cloud->push_back(point_mid);
			}
			float L = pcl::getMaxSegment(*mid_cloud, pmin, pmax);
			temp_node->pmin_ = pmin;
			temp_node->pmax_ = pmax;

			Eigen::Vector4f centroid;
			pcl::PointXYZ temp_center;
			pcl::compute3DCentroid(*cloud_new_a, centroid);
			temp_center.x = centroid[0];
			temp_center.y = centroid[1];
			temp_center.z = centroid[2];
			temp_node->node_thickness_index_ = node_index_a;
			temp_node->center_ = temp_center;
			temp_node->level_ = temp_a_node->level_ + temp_b_node->level_ + 1;
			temp_node->cloud_ = cloud_new_a;
			temp_node->cluster_num = temp_a_node->cluster_num + temp_b_node->cluster_num;
			node_list_.erase(node_list_.cbegin() + find_j);
			node_list_[find_i] = temp_node;

			//更新距离矩阵
			dis_matrix_.erase(dis_matrix_.cbegin() + find_j);
			for (int i = 0; i < dis_matrix_.size(); i++)
			{
				dis_matrix_[i].erase(dis_matrix_[i].cbegin() + find_j);
			}
			for (int i = 0; i < node_list_.size(); i++)
			{
				for (int j = 0; j < node_list_.size(); j++)
				{
					if (j == find_i && j > i)
					{
						Node* A = node_list_[i];
						Node* B = node_list_[j];
						float dis_curr_cen = calCenterDis(A, B);
						float dis_curr_fit = calFitLineDisSelf(A, B);
						float dis_curr_hou = calThicknessDisSelf(A, B);
						float dis_add_weight = calLineWeightingDis(dis_curr_cen, dis_curr_fit, dis_curr_hou);
						dis_matrix_[i][j] = dis_add_weight;
					}
				}
				if (i == find_i)
				{
					for (int k = 0; k < dis_matrix_[i].size(); k++)
					{
						if (k > i)
						{
							Node* A = node_list_[i];
							Node* B = node_list_[k];
							float dis_curr_cen = calCenterDis(A, B);
							float dis_curr_fit = calFitLineDisSelf(A, B);
							float dis_curr_hou = calThicknessDisSelf(A, B);
							float dis_add_weight = calLineWeightingDis(dis_curr_cen, dis_curr_fit, dis_curr_hou);
							dis_matrix_[i][k] = dis_add_weight;
						}
						else
						{
							dis_matrix_[i][k] = 0;
						}
					}
				}

			}
			//it++;
		}


		//圆合并
		//第一次计算距离矩阵
		for (int i = 0; i < node_list_.size(); i++)
		{
			for (int j = 0; j < node_list_.size(); j++)
			{
				if (i < j)
				{
					Node* A = node_list_[i];
					Node* B = node_list_[j];
					float dis_curr_cen = calCenterDis(A, B);
					float dis_curr_fit = calFitCricleDis_classWithin(A, B);
					float dis_curr_hou = calThicknessDisSelf(A, B);
					float dis_add_weight = calLineWeightingDis(dis_curr_cen, dis_curr_fit, dis_curr_hou);
					dis_matrix_[i][j] = dis_add_weight;

				}

				else
				{
					dis_matrix_[i][j] = 0;
				}

			}
		}

		while (true)
		{
			float MaxDst = 0;
			int find_i, find_j; //用于记录最小两簇的索引
			for (int i = 0; i < dis_matrix_.size(); i++) //遍历的方法找到距离最小的两簇
			{
				for (int j = i + 1; j < dis_matrix_[i].size(); j++)
				{
					if (dis_matrix_[i][j] > MaxDst)
					{
						find_i = i;
						find_j = j;
						MaxDst = dis_matrix_[i][j];
					}
				}
			}

			if (MaxDst < th_cri_)
			{
				break;
			}
			//更新nodeA,把nodeB加到A上，成为新的node
			Node* temp_a_node = node_list_[find_i];
			Node* temp_b_node = node_list_[find_j];
			Node* temp_node = new Node(temp_a_node, temp_b_node);
			auto node_index_a = temp_a_node->node_thickness_index_;
			auto node_index_b = temp_b_node->node_thickness_index_;
			for (auto i : node_index_b)
			{
				node_index_a.push_back(i);
			}
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_a(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_b(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_new_a(new pcl::PointCloud<pcl::PointXYZ>);
			cloud_a = temp_a_node->cloud_;
			cloud_b = temp_b_node->cloud_;
			*cloud_new_a = (*cloud_a) + (*cloud_b);

			//求端点
			pcl::PointCloud<pcl::PointXYZ>::Ptr mid_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointXYZ pmin, pmax;
			for (int index = 0; index < static_cast<int>(temp_node->node_thickness_index_.size()); index++)
			{
				int a = temp_node->node_thickness_index_[index].first;
				int b = temp_node->node_thickness_index_[index].second;
				pcl::PointXYZ point_mid;
				point_mid.x = (margin_C_->points[a].x + margin_C_->points[b].x) / 2;
				point_mid.y = (margin_C_->points[a].y + margin_C_->points[b].y) / 2;
				point_mid.z = (margin_C_->points[a].z + margin_C_->points[b].z) / 2;
				mid_cloud->push_back(point_mid);
			}
			float L = pcl::getMaxSegment(*mid_cloud, pmin, pmax);
			temp_node->pmin_ = pmin;
			temp_node->pmax_ = pmax;

			//求新的质心和点云
			Eigen::Vector4f centroid;
			pcl::PointXYZ temp_center;
			pcl::compute3DCentroid(*cloud_new_a, centroid);
			temp_center.x = centroid[0];
			temp_center.y = centroid[1];
			temp_center.z = centroid[2];
			temp_node->node_thickness_index_ = node_index_a;
			temp_node->center_ = temp_center;
			temp_node->level_ = temp_a_node->level_ + temp_b_node->level_ + 1;
			temp_node->cloud_ = cloud_new_a;
			temp_node->cluster_num = temp_a_node->cluster_num + temp_b_node->cluster_num;
			node_list_.erase(node_list_.cbegin() + find_j);
			node_list_[find_i] = temp_node;

			//更新距离矩阵
			dis_matrix_.erase(dis_matrix_.cbegin() + find_j);
			for (int i = 0; i < dis_matrix_.size(); i++)
			{
				dis_matrix_[i].erase(dis_matrix_[i].cbegin() + find_j);
			}
			for (int i = 0; i < node_list_.size(); i++)
			{
				for (int j = 0; j < node_list_.size(); j++)
				{
					if (j == find_i && j > i)
					{
						Node* A = node_list_[i];
						Node* B = node_list_[j];
						float dis_curr_cen = calCenterDis(A, B);
						float dis_curr_fit = calFitCricleDis_classWithin(A, B);
						float dis_curr_hou = calThicknessDisSelf(A, B);
						float dis_add_weight = calLineWeightingDis(dis_curr_cen, dis_curr_fit, dis_curr_hou);
						dis_matrix_[i][j] = dis_add_weight;
					}
				}
				if (i == find_i)
				{
					for (int k = 0; k < dis_matrix_[i].size(); k++)
					{
						if (k > i)
						{
							Node* A = node_list_[i];
							Node* B = node_list_[k];
							float dis_curr_cen = calCenterDis(A, B);
							float dis_curr_fit = calFitCricleDis_classWithin(A, B);
							float dis_curr_hou = calThicknessDisSelf(A, B);
							float dis_add_weight = calLineWeightingDis(dis_curr_cen, dis_curr_fit, dis_curr_hou);
							dis_matrix_[i][k] = dis_add_weight;
						}
						else
						{
							dis_matrix_[i][k] = 0;
						}
					}
				}

			}
		}
		//分类树和删除长度较小的厚度
		
		vector<int> node_height;
		for (auto onenode : node_list_)
		{
			node_height.push_back(onenode->level_);
		}
		sort(node_height.begin(), node_height.end());
		int n = node_height.size();
		int i = 0;
		int MaxCount = 1;
		int index = 0;
		while (i < n - 1)
		{
			int count = 1;
			int j;
			for (j = i; j < n - 1; j++)
			{
				if (node_height[j] == node_height[j + 1])//存在连续两个数相等，则众数+1
				{
					count++;
				}
				else
				{
					break;
				}
			}
			if (MaxCount < count)
			{
				MaxCount = count;//当前最大众数
				index = j;//当前众数标记位置
			}
			++j;
			i = j;//位置后移到下一个未出现的数字
		}
		int max_level = node_height[index] + 2;//在众数上加3作为限制的树高
		if (max_level >= *(node_height.end()-1))
		{
			max_level = *(node_height.end() - 1);
		}
		while (1)
		{
			int Maxlevel = 0;
			for (int i = 0; i < node_list_.size(); i++) 
			{
				if (node_list_[i]->level_ > Maxlevel)
				{
					Maxlevel = node_list_[i]->level_;
				}
			}
			if (Maxlevel < max_level)
			{
				break;
			}
			if (max_level == 0)
			{
				break;
			}
			for (int onenode_i = 0; onenode_i < node_list_.size(); onenode_i++)
			{
				if (node_list_[onenode_i]->level_ >= max_level)
				{
					Node* temp_a = node_list_[onenode_i]->left_;
					Node* temp_b = node_list_[onenode_i]->right_;
					node_list_.push_back(temp_b);
					node_list_[onenode_i] = temp_a;
				}
			}

		}
		//删除质心z坐标大于一定阈值的node
		for (auto onenode = node_list_.begin(); onenode != node_list_.end();)
		{
			if ((*onenode)->center_.z > z_th_)
			{
				onenode = node_list_.erase(onenode);
			}
			else
			{
				onenode++;
			}
		}

		//删除视角角度大于一定阈值的node
		vector<float> ang_results;
		for (auto onenode = node_list_.begin(); onenode != node_list_.end();)
		{
			float ang_res = calNodeAngle((*onenode));
			ang_results.push_back(ang_res);
			if (ang_res > view_ang_th_)
			{
				onenode = node_list_.erase(onenode);
			}
			else
			{
				onenode++;
			}
		}
		


		for (auto onenode = node_list_.begin(); onenode != node_list_.end();)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr mid_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointXYZ pmin, pmax;
			for (int index = 0; index < static_cast<int>((*onenode)->node_thickness_index_.size()); index++)
			{
				int a = (*onenode)->node_thickness_index_[index].first;
				int b = (*onenode)->node_thickness_index_[index].second;
				pcl::PointXYZ point_mid;
				point_mid.x = (margin_C_->points[a].x + margin_C_->points[b].x) / 2;
				point_mid.y = (margin_C_->points[a].y + margin_C_->points[b].y) / 2;
				point_mid.z = (margin_C_->points[a].z + margin_C_->points[b].z) / 2;
				mid_cloud->push_back(point_mid);
			}
			float L = pcl::getMaxSegment(*mid_cloud, pmin, pmax);
			
			if (L < 0.02)
			{
				onenode = node_list_.erase(onenode);
			}
			else
			{
				onenode++;
			}
		}
		if (node_list_.size() == 0)
		{
			cout << "未检测出类内厚度特征！！" << endl;
			return;
		}
		for (auto onenode : node_list_)
		{
			thickness_index_arg.push_back(onenode->node_thickness_index_);

		}
		//测试用,用于保存聚类后的点云
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_after(new pcl::PointCloud<pcl::PointXYZRGB>);
		for (auto onenode : node_list_)
		{
			int Random_color_r, Random_color_g, Random_color_b;
			Random_color_r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
			Random_color_g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
			Random_color_b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
			pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			temp_cloud = onenode->cloud_;
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
	/*	cloud_after->width = cloud_after->points.size();
		cloud_after->height = 1;
		cloud_after->is_dense = true;
		stringstream ss;
		ss << "类内聚类后的全部厚度" << ".pcd";
		pcl::PCDWriter writer;
		writer.write<pcl::PointXYZRGB>(ss.str(), *cloud_after, false);*/

	}


	if (Type_ == 1)
	{
		bool cluster_is_possible = this->prepareWithClustering();
		if (!cluster_is_possible)
		{
			std::cout << "未检测出类间厚度特征！！" << std::endl;
			return;
		}
		//初始化
		initialization();
		if (node_list_.size() == 0)
		{
			cout << "未检测出类间厚度特征！！" << endl;
			return;
		}
		//第一次计算距离矩阵
		for (int i = 0; i < node_list_.size(); i++)
		{
			std::vector<float> temp;
			for (int j = 0; j < node_list_.size(); j++)
			{
				if (i < j)
				{
					Node* A = node_list_[i];
					Node* B = node_list_[j];
					float dis_curr_cen = calCenterDis(A, B);
					float dis_curr_fit = calFitLineDis(A, B);
					float dis_curr_hou = calThicknessDis(A, B);
					float dis_add_weight = calLineWeightingDis(dis_curr_cen, dis_curr_fit, dis_curr_hou);
					temp.push_back(dis_add_weight);

				}
				else
				{
					temp.push_back(0);
				}

			}
			dis_matrix_.push_back(temp);
		}

		/*for (int i = 0; i < node_list_.size(); i++)
		{
			for (int j = i+1; j < node_list_.size(); j++)
			{
				Node* A = node_list_[i];
				Node* B = node_list_[j];
				float dis_curr_cen = calCenterDis(A, B);
				float dis_curr_fit = calFitLineDis(A, B);
				float dis_curr_hou = calThicknessDis(A, B);
				float dis_add_weight = calLineWeightingDis(dis_curr_cen, dis_curr_fit, dis_curr_hou);
				DataMap[i][j] = dis_add_weight;
				DataMap[j][i] = dis_add_weight;
			}
		}*/
		//int it = 0;
		while (true)
		{
			/*if (it == 80)
			{
				break;
			}*/
			////求当前两node之间质心最小聚类距离
			//float cen_dis_min = INT_MAX;
			//for (int i = 0; i < node_list_.size(); i++)
			//{
			//	for (int j = i + 1; j < node_list_.size(); j++)
			//	{
			//		Node* A = node_list_[i];
			//		Node* B = node_list_[j];
			//		float dis_curr_cen = calCenterDis(A, B);
			//		if (dis_curr_cen < cen_dis_min)
			//		{
			//			cen_dis_min = dis_curr_cen;
			//		}

			//	}
			//}

			float MaxDst = 0;
			int find_i, find_j; //用于记录最小两簇的索引
			for (int i = 0; i < dis_matrix_.size(); i++) //遍历的方法找到距离最小的两簇
			{
				for (int j = i + 1; j < dis_matrix_[i].size(); j++)
				{
					if (dis_matrix_[i][j] > MaxDst)
					{
						find_i = i;
						find_j = j;
						MaxDst = dis_matrix_[i][j];
					}
				}
			}
			//cout << MaxDst << endl;
			//进行聚类前判断加权间距是否大于阈值
			if (MaxDst < th_cen_)
			{
				break;
			}
			//更新nodeA,把nodeB加到A上，成为新的node
			Node* temp_a_node = node_list_[find_i];
			Node* temp_b_node = node_list_[find_j];
			Node* temp_node = new Node(temp_a_node, temp_b_node);
			auto node_index_a = temp_a_node->node_thickness_index_;
			auto node_index_b = temp_b_node->node_thickness_index_;
			for (auto i : node_index_b)
			{
				node_index_a.push_back(i);
			}
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_a(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_b(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_new_a(new pcl::PointCloud<pcl::PointXYZ>);
			cloud_a = temp_a_node->cloud_;
			cloud_b = temp_b_node->cloud_;
			*cloud_new_a = (*cloud_a) + (*cloud_b);

			//测试用
			/*cloud_new_a->width = cloud_new_a->points.size();
			cloud_new_a->height = 1;
			cloud_new_a->is_dense = true;
			stringstream ss;
			ss << "new" << ".pcd";
			pcl::PCDWriter writer;
			writer.write<pcl::PointXYZ>(ss.str(), *cloud_new_a, false);*/

			pcl::PointCloud<pcl::PointXYZ>::Ptr mid_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointXYZ pmin, pmax;
			for (int index = 0; index < static_cast<int>(temp_node->node_thickness_index_.size()); index++)
			{
				int a = temp_node->node_thickness_index_[index].first;
				int b = temp_node->node_thickness_index_[index].second;
				pcl::PointXYZ point_mid;
				point_mid.x = (margin_A_->points[a].x + margin_B_->points[b].x) / 2;
				point_mid.y = (margin_A_->points[a].y + margin_B_->points[b].y) / 2;
				point_mid.z = (margin_A_->points[a].z + margin_B_->points[b].z) / 2;
				mid_cloud->push_back(point_mid);
			}
			float L = pcl::getMaxSegment(*mid_cloud, pmin, pmax);
			temp_node->pmin_ = pmin;
			temp_node->pmax_ = pmax;

			Eigen::Vector4f centroid;
			pcl::PointXYZ temp_center;
			pcl::compute3DCentroid(*cloud_new_a, centroid);
			temp_center.x = centroid[0];
			temp_center.y = centroid[1];
			temp_center.z = centroid[2];
			temp_node->node_thickness_index_ = node_index_a;
			temp_node->center_ = temp_center;
			temp_node->level_ = temp_a_node->level_ + temp_b_node->level_ + 1;
			temp_node->cloud_ = cloud_new_a;
			temp_node->cluster_num = temp_a_node->cluster_num + temp_b_node->cluster_num;
			node_list_.erase(node_list_.cbegin() + find_j);
			node_list_[find_i] = temp_node;

			//更新距离矩阵
			dis_matrix_.erase(dis_matrix_.cbegin() + find_j);
			for (int i = 0; i < dis_matrix_.size(); i++)
			{
				dis_matrix_[i].erase(dis_matrix_[i].cbegin() + find_j);
			}
			for (int i = 0; i < node_list_.size(); i++)
			{
				for (int j = 0; j < node_list_.size(); j++)
				{
					if (j == find_i && j > i)
					{
						Node* A = node_list_[i];
						Node* B = node_list_[j];
						float dis_curr_cen = calCenterDis(A, B);
						float dis_curr_fit = calFitLineDis(A, B);
						float dis_curr_hou = calThicknessDis(A, B);
						float dis_add_weight = calLineWeightingDis(dis_curr_cen, dis_curr_fit, dis_curr_hou);
						dis_matrix_[i][j] = dis_add_weight;
					}
				}
				if (i == find_i)
				{
					for (int k = 0; k < dis_matrix_[i].size(); k++)
					{
						if (k > i)
						{
							Node* A = node_list_[i];
							Node* B = node_list_[k];
							float dis_curr_cen = calCenterDis(A, B);
							float dis_curr_fit = calFitLineDis(A, B);
							float dis_curr_hou = calThicknessDis(A, B);
							float dis_add_weight = calLineWeightingDis(dis_curr_cen, dis_curr_fit, dis_curr_hou);
							dis_matrix_[i][k] = dis_add_weight;
						}
						else
						{
							dis_matrix_[i][k] = 0;
						}
					}
				}

			}
			//it++;
		}

		//圆合并
		//第一次计算距离矩阵
		
	
		for (int i = 0; i < node_list_.size(); i++)
		{
			for (int j = 0; j < node_list_.size(); j++)
			{
				if (i < j)
				{
					Node* A = node_list_[i];
					Node* B = node_list_[j];
					float dis_curr_cen = calCenterDis(A, B);
					float dis_curr_fit = calFitCricleDis_classBetween(A, B);
					float dis_curr_hou = calThicknessDis(A, B);
					float dis_add_weight = calLineWeightingDis(dis_curr_cen, dis_curr_fit, dis_curr_hou);
					dis_matrix_[i][j] = dis_add_weight;

				}

				else
				{
					dis_matrix_[i][j] = 0;
				}

			}
		}

		while (true)
		{
			float MaxDst = 0;
			int find_i, find_j; //用于记录最小两簇的索引
			for (int i = 0; i < dis_matrix_.size(); i++) //遍历的方法找到距离最小的两簇
			{
				for (int j = i + 1; j < dis_matrix_[i].size(); j++)
				{
					if (dis_matrix_[i][j] > MaxDst)
					{
						find_i = i;
						find_j = j;
						MaxDst = dis_matrix_[i][j];
					}
				}
			}

			if (MaxDst < th_cri_)
			{
				break;
			}
			//更新nodeA,把nodeB加到A上，成为新的node
			Node* temp_a_node = node_list_[find_i];
			Node* temp_b_node = node_list_[find_j];
			Node* temp_node = new Node(temp_a_node, temp_b_node);
			auto node_index_a = temp_a_node->node_thickness_index_;
			auto node_index_b = temp_b_node->node_thickness_index_;
			for (auto i : node_index_b)
			{
				node_index_a.push_back(i);
			}
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_a(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_b(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_new_a(new pcl::PointCloud<pcl::PointXYZ>);
			cloud_a = temp_a_node->cloud_;
			cloud_b = temp_b_node->cloud_;
			*cloud_new_a = (*cloud_a) + (*cloud_b);

			pcl::PointCloud<pcl::PointXYZ>::Ptr mid_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointXYZ pmin, pmax;
			for (int index = 0; index < static_cast<int>(temp_node->node_thickness_index_.size()); index++)
			{
				int a = temp_node->node_thickness_index_[index].first;
				int b = temp_node->node_thickness_index_[index].second;
				pcl::PointXYZ point_mid;
				point_mid.x = (margin_A_->points[a].x + margin_B_->points[b].x) / 2;
				point_mid.y = (margin_A_->points[a].y + margin_B_->points[b].y) / 2;
				point_mid.z = (margin_A_->points[a].z + margin_B_->points[b].z) / 2;
				mid_cloud->push_back(point_mid);
			}
			float L = pcl::getMaxSegment(*mid_cloud, pmin, pmax);
			temp_node->pmin_ = pmin;
			temp_node->pmax_ = pmax;

			Eigen::Vector4f centroid;
			pcl::PointXYZ temp_center;
			pcl::compute3DCentroid(*cloud_new_a, centroid);
			temp_center.x = centroid[0];
			temp_center.y = centroid[1];
			temp_center.z = centroid[2];
			temp_node->node_thickness_index_ = node_index_a;
			temp_node->center_ = temp_center;
			temp_node->level_ = temp_a_node->level_ + temp_b_node->level_ + 1;
			temp_node->cloud_ = cloud_new_a;
			temp_node->cluster_num = temp_a_node->cluster_num + temp_b_node->cluster_num;
			node_list_.erase(node_list_.cbegin() + find_j);
			node_list_[find_i] = temp_node;

			//更新距离矩阵
			dis_matrix_.erase(dis_matrix_.cbegin() + find_j);
			for (int i = 0; i < dis_matrix_.size(); i++)
			{
				dis_matrix_[i].erase(dis_matrix_[i].cbegin() + find_j);
			}
			for (int i = 0; i < node_list_.size(); i++)
			{
				for (int j = 0; j < node_list_.size(); j++)
				{
					if (j == find_i && j > i)
					{
						Node* A = node_list_[i];
						Node* B = node_list_[j];
						float dis_curr_cen = calCenterDis(A, B);
						float dis_curr_fit = calFitCricleDis_classBetween(A, B);
						float dis_curr_hou = calThicknessDis(A, B);
						float dis_add_weight = calLineWeightingDis(dis_curr_cen, dis_curr_fit, dis_curr_hou);
						dis_matrix_[i][j] = dis_add_weight;
					}
				}
				if (i == find_i)
				{
					for (int k = 0; k < dis_matrix_[i].size(); k++)
					{
						if (k > i)
						{
							Node* A = node_list_[i];
							Node* B = node_list_[k];
							float dis_curr_cen = calCenterDis(A, B);
							float dis_curr_fit = calFitCricleDis_classBetween(A, B);
							float dis_curr_hou = calThicknessDis(A, B);
							float dis_add_weight = calLineWeightingDis(dis_curr_cen, dis_curr_fit, dis_curr_hou);
							dis_matrix_[i][k] = dis_add_weight;
						}
						else
						{
							dis_matrix_[i][k] = 0;
						}
					}
				}

			}
		}

		//分类树和删除长度较小的厚度
		vector<int> node_height;
		for (auto onenode : node_list_)
		{
			node_height.push_back(onenode->level_);
		}
		sort(node_height.begin(), node_height.end());
		int n = node_height.size();
		int i = 0;
		int MaxCount = 1;
		int index = 0;
		while (i < n - 1)
		{
			int count = 1;
			int j;
			for (j = i; j < n - 1; j++)
			{
				if (node_height[j] == node_height[j + 1])//存在连续两个数相等，则众数+1
				{
					count++;
				}
				else
				{
					break;
				}
			}
			if (MaxCount < count)
			{
				MaxCount = count;//当前最大众数
				index = j;//当前众数标记位置
			}
			++j;
			i = j;//位置后移到下一个未出现的数字
		}
		int max_level = node_height[index] + 2;//在众数上加个数作为限制的树高
		if (max_level >= *(node_height.end() - 1))//加2后不能大于当前list的最大树高
		{
			max_level = *(node_height.end() - 1);
		}
		while (1)
		{
			int Maxlevel = 0;
			for (int i = 0; i < node_list_.size(); i++) 
			{
				if (node_list_[i]->level_ > Maxlevel)
				{
					Maxlevel = node_list_[i]->level_;
				}
			}
			if (Maxlevel < max_level)
			{
				break;
			}
			if (max_level == 0)
			{
				break;
			}
			for (int onenode_i = 0; onenode_i < node_list_.size(); onenode_i++)
			{
				if (node_list_[onenode_i]->level_ >= max_level)
				{
					Node* temp_a = node_list_[onenode_i]->left_;
					Node* temp_b = node_list_[onenode_i]->right_;
					node_list_.push_back(temp_b);
					node_list_[onenode_i] = temp_a;
				}
			}
			
		}
		//删除质心z坐标大于一定阈值的node
		for (auto onenode = node_list_.begin(); onenode != node_list_.end();)
		{
			
			if ((*onenode)->center_.z > z_th_)
			{
				onenode = node_list_.erase(onenode);
			}
			else
			{
				onenode++;
			}
		}

		//删除视角角度大于一定阈值的node
		vector<float> ang_results;
		for (auto onenode = node_list_.begin(); onenode != node_list_.end();)
		{
			float ang_res = calNodeAngle((*onenode));
			ang_results.push_back(ang_res);
			if (ang_res > view_ang_th_)
			{
				onenode = node_list_.erase(onenode);
			}
			else
			{
				onenode++;
			}
		}
		for (auto onenode = node_list_.begin(); onenode != node_list_.end();)
		{
			pcl::PointCloud<pcl::PointXYZ>::Ptr mid_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			pcl::PointXYZ pmin, pmax;
			for (int index = 0; index < static_cast<int>((*onenode)->node_thickness_index_.size()); index++)
			{
				int a = (*onenode)->node_thickness_index_[index].first;
				int b = (*onenode)->node_thickness_index_[index].second;
				pcl::PointXYZ point_mid;
				point_mid.x = (margin_A_->points[a].x + margin_B_->points[b].x) / 2;
				point_mid.y = (margin_A_->points[a].y + margin_B_->points[b].y) / 2;
				point_mid.z = (margin_A_->points[a].z + margin_B_->points[b].z) / 2;
				mid_cloud->push_back(point_mid);
			}
			float L = pcl::getMaxSegment(*mid_cloud, pmin, pmax);
			if (L < 0.02)
			{
				onenode = node_list_.erase(onenode);
			}
			else
			{
				onenode++;
			}
		}
		if (node_list_.size() == 0)
		{
			cout << "未检测出类间厚度特征！！" << endl;
			return;
		}
		for (auto onenode : node_list_)
		{
			thickness_index_arg.push_back(onenode->node_thickness_index_);
		}
		//测试用,用于保存聚类后的点云
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_after(new pcl::PointCloud<pcl::PointXYZRGB>);
		for (auto onenode : node_list_)
		{
			int Random_color_r, Random_color_g, Random_color_b;
			Random_color_r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
			Random_color_g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
			Random_color_b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
			pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
			temp_cloud = onenode->cloud_;
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
	/*	cloud_after->width = cloud_after->points.size();
		cloud_after->height = 1;
		cloud_after->is_dense = true;
		stringstream ss;
		ss << "类间聚类后的全部厚度" << ".pcd";
		pcl::PCDWriter writer;
		writer.write<pcl::PointXYZRGB>(ss.str(), *cloud_after, false);*/

	}
	
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> float
thickness::Cluster<PointT>::calCenterDis(Node* A, Node* B)
{
	pcl::PointXYZ center_a, center_b;
	center_a = A->center_;
	center_b = B->center_;
	float dis = pcl::euclideanDistance(center_a, center_b);
	return dis;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PoinT> float
thickness::Cluster<PoinT>::calPairDis(std::pair<int, int> seed_pair)
{
	pcl::PointXYZ p1, p2;
	p1.x = margin_A_->points[seed_pair.first].x;
	p1.y = margin_A_->points[seed_pair.first].y;
	p1.z = margin_A_->points[seed_pair.first].z;
	p2.x = margin_B_->points[seed_pair.second].x;
	p2.y = margin_B_->points[seed_pair.second].y;
	p2.z = margin_B_->points[seed_pair.second].z;
	float dis = pcl::euclideanDistance(p1, p2);
	return dis;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PoinT> float
thickness::Cluster<PoinT>::calPairDisSelf(std::pair<int, int> seed_pair)
{
	pcl::PointXYZ p1, p2;
	p1.x = margin_C_->points[seed_pair.first].x;
	p1.y = margin_C_->points[seed_pair.first].y;
	p1.z = margin_C_->points[seed_pair.first].z;
	p2.x = margin_C_->points[seed_pair.second].x;
	p2.y = margin_C_->points[seed_pair.second].y;
	p2.z = margin_C_->points[seed_pair.second].z;
	float dis = pcl::euclideanDistance(p1, p2);
	return dis;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PoinT> float
thickness::Cluster<PoinT>::calThicknessDis(Node* A, Node* B)
{
	int num_a, num_b;
	float sum_a, sum_b;
	num_a = A->node_thickness_index_.size();
	num_b = B->node_thickness_index_.size();
	sum_a = 0;
	sum_b = 0;

	//计算A的平均厚度
	for (int index = 0; index < static_cast<int>(A->node_thickness_index_.size()); index++)
	{
		pair<int, int> one_pair;
		one_pair.first = A->node_thickness_index_[index].first;
		one_pair.second = A->node_thickness_index_[index].second;
		float cal_res_a = calPairDis(one_pair);
		sum_a += cal_res_a;
	}
	float avg_thickness_res_a = sum_a / static_cast<float>(num_a);

	//计算B的平均厚度
	for (int index = 0; index < static_cast<int>(B->node_thickness_index_.size()); index++)
	{
		pair<int, int> one_pair;
		one_pair.first = B->node_thickness_index_[index].first;
		one_pair.second = B->node_thickness_index_[index].second;
		float cal_res_b = calPairDis(one_pair);
		sum_b += cal_res_b;
	}
	float avg_thickness_res_b = sum_b / static_cast<float>(num_b);
	float dis = abs(avg_thickness_res_a - avg_thickness_res_b);
	return dis;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PoinT> float
thickness::Cluster<PoinT>::calThicknessDisSelf(Node* A, Node* B)
{
	int num_a, num_b;
	float sum_a, sum_b;
	num_a = A->node_thickness_index_.size();
	num_b = B->node_thickness_index_.size();
	sum_a = 0;
	sum_b = 0;

	//计算A的平均厚度
	for (int index = 0; index < static_cast<int>(A->node_thickness_index_.size()); index++)
	{
		pair<int, int> one_pair;
		one_pair.first = A->node_thickness_index_[index].first;
		one_pair.second = A->node_thickness_index_[index].second;
		float cal_res_a = calPairDisSelf(one_pair);
		sum_a += cal_res_a;
	}
	float avg_thickness_res_a = sum_a / static_cast<float>(num_a);

	//计算B的平均厚度
	for (int index = 0; index < static_cast<int>(B->node_thickness_index_.size()); index++)
	{
		pair<int, int> one_pair;
		one_pair.first = B->node_thickness_index_[index].first;
		one_pair.second = B->node_thickness_index_[index].second;
		float cal_res_b = calPairDisSelf(one_pair);
		sum_b += cal_res_b;
	}
	float avg_thickness_res_b = sum_b / static_cast<float>(num_b);
	float dis = abs(avg_thickness_res_a - avg_thickness_res_b);
	return dis;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PoinT> float
thickness::Cluster<PoinT>::calFitLineDis(Node* A, Node* B)
{
	//求a和b的中点所在的点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr a_mid_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr b_mid_cloud(new pcl::PointCloud<pcl::PointXYZ>);

	for (int index = 0; index < static_cast<int>(A->node_thickness_index_.size()); index++)
	{
		int a = A->node_thickness_index_[index].first;
		int b = A->node_thickness_index_[index].second;
		pcl::PointXYZ point_mid;
		point_mid.x = (margin_A_->points[a].x + margin_B_->points[b].x) / 2;
		point_mid.y = (margin_A_->points[a].y + margin_B_->points[b].y) / 2;
		point_mid.z = (margin_A_->points[a].z + margin_B_->points[b].z) / 2;
		a_mid_cloud->push_back(point_mid);
		
	}

	for (int index = 0; index < static_cast<int>(B->node_thickness_index_.size()); index++)
	{
		int a = B->node_thickness_index_[index].first;
		int b = B->node_thickness_index_[index].second;
		pcl::PointXYZ point_mid;
		point_mid.x = (margin_A_->points[a].x + margin_B_->points[b].x) / 2;
		point_mid.y = (margin_A_->points[a].y + margin_B_->points[b].y) / 2;
		point_mid.z = (margin_A_->points[a].z + margin_B_->points[b].z) / 2;
		b_mid_cloud->push_back(point_mid);
	}

	/*a_mid_cloud->width = a_mid_cloud->points.size();
	a_mid_cloud->height = 1;
	a_mid_cloud->is_dense = true;
	stringstream ss;
	ss << A->cluster_id_ << ".pcd";
	pcl::PCDWriter writer;
	writer.write<pcl::PointXYZ>(ss.str(), *a_mid_cloud, false);*/
	

	pcl::PCA<pcl::PointXYZ> pca_a;
	pca_a.setInputCloud(a_mid_cloud);
	Eigen::RowVector3f a_V1 = pca_a.getEigenVectors().col(0);

	pcl::PCA<pcl::PointXYZ> pca_b;
	pca_b.setInputCloud(b_mid_cloud);
	Eigen::RowVector3f b_V1 = pca_b.getEigenVectors().col(0);

	//测量距离
	Eigen::Vector4f a_line_point;
	Eigen::Vector4f a_line_dir;
	Eigen::Vector4f b_line_point;
	Eigen::Vector4f b_line_dir;
	Eigen::Vector4f temp_point;
	int num_a, num_b;
	float sum_a, sum_b;
	sum_a = 0;
	sum_b = 0;
	num_a = a_mid_cloud->size();
	num_b = b_mid_cloud->size();

	a_line_point = pca_a.getMean();
	a_line_dir[0] = a_V1[0];
	a_line_dir[1] = a_V1[1];
	a_line_dir[2] = a_V1[2];
	a_line_dir[3] = 0;

	b_line_point = pca_b.getMean();
	b_line_dir[0] = b_V1[0];
	b_line_dir[1] = b_V1[1];
	b_line_dir[2] = b_V1[2];
	b_line_dir[3] = 0;

	for (auto p : a_mid_cloud->points)
	{
		temp_point[0] = p.x;
		temp_point[1] = p.y;
		temp_point[2] = p.z;
		temp_point[3] = 0;
		double dis_p_line = sqrt(pcl::sqrPointToLineDistance(temp_point, b_line_point, b_line_dir));
		sum_a += dis_p_line;
	}
	float avg_a_fit_dis = sum_a / static_cast<float>(num_a);

	for (auto p : b_mid_cloud->points)
	{
		temp_point[0] = p.x;
		temp_point[1] = p.y;
		temp_point[2] = p.z;
		temp_point[3] = 0;
		double dis_p_line = sqrt(pcl::sqrPointToLineDistance(temp_point, a_line_point, a_line_dir));
		sum_b += dis_p_line;
	}
	float avg_b_fit_dis = sum_b / static_cast<float>(num_b);
	//求夹角
	float ang_ab = pcl::getAngle3D(a_line_dir, b_line_dir, false);
	float dis = (avg_b_fit_dis + avg_a_fit_dis) / 2;
	if (ang_ab > M_PI_2)
	{
		ang_ab = ang_ab - M_PI_2;
	}
	if (ang_ab > M_PI / 9)
	{
		dis += 10000000;
	}
	//求端点距离
	//float dis1, dis2, dis3, dis4;
	//dis1 = pcl::euclideanDistance(A->pmin_, B->pmin_);
	//dis2 = pcl::euclideanDistance(A->pmax_, B->pmax_);
	//dis3 = pcl::euclideanDistance(A->pmin_, B->pmax_);
	//dis4 = pcl::euclideanDistance(A->pmax_, B->pmin_);
	//if (dis1 > th_fit_dis && dis2 > th_fit_dis && dis3 > th_fit_dis && dis4 > th_fit_dis)
	//{
	//	dis += 10000000;
	//}
	return dis;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PoinT> float
thickness::Cluster<PoinT>::calFitLineDisSelf(Node* A, Node* B)
{
	//求a和b的中点所在的点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr a_mid_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr b_mid_cloud(new pcl::PointCloud<pcl::PointXYZ>);

	for (int index = 0; index < static_cast<int>(A->node_thickness_index_.size()); index++)
	{
		int a = A->node_thickness_index_[index].first;
		int b = A->node_thickness_index_[index].second;
		pcl::PointXYZ point_mid;
		point_mid.x = (margin_C_->points[a].x + margin_C_->points[b].x) / 2;
		point_mid.y = (margin_C_->points[a].y + margin_C_->points[b].y) / 2;
		point_mid.z = (margin_C_->points[a].z + margin_C_->points[b].z) / 2;
		a_mid_cloud->push_back(point_mid);

	}

	for (int index = 0; index < static_cast<int>(B->node_thickness_index_.size()); index++)
	{
		int a = B->node_thickness_index_[index].first;
		int b = B->node_thickness_index_[index].second;
		pcl::PointXYZ point_mid;
		point_mid.x = (margin_C_->points[a].x + margin_C_->points[b].x) / 2;
		point_mid.y = (margin_C_->points[a].y + margin_C_->points[b].y) / 2;
		point_mid.z = (margin_C_->points[a].z + margin_C_->points[b].z) / 2;
		b_mid_cloud->push_back(point_mid);
	}

	/*a_mid_cloud->width = a_mid_cloud->points.size();
	a_mid_cloud->height = 1;
	a_mid_cloud->is_dense = true;
	stringstream ss;
	ss << A->cluster_id_ << ".pcd";
	pcl::PCDWriter writer;
	writer.write<pcl::PointXYZ>(ss.str(), *a_mid_cloud, false);*/


	pcl::PCA<pcl::PointXYZ> pca_a;
	pca_a.setInputCloud(a_mid_cloud);
	Eigen::RowVector3f a_V1 = pca_a.getEigenVectors().col(0);

	pcl::PCA<pcl::PointXYZ> pca_b;
	pca_b.setInputCloud(b_mid_cloud);
	Eigen::RowVector3f b_V1 = pca_b.getEigenVectors().col(0);

	//测量距离
	Eigen::Vector4f a_line_point;
	Eigen::Vector4f a_line_dir;
	Eigen::Vector4f b_line_point;
	Eigen::Vector4f b_line_dir;
	Eigen::Vector4f temp_point;
	int num_a, num_b;
	float sum_a, sum_b;
	sum_a = 0;
	sum_b = 0;
	num_a = a_mid_cloud->size();
	num_b = b_mid_cloud->size();

	a_line_point = pca_a.getMean();
	a_line_dir[0] = a_V1[0];
	a_line_dir[1] = a_V1[1];
	a_line_dir[2] = a_V1[2];
	a_line_dir[3] = 0;

	b_line_point = pca_b.getMean();
	b_line_dir[0] = b_V1[0];
	b_line_dir[1] = b_V1[1];
	b_line_dir[2] = b_V1[2];
	b_line_dir[3] = 0;

	for (auto p : a_mid_cloud->points)
	{
		temp_point[0] = p.x;
		temp_point[1] = p.y;
		temp_point[2] = p.z;
		temp_point[3] = 0;
		double dis_p_line = sqrt(pcl::sqrPointToLineDistance(temp_point, b_line_point, b_line_dir));
		sum_a += dis_p_line;
	}
	float avg_a_fit_dis = sum_a / static_cast<float>(num_a);

	for (auto p : b_mid_cloud->points)
	{
		temp_point[0] = p.x;
		temp_point[1] = p.y;
		temp_point[2] = p.z;
		temp_point[3] = 0;
		double dis_p_line = sqrt(pcl::sqrPointToLineDistance(temp_point, a_line_point, a_line_dir));
		sum_b += dis_p_line;
	}
	float avg_b_fit_dis = sum_b / static_cast<float>(num_b);

	float ang_ab = pcl::getAngle3D(a_line_dir, b_line_dir, false);
	float dis = (avg_b_fit_dis + avg_a_fit_dis) / 2;
	if (ang_ab > M_PI_2)
	{
		ang_ab = ang_ab - M_PI_2;
	}
	if (ang_ab > M_PI / 9)
	{
		dis += 10000000;
	}

	//求端点距离
	/*float dis1, dis2, dis3, dis4;
	dis1 = pcl::euclideanDistance(A->pmin_, B->pmin_);
	dis2 = pcl::euclideanDistance(A->pmax_, B->pmax_);
	dis3 = pcl::euclideanDistance(A->pmin_, B->pmax_);
	dis4 = pcl::euclideanDistance(A->pmax_, B->pmin_);
	if (dis1 > th_fit_dis && dis2 > th_fit_dis && dis3 > th_fit_dis && dis4 > th_fit_dis)
	{
		dis += 10000000;
	}*/
	return dis;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> float
thickness::Cluster<PointT>::calFitCricleDis_classBetween(Node* A, Node* B)
{
	std::vector<std::pair<int, int>> A_thickness_index = A->node_thickness_index_;
	std::vector<std::pair<int, int>> B_thickness_index = B->node_thickness_index_;
	//求A和B的质心
	pcl::PointXYZ AB_center;
	AB_center.x = ((A->center_.x) * A_thickness_index.size() * 2 + (B->center_.x) * B_thickness_index.size() * 2) / (float)(A_thickness_index.size() * 2 + B_thickness_index.size() * 2);
	AB_center.y = ((A->center_.y) * A_thickness_index.size() * 2 + (B->center_.y) * B_thickness_index.size() * 2) / (float)(A_thickness_index.size() * 2 + B_thickness_index.size() * 2);
	AB_center.z = ((A->center_.z) * A_thickness_index.size() * 2 + (B->center_.z) * B_thickness_index.size() * 2) / (float)(A_thickness_index.size() * 2 + B_thickness_index.size() * 2);
	//提取NodeA厚度对每一个点
	std::vector<pcl::PointXYZ> A_p;
	std::vector<pcl::PointXYZ> A_q;
	for (std::vector<std::pair<int, int>>::const_iterator pit = A_thickness_index.begin(); pit != A_thickness_index.end(); ++pit)
	{
		pcl::PointXYZ p1, q1;
		p1.x = margin_A_->points[(*pit).first].x;
		p1.y = margin_A_->points[(*pit).first].y;
		p1.z = margin_A_->points[(*pit).first].z;
		q1.x = margin_B_->points[(*pit).second].x;
		q1.y = margin_B_->points[(*pit).second].y;
		q1.z = margin_B_->points[(*pit).second].z;
		A_p.push_back(p1);
		A_q.push_back(q1);
	}
	//计算质心到NodeA所有直线的距离
	float sum_AB = 0;
	for (int i = 0; i < A_p.size(); i++)
	{
		float normal01_x = (float)A_q[i].x - (float)A_p[i].x;
		float normal01_y = (float)A_q[i].y - (float)A_p[i].y;
		float normal01_z = (float)A_q[i].z - (float)A_p[i].z;
		float normal02_x = (float)AB_center.x - (float)A_p[i].x;
		float normal02_y = (float)AB_center.y - (float)A_p[i].y;
		float normal02_z = (float)AB_center.z - (float)A_p[i].z;
		float fenzi = normal01_x * normal02_x + normal01_y * normal02_y + normal01_z * normal02_z;
		float lengthN1 = sqrt(normal01_x * normal01_x + normal01_y * normal01_y + normal01_z * normal01_z);
		float lengthN2 = sqrt(normal02_x * normal02_x + normal02_y * normal02_y + normal02_z * normal02_z);
		float hudu = acos(fenzi / (lengthN1 * lengthN2));
		float ds = abs(lengthN2 * sin(hudu));
		sum_AB += ds;
	}
	//提取NodeB厚度对每一个点
	std::vector<pcl::PointXYZ> B_p;
	std::vector<pcl::PointXYZ> B_q;
	for (std::vector<std::pair<int, int>>::const_iterator pit = B_thickness_index.begin(); pit != B_thickness_index.end(); ++pit)
	{
		pcl::PointXYZ p1, q1;
		p1.x = margin_A_->points[(*pit).first].x;
		p1.y = margin_A_->points[(*pit).first].y;
		p1.z = margin_A_->points[(*pit).first].z;
		q1.x = margin_B_->points[(*pit).second].x;
		q1.y = margin_B_->points[(*pit).second].y;
		q1.z = margin_B_->points[(*pit).second].z;
		B_p.push_back(p1);
		B_q.push_back(q1);
	}
	//计算质心到NodeB所有直线的距离		
	for (int i = 0; i < B_p.size(); i++)
	{
		float normal01_x = (float)B_q[i].x - (float)B_p[i].x;
		float normal01_y = (float)B_q[i].y - (float)B_p[i].y;
		float normal01_z = (float)B_q[i].z - (float)B_p[i].z;
		float normal02_x = (float)AB_center.x - (float)B_p[i].x;
		float normal02_y = (float)AB_center.y - (float)B_p[i].y;
		float normal02_z = (float)AB_center.z - (float)B_p[i].z;
		float fenzi = normal01_x * normal02_x + normal01_y * normal02_y + normal01_z * normal02_z;
		float lengthN1 = sqrt(normal01_x * normal01_x + normal01_y * normal01_y + normal01_z * normal01_z);
		float lengthN2 = sqrt(normal02_x * normal02_x + normal02_y * normal02_y + normal02_z * normal02_z);
		float hudu = acos(fenzi / (lengthN1 * lengthN2));
		float ds = abs(lengthN2 * sin(hudu));
		sum_AB += ds;
	}
	float avg_distance;
	avg_distance = (float)sum_AB / (float)(A_p.size() + B_p.size());

	//求端点距离
	/*float dis1, dis2, dis3, dis4;
	dis1 = pcl::euclideanDistance(A->pmin_, B->pmin_);
	dis2 = pcl::euclideanDistance(A->pmax_, B->pmax_);
	dis3 = pcl::euclideanDistance(A->pmin_, B->pmax_);
	dis4 = pcl::euclideanDistance(A->pmax_, B->pmin_);
	if (dis1 > th_fit_dis && dis2 > th_fit_dis && dis3 > th_fit_dis && dis4 > th_fit_dis)
	{
		avg_distance += 10000000;
	}*/
	return avg_distance;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PointT> float
thickness::Cluster<PointT>::calFitCricleDis_classWithin(Node* A, Node* B)
{
	std::vector<std::pair<int, int>> A_thickness_index = A->node_thickness_index_;
	std::vector<std::pair<int, int>> B_thickness_index = B->node_thickness_index_;
	//求A和B的质心
	pcl::PointXYZ AB_center;
	AB_center.x = ((A->center_.x) * A_thickness_index.size() * 2 + (B->center_.x) * B_thickness_index.size() * 2) / (float)(A_thickness_index.size() * 2 + B_thickness_index.size() * 2);
	AB_center.y = ((A->center_.y) * A_thickness_index.size() * 2 + (B->center_.y) * B_thickness_index.size() * 2) / (float)(A_thickness_index.size() * 2 + B_thickness_index.size() * 2);
	AB_center.z = ((A->center_.z) * A_thickness_index.size() * 2 + (B->center_.z) * B_thickness_index.size() * 2) / (float)(A_thickness_index.size() * 2 + B_thickness_index.size() * 2);
	//提取NodeA厚度对每一个点
	std::vector<pcl::PointXYZ> A_p;
	std::vector<pcl::PointXYZ> A_q;
	for (std::vector<std::pair<int, int>>::const_iterator pit = A_thickness_index.begin(); pit != A_thickness_index.end(); ++pit)
	{
		pcl::PointXYZ p1, q1;
		p1.x = margin_C_->points[(*pit).first].x;
		p1.y = margin_C_->points[(*pit).first].y;
		p1.z = margin_C_->points[(*pit).first].z;
		q1.x = margin_C_->points[(*pit).second].x;
		q1.y = margin_C_->points[(*pit).second].y;
		q1.z = margin_C_->points[(*pit).second].z;
		A_p.push_back(p1);
		A_q.push_back(q1);
	}
	//计算质心到NodeA所有直线的距离
	float sum_AB = 0;
	for (int i = 0; i < A_p.size(); i++)
	{
		float normal01_x = (float)A_q[i].x - (float)A_p[i].x;
		float normal01_y = (float)A_q[i].y - (float)A_p[i].y;
		float normal01_z = (float)A_q[i].z - (float)A_p[i].z;
		float normal02_x = (float)AB_center.x - (float)A_p[i].x;
		float normal02_y = (float)AB_center.y - (float)A_p[i].y;
		float normal02_z = (float)AB_center.z - (float)A_p[i].z;
		float fenzi = normal01_x * normal02_x + normal01_y * normal02_y + normal01_z * normal02_z;
		float lengthN1 = sqrt(normal01_x * normal01_x + normal01_y * normal01_y + normal01_z * normal01_z);
		float lengthN2 = sqrt(normal02_x * normal02_x + normal02_y * normal02_y + normal02_z * normal02_z);
		float hudu = acos(fenzi / (lengthN1 * lengthN2));
		float ds = abs(lengthN2 * sin(hudu));
		sum_AB += ds;
	}
	//提取NodeB厚度对每一个点
	std::vector<pcl::PointXYZ> B_p;
	std::vector<pcl::PointXYZ> B_q;
	for (std::vector<std::pair<int, int>>::const_iterator pit = B_thickness_index.begin(); pit != B_thickness_index.end(); ++pit)
	{
		pcl::PointXYZ p1, q1;
		p1.x = margin_C_->points[(*pit).first].x;
		p1.y = margin_C_->points[(*pit).first].y;
		p1.z = margin_C_->points[(*pit).first].z;
		q1.x = margin_C_->points[(*pit).second].x;
		q1.y = margin_C_->points[(*pit).second].y;
		q1.z = margin_C_->points[(*pit).second].z;
		B_p.push_back(p1);
		B_q.push_back(q1);
	}
	//计算质心到NodeB所有直线的距离		
	for (int i = 0; i < B_p.size(); i++)
	{
		float normal01_x = (float)B_q[i].x - (float)B_p[i].x;
		float normal01_y = (float)B_q[i].y - (float)B_p[i].y;
		float normal01_z = (float)B_q[i].z - (float)B_p[i].z;
		float normal02_x = (float)AB_center.x - (float)B_p[i].x;
		float normal02_y = (float)AB_center.y - (float)B_p[i].y;
		float normal02_z = (float)AB_center.z - (float)B_p[i].z;
		float fenzi = normal01_x * normal02_x + normal01_y * normal02_y + normal01_z * normal02_z;
		float lengthN1 = sqrt(normal01_x * normal01_x + normal01_y * normal01_y + normal01_z * normal01_z);
		float lengthN2 = sqrt(normal02_x * normal02_x + normal02_y * normal02_y + normal02_z * normal02_z);
		float hudu = acos(fenzi / (lengthN1 * lengthN2));
		float ds = abs(lengthN2 * sin(hudu));
		sum_AB += ds;
	}
	float avg_distance;
	avg_distance = (float)sum_AB / (float)(A_p.size() + B_p.size());


	//求端点距离
	//float dis1, dis2, dis3, dis4;
	//dis1 = pcl::euclideanDistance(A->pmin_, B->pmin_);
	//dis2 = pcl::euclideanDistance(A->pmax_, B->pmax_);
	//dis3 = pcl::euclideanDistance(A->pmin_, B->pmax_);
	//dis4 = pcl::euclideanDistance(A->pmax_, B->pmin_);
	//if (dis1 > th_fit_dis && dis2 > th_fit_dis && dis3 > th_fit_dis && dis4 > th_fit_dis)
	//{
	//	avg_distance += 1000000;
	//}
	return avg_distance;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PoinT> float
thickness::Cluster<PoinT>::calLineWeightingDis(float cen_dis, float fit_dis, float hou_dis)
{
	float add_weight_dis = sqrt(line_w1_ * (cen_dis * cen_dis) + line_w2_ * (fit_dis * fit_dis) + line_w3_ * (hou_dis * hou_dis));
	float dis = exp(-add_weight_dis);
	return dis;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PoinT> float
thickness::Cluster<PoinT>::calCricleWeightingDis(float cen_dis, float cri_dis, float hou_dis)
{
	float add_weight_dis = sqrt(cricle_w1_ * (cen_dis * cen_dis) + cricle_w2_ * (cri_dis * cri_dis) + cricle_w3_ * (hou_dis * hou_dis));
	float dis = exp(-add_weight_dis);
	return dis;
}