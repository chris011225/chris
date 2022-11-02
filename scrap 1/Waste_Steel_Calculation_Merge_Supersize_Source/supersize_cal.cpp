#pragma once
#include "supersize_cal.h"
#include<pcl/common/distances.h>
#include <pcl/common/common.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/features/moment_of_inertia_estimation.h>
using namespace std;
using namespace pcl;
using namespace Eigen;
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PoinT>
supersize::SuperSize<PoinT>::SuperSize() :
	m_angle_threshold_(18),
	m_min_segment_size_(0),
	m_sum_segment_(0)
{
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PoinT>
supersize::SuperSize<PoinT>::~SuperSize()
{



}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PoinT> void
supersize::SuperSize<PoinT>::reset()
{
	m_processed_.clear();
	m_sv_adjacency_list_.clear();
	m_sv_label_to_supervoxel_map_.clear();
	m_sv_label_to_seg_label_map_.clear();
	m_seg_label_to_neighbor_set_map_.clear();
	m_seg_label_to_sv_list_map_.clear();
	m_seg_label_to_bou_list_map_.clear();
	m_seg_label_to_neighbor_bou_label_list_map_.clear();
	m_sv_label_to_seg_label_map_.clear();
	m_supervoxels_set_ = false;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename PoinT> void
supersize::SuperSize<PoinT>::compute_segment_adjacency()
{
	m_seg_label_to_neighbor_set_map_.clear();
	std::uint32_t current_segLabel;
	std::uint32_t neigh_segLabel;

	VertexIterator sv_itr, sv_itr_end;
	//查找当前seglabel的邻域的seg
	for (std::tie(sv_itr, sv_itr_end) = boost::vertices(m_sv_adjacency_list_); sv_itr != sv_itr_end; ++sv_itr)
	{
		const std::uint32_t& sv_label = m_sv_adjacency_list_[*sv_itr];
		current_segLabel = m_sv_label_to_seg_label_map_[sv_label];

		AdjacencyIterator itr_neighbor, itr_neighbor_end;

		for (std::tie(itr_neighbor, itr_neighbor_end) = boost::adjacent_vertices(*sv_itr, m_sv_adjacency_list_); itr_neighbor != itr_neighbor_end; ++itr_neighbor)
		{
			const std::uint32_t& neigh_label = m_sv_adjacency_list_[*itr_neighbor];
			neigh_segLabel = m_sv_label_to_seg_label_map_[neigh_label];

			if (current_segLabel != neigh_segLabel)
			{
				m_seg_label_to_neighbor_set_map_[current_segLabel].insert(neigh_segLabel);
			}
		}
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PoinT> std::set<std::uint32_t>
supersize::SuperSize<PoinT>::computer_two_seg_bou(const std::uint32_t& seg_a, const std::uint32_t& seg_b)
{
	std::set<std::uint32_t> bou_set;
	for (auto v: all_bou_v_)
	{
		for (auto seg_itr = m_seg_label_to_bou_list_map_[seg_a].begin(); seg_itr != m_seg_label_to_bou_list_map_[seg_a].end(); ++seg_itr)
		{
			if (m_sv_adjacency_list_[v] == *seg_itr)
			{
				AdjacencyIterator itr_neighbor, itr_neighbor_end;

				for (std::tie(itr_neighbor, itr_neighbor_end) = boost::adjacent_vertices(v, m_sv_adjacency_list_); itr_neighbor != itr_neighbor_end; ++itr_neighbor)
				{
					for (auto seg_itr_ne = m_seg_label_to_bou_list_map_[seg_b].begin(); seg_itr_ne != m_seg_label_to_bou_list_map_[seg_b].end(); ++seg_itr_ne)
					{
						if (m_sv_adjacency_list_[*itr_neighbor] == *seg_itr_ne)
						{
							bou_set.insert(*seg_itr);
						}
					}
				}
			}
		}
	}
	return bou_set;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PoinT> void
supersize::SuperSize<PoinT>::compute_segments_public_boundary()
{
	compute_segment_adjacency();
	m_seg_label_to_neighbor_bou_label_list_map_.clear();
	std::map<std::uint32_t, std::set<std::uint32_t>> one_seg_bou;

	for (auto segment_itr = m_seg_label_to_neighbor_set_map_.begin(); segment_itr != m_seg_label_to_neighbor_set_map_.end(); ++segment_itr++)
	{
		const std::uint32_t& segment_label = segment_itr->first;
		if (segment_itr->second.size() == 0)
		{
			continue;
		}

		//遍历邻接的聚类，查找当前聚类在邻接聚类内的边界
		for (auto seg_neighbor_itr = segment_itr->second.begin(); seg_neighbor_itr != segment_itr->second.end(); ++seg_neighbor_itr)
		{
			std::set<std::uint32_t> bou_indexs;
			const std::uint32_t& segment_neighbor_label = *seg_neighbor_itr;
			bou_indexs = computer_two_seg_bou(segment_label, segment_neighbor_label);
			one_seg_bou[segment_neighbor_label] = bou_indexs;
		}
		m_seg_label_to_neighbor_bou_label_list_map_[segment_label] = one_seg_bou;
		one_seg_bou.clear();
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PoinT> void
supersize::SuperSize<PoinT>::compute_sgement_adjacencyList_weight()
{
	//分割后的邻接图初始化
	m_seg_adjacency_list_.clear();
	std::map<std::uint32_t, VertexID> label_ID_map;

	for (auto seg_itr = m_seg_label_to_sv_list_map_.begin(); seg_itr != m_seg_label_to_sv_list_map_.end(); ++seg_itr)
	{
		const std::uint32_t& seg_label = seg_itr->first;
		Seg_VertexID node_id = boost::add_vertex(m_seg_adjacency_list_);
		m_seg_adjacency_list_[node_id] = seg_label;
		label_ID_map[seg_label] = node_id;
	}

	for (auto seg_ner_itr = m_seg_label_to_neighbor_set_map_.begin(); seg_ner_itr != m_seg_label_to_neighbor_set_map_.end(); ++seg_ner_itr)
	{
		const std::uint32_t& seg_label = seg_ner_itr->first;
		for (auto seg_ners_itr = seg_ner_itr->second.begin(); seg_ners_itr != seg_ner_itr->second.end(); ++seg_ners_itr)
		{
			const std::uint32_t& neighbor_label = *seg_ners_itr;
			Seg_VertexID u = label_ID_map[seg_label];
			Seg_VertexID v = label_ID_map[neighbor_label];

			boost::add_edge(u, v, m_seg_adjacency_list_);
		}
	}
	//计算分割邻接图边的权重
	Seg_EdgeIterator edge_itr, edge_itr_end, next_edge;

	for (std::tie(edge_itr, edge_itr_end) = boost::edges(m_seg_adjacency_list_), next_edge = edge_itr; edge_itr != edge_itr_end; edge_itr = next_edge)
	{
		++next_edge;
		std::uint32_t source_seg_label = m_seg_adjacency_list_[boost::source(*edge_itr, m_seg_adjacency_list_)];
		std::uint32_t target_seg_label = m_seg_adjacency_list_[boost::target(*edge_itr, m_seg_adjacency_list_)];
		float W = cal_seg_edge_weight(source_seg_label, target_seg_label);
		m_seg_adjacency_list_[*edge_itr].Weight = W;
		

	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PoinT> void
supersize::SuperSize<PoinT>::prepare_segmentation(const std::map<std::uint32_t, SuperVoxelPtr>& supervoxel_clusters_arg, \
	                                              const std::multimap<std::uint32_t, std::uint32_t>& label_adjacency_arg)
{
	reset();
	m_sv_label_to_supervoxel_map_ = supervoxel_clusters_arg;
	std::map<std::uint32_t, VertexID> label_ID_map;

	//构建超体素邻接图，顶点为每个超体素的label
	for (typename std::map<std::uint32_t, SuperVoxelPtr>::iterator svlabel_itr = m_sv_label_to_supervoxel_map_.begin(); \
		svlabel_itr != m_sv_label_to_supervoxel_map_.end(); ++svlabel_itr)
	{
		const std::uint32_t& sv_label = svlabel_itr->first;
		VertexID node_id = boost::add_vertex(m_sv_adjacency_list_);
		m_sv_adjacency_list_[node_id] = sv_label;
		label_ID_map[sv_label] = node_id;
	}

	//加入邻接图的边
	for (const auto& sv_neighbors_itr:label_adjacency_arg)
	{
		const std::uint32_t& sv_label = sv_neighbors_itr.first;
		const std::uint32_t& neighbor_label = sv_neighbors_itr.second;

		VertexID u = label_ID_map[sv_label];
		VertexID v = label_ID_map[neighbor_label];

		boost::add_edge(u, v, m_sv_adjacency_list_);
	}

	//初始化
	m_seg_label_to_sv_list_map_.clear();
	for (typename std::map<std::uint32_t, SuperVoxelPtr>::iterator svlabel_itr = m_sv_label_to_supervoxel_map_.begin(); \
		svlabel_itr != m_sv_label_to_supervoxel_map_.end(); ++svlabel_itr)
	{
		const std::uint32_t& sv_label = svlabel_itr->first;
		m_processed_[sv_label] = false;
		m_sv_label_to_seg_label_map_[sv_label] = 0;//开始所有的超体素没有合并，所以对应的标签都是0标签
	}

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PoinT> void
supersize::SuperSize<PoinT>::segment()
{
	if (m_supervoxels_set_)
	{
		calculate_edges_isvalid(m_sv_adjacency_list_);//计算相邻超体素是否valid
		supersize_detection();
		compute_segments_public_boundary();
		compute_sgement_adjacencyList_weight();
		//一次
		merge_segments();
		compute_segments_public_boundary();
		compute_sgement_adjacencyList_weight();
		int num_segments = m_seg_label_to_sv_list_map_.size();
		int steps = 1;
		while (1)
		{
			merge_segments();
			cout << "第" << steps << "次迭代" << endl;
			++steps;
			int after_num_segment = m_seg_label_to_sv_list_map_.size();
			if (after_num_segment == num_segments)
			{
				break;
			}
			if (after_num_segment < num_segments)
			{
				num_segments = after_num_segment;
			}
			compute_segments_public_boundary();
			compute_sgement_adjacencyList_weight();
			
		}
		merge_small_segments();
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PoinT> void
supersize::SuperSize<PoinT>::merge_small_segments()
{
	compute_segment_adjacency();
	std::set<std::uint32_t> filteredSegLabels;

	bool continue_filtering = true;

	while (continue_filtering)
	{
		continue_filtering = false;
		unsigned int nr_filtered = 0;

		VertexIterator sv_itr, sv_itr_end;

		for (std::tie(sv_itr, sv_itr_end) = boost::vertices(m_sv_adjacency_list_); sv_itr != sv_itr_end; ++sv_itr)
		{
			const std::uint32_t& sv_label = m_sv_adjacency_list_[*sv_itr];
			std::uint32_t current_seg_label = m_sv_label_to_seg_label_map_[sv_label];
			std::uint32_t largest_neigh_seg_label = current_seg_label;
			std::uint32_t largest_neigh_size = m_seg_label_to_sv_list_map_[current_seg_label].size();

			const std::uint32_t& nr_neighbors = m_seg_label_to_neighbor_set_map_[current_seg_label].size();
			if (nr_neighbors == 0)
				continue;

			if (m_seg_label_to_sv_list_map_[current_seg_label].size() <= 10)//超参数
			{
				continue_filtering = true;
				nr_filtered++;

				for (auto neighbors_itr = m_seg_label_to_neighbor_set_map_[current_seg_label].cbegin(); neighbors_itr != m_seg_label_to_neighbor_set_map_[current_seg_label].cend(); ++neighbors_itr)
				{
					if (m_seg_label_to_sv_list_map_[*neighbors_itr].size() >= largest_neigh_size)
					{
						largest_neigh_seg_label = *neighbors_itr;
						largest_neigh_size = m_seg_label_to_sv_list_map_[*neighbors_itr].size();
					}
				}

				if (largest_neigh_seg_label != current_seg_label)
				{
					if (filteredSegLabels.count(largest_neigh_seg_label) > 0)
						continue;

					m_sv_label_to_seg_label_map_[sv_label] = largest_neigh_seg_label;
					filteredSegLabels.insert(current_seg_label);

					for (auto sv_ID_itr = m_seg_label_to_sv_list_map_[current_seg_label].cbegin(); sv_ID_itr != m_seg_label_to_sv_list_map_[current_seg_label].cend(); ++sv_ID_itr)
					{
						m_seg_label_to_sv_list_map_[largest_neigh_seg_label].insert(*sv_ID_itr);
					}
				}
			}
		}

		for (const unsigned int& filteredSegLabel : filteredSegLabels)
		{
			m_seg_label_to_sv_list_map_.erase(filteredSegLabel);
		}

		compute_segment_adjacency();
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PoinT> void
supersize::SuperSize<PoinT>::relabel_cloud(pcl::PointCloud<pcl::PointXYZL>& labeled_cloud_arg)
{
	for (auto& voxel : labeled_cloud_arg)
	{
		voxel.label = m_sv_label_to_seg_label_map_[voxel.label];
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PoinT> void
supersize::SuperSize<PoinT>::input_labeled_cloud(const pcl::PointCloud<pcl::PointXYZL>::ConstPtr labeled_cloud)
{
	m_sv_labeled_cloud = labeled_cloud;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PoinT> void
supersize::SuperSize<PoinT>::get_recolor_cloud(pcl::PointCloud<pcl::PointXYZL>& relabeled_cloud,pcl::PointCloud<pcl::PointXYZRGB>& recolor_cloud)
{
	for (auto ite = m_seg_label_to_sv_list_map_.begin(); ite != m_seg_label_to_sv_list_map_.end(); ite++)
	{
		int Random_color_r, Random_color_g, Random_color_b;
		Random_color_r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		pcl::PointXYZRGB temp;
		for (auto set_ite = ite->second.begin(); set_ite != ite->second.end(); set_ite++)
		{
			//for (int j = 0; j < m_sv_label_to_supervoxel_map_[(*set_ite)]->voxels_->size(); j++)//按颜色保存质心点云
			//{
			//	temp.x = m_sv_label_to_supervoxel_map_[(*set_ite)]->voxels_->points[j].x;
			//	temp.y = m_sv_label_to_supervoxel_map_[(*set_ite)]->voxels_->points[j].y;
			//	temp.z = m_sv_label_to_supervoxel_map_[(*set_ite)]->voxels_->points[j].z;
			//	temp.r = Random_color_r;
			//	temp.g = Random_color_g;
			//	temp.b = Random_color_b;
			//	//cout << temp.x << " " << temp.y << " " << temp.z << endl;
			//	recolor_cloud.push_back(temp);
			//}
		}

		for (auto point = relabeled_cloud.begin();point!= relabeled_cloud.end();point++)
		{
			if (point->label == ite->first)
			{
				temp.x = point->x;
				temp.y = point->y;
				temp.z = point->z;
				temp.r = Random_color_r;
				temp.g = Random_color_g;
				temp.b = Random_color_b;
				recolor_cloud.push_back(temp);
			}
		}
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PoinT> void 
supersize::SuperSize<PoinT>::get_color_bou_cloud(pcl::PointCloud<pcl::PointXYZL>& super_labeled_cloud, pcl::PointCloud<pcl::PointXYZRGB>& recolor_bou_cloud)
{
	for (auto ite = m_seg_label_to_bou_list_map_.begin(); ite != m_seg_label_to_bou_list_map_.end(); ite++)
	{
		int Random_color_r, Random_color_g, Random_color_b;
		Random_color_r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		Random_color_b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
		pcl::PointXYZRGB temp;

		for (auto point = super_labeled_cloud.begin(); point != super_labeled_cloud.end(); point++)
		{
			for (auto set_ite = ite->second.begin(); set_ite != ite->second.end(); set_ite++)
			{
				if (point->label == *set_ite)
				{
					temp.x = point->x;
					temp.y = point->y;
					temp.z = point->z;
					temp.r = Random_color_r;
					temp.g = Random_color_g;
					temp.b = Random_color_b;
					recolor_bou_cloud.push_back(temp);
				}
			}
			
		}

	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PoinT> void
supersize::SuperSize<PoinT>::get_color_public_bou_cloud(pcl::PointCloud<pcl::PointXYZL>& super_labeled_cloud, pcl::PointCloud<pcl::PointXYZRGB>& recolor_public_bou_cloud)
{
	for (auto ite = m_seg_label_to_neighbor_bou_label_list_map_.begin(); ite != m_seg_label_to_neighbor_bou_label_list_map_.end(); ++ite)
	{
		for (auto bou_ite = ite->second.begin(); bou_ite != ite->second.end(); ++bou_ite)
		{
			int Random_color_r, Random_color_g, Random_color_b;
			Random_color_r = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
			Random_color_g = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
			Random_color_b = 255 * (1024 * rand() / (RAND_MAX + 1.0f));
			pcl::PointXYZRGB temp;

			for (auto point = super_labeled_cloud.begin(); point != super_labeled_cloud.end(); point++)
			{
				for (auto set_ite = bou_ite->second.begin(); set_ite != bou_ite->second.end(); set_ite++)
				{
					if (point->label == *set_ite)
					{
						temp.x = point->x;
						temp.y = point->y;
						temp.z = point->z;
						temp.r = Random_color_r;
						temp.g = Random_color_g;
						temp.b = Random_color_b;
						recolor_public_bou_cloud.push_back(temp);
					}
				}
			}
		}
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PoinT> void
supersize::SuperSize<PoinT>::cal_supersize_properties(pcl::PointCloud<pcl::PointXYZL>& relabeled_cloud)
{
	int id = 0;
	for (auto ite = m_seg_label_to_sv_list_map_.begin(); ite != m_seg_label_to_sv_list_map_.end(); ite++)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointXYZ point_temp;
		pcl::PointXYZ aabb_max, aabb_min, obb_max, obb_min;
		float first_size, second_size, third_size;
		//声明OBB position_OBB
		pcl::PointXYZ position_OBB;
		//声明旋转矩阵rotational_matrix_OBB
		Eigen::Matrix3f rotational_matrix_OBB;

		for (auto point = relabeled_cloud.begin(); point != relabeled_cloud.end(); point++)
		{
			if (point->label == ite->first)
			{
				point_temp.x = point->x;
				point_temp.y = point->y;
				point_temp.z = point->z;
				cloud_temp->push_back(point_temp);
				//relabeled_cloud.erase(point);
			}
		}

		//pcl::io::savePCDFile("hh"+to_string(id)+".pcd", *cloud_temp);
		pcl::MomentOfInertiaEstimation<pcl::PointXYZ> feature_extractor;
		feature_extractor.setInputCloud(cloud_temp);
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

		supersize_information temp_information;
		temp_information.aabb_min = aabb_min;
		temp_information.aabb_max = aabb_max;
		temp_information.obb_min = obb_min;
		temp_information.obb_max = obb_max;
		temp_information.first_size = first_size*1000;
		temp_information.second_size = second_size*1000;
		temp_information.third_size = third_size*1000;
		temp_information.area = temp_information.first_size * temp_information.second_size;
		m_supersizes_properties.insert(std::pair<std::uint32_t, supersize_information>(ite->first, temp_information));
		cloud_temp->clear();
		id++;
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PoinT> void
supersize::SuperSize<PoinT>::calculate_edges_isvalid(SupervoxelAdjacencyList& adjacency_list_arg)
{
	EdgeIterator edge_itr, edge_itr_end, next_edge;

	for (std::tie(edge_itr, edge_itr_end) = boost::edges(adjacency_list_arg), next_edge = edge_itr; edge_itr != edge_itr_end; edge_itr = next_edge)
	{
		++next_edge;  
		std::uint32_t source_sv_label = adjacency_list_arg[boost::source(*edge_itr, adjacency_list_arg)];
		std::uint32_t target_sv_label = adjacency_list_arg[boost::target(*edge_itr, adjacency_list_arg)];
		float normal_difference, centroid_dis;
		bool is_valid = edge_is_valid(source_sv_label, target_sv_label, normal_difference, centroid_dis);
		adjacency_list_arg[*edge_itr].is_valid = is_valid;
		adjacency_list_arg[*edge_itr].normal_difference = normal_difference;
		adjacency_list_arg[*edge_itr].centroid_dis = centroid_dis;
	}
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PoinT> void
supersize::SuperSize<PoinT>::supersize_detection()
{
	m_seg_label_to_sv_list_map_.clear();
	m_seg_label_to_bou_list_map_.clear();
	int sum_sv = 0;
	int sum_sv_seg = 0;

	for (typename std::map<std::uint32_t, SuperVoxelPtr>::iterator svlabel_itr = m_sv_label_to_supervoxel_map_.begin(); \
		svlabel_itr != m_sv_label_to_supervoxel_map_.end(); ++svlabel_itr)
	{
		const std::uint32_t& sv_label = svlabel_itr->first;
		m_processed_[sv_label] = false;
		m_sv_label_to_seg_label_map_[sv_label] = 0;
		sum_sv++;
	}

	VertexIterator sv_itr, sv_itr_end;
	unsigned int segment_label = 1;

	for (std::tie(sv_itr, sv_itr_end) = boost::vertices(m_sv_adjacency_list_); sv_itr != sv_itr_end; ++sv_itr)
	{
		const VertexID sv_vertex_id = *sv_itr;
		const std::uint32_t& sv_label = m_sv_adjacency_list_[sv_vertex_id];
		std::set<VertexID> temp_set;

		if (!m_processed_[sv_label])
		{
			int one_seg_num = segment_growing(sv_vertex_id, segment_label, temp_set);
			segment_label++;
			sum_sv_seg += one_seg_num;
			for (auto v : temp_set)
			{
				all_bou_v_.insert(v);
			}
			temp_set.clear();
		}
	}

	m_sum_segment_ = segment_label-1;//测试用
	


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PoinT> int
supersize::SuperSize<PoinT>::segment_growing(const VertexID& query_pointid, const unsigned int group_label, std::set<VertexID>& temp_v)
{
	const std::uint32_t& sv_label = m_sv_adjacency_list_[query_pointid];
	std::queue<VertexID> seeds;
	std::set<VertexID> bou_v, plane_v;
	m_processed_[sv_label] = true;//当前的超体素被访问了
	seeds.push(query_pointid);
	int num_sv_in_segment = 1;
	m_sv_label_to_seg_label_map_[sv_label] = group_label;
	m_seg_label_to_sv_list_map_[group_label].insert(sv_label);
	plane_v.insert(query_pointid);

	while (!seeds.empty())
	{
		VertexID curr_pointid = seeds.front();
		seeds.pop();
		OutEdgeIterator out_Edge_itr, out_Edge_itr_end;
		for (std::tie(out_Edge_itr, out_Edge_itr_end) = boost::out_edges(curr_pointid, m_sv_adjacency_list_); out_Edge_itr != out_Edge_itr_end; ++out_Edge_itr)
		{
			const VertexID neighbor_ID = boost::target(*out_Edge_itr, m_sv_adjacency_list_);
			const std::uint32_t& neighbor_label = m_sv_adjacency_list_[neighbor_ID];

			if (!m_processed_[neighbor_label])
			{
				if (m_sv_adjacency_list_[*out_Edge_itr].is_valid)
				{
					seeds.push(neighbor_ID);
					m_processed_[neighbor_label] = true;
					m_sv_label_to_seg_label_map_[neighbor_label] = group_label;
					m_seg_label_to_sv_list_map_[group_label].insert(neighbor_label);
					plane_v.insert(neighbor_ID);
					num_sv_in_segment++;
				}

			}
			//求每个class增长时边不是valid的情况放到边界集合里
			if (!m_sv_adjacency_list_[*out_Edge_itr].is_valid)
			{
				m_seg_label_to_bou_list_map_[group_label].insert(neighbor_label);
				bou_v.insert(neighbor_ID);
				

			}
			
		}
		//两个set做差集，求得最终的每个class对应的边界的超体素集合
		std::set<std::uint32_t> temp_set;
		set_difference(m_seg_label_to_bou_list_map_[group_label].begin(), m_seg_label_to_bou_list_map_[group_label].end(), m_seg_label_to_sv_list_map_[group_label].begin(), m_seg_label_to_sv_list_map_[group_label].end(),inserter(temp_set,temp_set.begin()));
		m_seg_label_to_bou_list_map_[group_label].swap(temp_set);

		set_difference(bou_v.begin(), bou_v.end(), plane_v.begin(), plane_v.end(), inserter(temp_v, temp_v.begin()));
	
	}
	return (num_sv_in_segment);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PoinT> void
supersize::SuperSize<PoinT>::merge_segments()
{
	m_seg_label_to_bou_indexs_.clear();
	m_seg_label_to_seg_list_map_.clear();
	m_temp_seg_label_to_sv_list_map_.clear();
	m_temp_seg_label_to_bou_list_map_.clear();
	m_seg_processed_.clear();

	for (auto label_itr = m_seg_label_to_sv_list_map_.begin(); label_itr != m_seg_label_to_sv_list_map_.end(); ++label_itr)
	{
		const std::uint32_t& seg_label = label_itr->first;
		m_seg_processed_[seg_label] = false;
	}
	Seg_VertexIterator seg_itr, seg_itr_end;
	unsigned int segment_label = 1;

	for (std::tie(seg_itr, seg_itr_end) = boost::vertices(m_seg_adjacency_list_); seg_itr != seg_itr_end; ++seg_itr)
	{
		const Seg_VertexID seg_vertex_id = *seg_itr;
		const std::uint32_t& seg_label = m_seg_adjacency_list_[seg_vertex_id];
		if (!m_seg_processed_[seg_label])
		{
			segment_growing_with_segment(seg_vertex_id, segment_label);
			segment_label++;
		}
	}

	//把一次class融合的结果更新到最开始超体素融合结果的变量里,更新m_seg_label_to_sv_list_map_
	for (auto merge_itr = m_seg_label_to_seg_list_map_.begin(); merge_itr != m_seg_label_to_seg_list_map_.end(); ++merge_itr)
	{
		for (auto itr_one = merge_itr->second.begin(); itr_one != merge_itr->second.end(); ++itr_one)
		{
			for (auto itr = m_seg_label_to_sv_list_map_[*itr_one].begin(); itr != m_seg_label_to_sv_list_map_[*itr_one].end(); ++itr)
			{
				m_temp_seg_label_to_sv_list_map_[merge_itr->first].insert(*itr);
			}
		}

		if (merge_itr->second.size() == 1)//还是1个聚类的话他的边界不变
		{
			m_temp_seg_label_to_bou_list_map_[merge_itr->first] = m_seg_label_to_bou_list_map_[*(merge_itr->second.begin())];
		}

		else if(merge_itr->second.size() > 1)
		{
			std::set<uint32_t> set_unions, set_bou_unions, set_diff;
			for (auto bou_itr = m_seg_label_to_bou_indexs_[merge_itr->first].begin(); bou_itr != m_seg_label_to_bou_indexs_[merge_itr->first].end(); ++bou_itr)
			{
				std::set<uint32_t> set_a = m_seg_label_to_bou_list_map_[bou_itr->first];
				std::set<uint32_t> set_b = m_seg_label_to_bou_list_map_[bou_itr->second];
				std::set<uint32_t> set_a_bou = m_seg_label_to_neighbor_bou_label_list_map_[bou_itr->first][bou_itr->second];
				std::set<uint32_t> set_b_bou = m_seg_label_to_neighbor_bou_label_list_map_[bou_itr->second][bou_itr->first];

				for (auto a : set_a)
				{
					set_unions.insert(a);
				}
				for (auto b : set_b)
				{
					set_unions.insert(b);
				}
				for (auto a_bou : set_a_bou)
				{
					set_bou_unions.insert(a_bou);
				}
				for (auto b_bou : set_b_bou)
				{
					set_bou_unions.insert(b_bou);
				}
			}

			set_difference(set_unions.begin(), set_unions.end(), set_bou_unions.begin(), set_bou_unions.end(), inserter(set_diff, set_diff.begin()));
			m_temp_seg_label_to_bou_list_map_[merge_itr->first] = set_diff;
			set_unions.clear();
			set_bou_unions.clear();
			set_diff.clear();
		}
		
	}
	m_seg_label_to_sv_list_map_ = m_temp_seg_label_to_sv_list_map_;
	m_seg_label_to_bou_list_map_ = m_temp_seg_label_to_bou_list_map_;

	//更新m_sv_label_to_seg_label_map_
	m_sv_label_to_seg_label_map_.clear();
	for (auto itr = m_seg_label_to_sv_list_map_.begin(); itr != m_seg_label_to_sv_list_map_.end(); ++itr)
	{
		for (auto sv_itr = itr->second.begin(); sv_itr != itr->second.end(); ++sv_itr)
		{
			m_sv_label_to_seg_label_map_[*sv_itr] = itr->first;
		}
	}
	


}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PoinT> void
supersize::SuperSize<PoinT>::segment_growing_with_segment(const Seg_VertexID& query_pointid, const unsigned int group_label)
{
	const std::uint32_t& seg_label = m_seg_adjacency_list_[query_pointid];
	std::queue<Seg_VertexID> seeds;
	m_seg_processed_[seg_label] = true;
	seeds.push(query_pointid);
	m_seg_label_to_seg_list_map_[group_label].insert(seg_label);

	while (!seeds.empty())
	{
		Seg_VertexID curr_pointid = seeds.front();
		const std::uint32_t& curr_label = m_seg_adjacency_list_[curr_pointid];
		seeds.pop();
		Seg_OutEdgeIterator out_Edge_itr, out_Edge_itr_end;
		for (std::tie(out_Edge_itr, out_Edge_itr_end) = boost::out_edges(curr_pointid, m_seg_adjacency_list_); out_Edge_itr != out_Edge_itr_end; ++out_Edge_itr)
		{
			const Seg_VertexID neighbor_ID = boost::target(*out_Edge_itr, m_seg_adjacency_list_);
			const std::uint32_t& neighbor_label = m_seg_adjacency_list_[neighbor_ID];
			if (!m_seg_processed_[neighbor_label])
			{
				if (m_seg_adjacency_list_[*out_Edge_itr].Weight == 1.0)//可以设置的超参数
				{
					seeds.push(neighbor_ID);
					m_seg_processed_[neighbor_label] = true;
					m_seg_label_to_seg_list_map_[group_label].insert(neighbor_label);
					std::pair<std::uint32_t, std::uint32_t> seg_pair;
					seg_pair.first = curr_label;
					seg_pair.second = neighbor_label;
					m_seg_label_to_bou_indexs_[group_label].push_back(seg_pair);
				}
			}

		}

	}

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PoinT> float
supersize::SuperSize<PoinT>::cal_seg_edge_weight(const std::uint32_t source_label_arg, const std::uint32_t target_label_arg)
{
	std::set<std::uint32_t> s_sv_set, t_sv_set, s_sv_bou_set, t_sv_bou_set;
	s_sv_set = m_seg_label_to_sv_list_map_[source_label_arg];
	t_sv_set = m_seg_label_to_sv_list_map_[target_label_arg];
	s_sv_bou_set = m_seg_label_to_neighbor_bou_label_list_map_[source_label_arg][target_label_arg];
	t_sv_bou_set = m_seg_label_to_neighbor_bou_label_list_map_[target_label_arg][source_label_arg];
	if (s_sv_bou_set.size() == 0 || t_sv_bou_set.size() == 0|| s_sv_set.size()==0|| t_sv_set.size()==0)
	{
		return 0;
	}
	else
	{

		pcl::PointCloud<pcl::PointXYZ>::Ptr source_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr target_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr source_bou_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr target_bou_cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointXYZ temp;
		/*for (auto point = m_sv_labeled_cloud->cbegin(); point != m_sv_labeled_cloud->cend(); point++)
		{
			for (auto s_itr = s_sv_set.cbegin(); s_itr != s_sv_set.cend(); ++s_itr)
			{
				if (point->label == *s_itr)
				{
					temp.x = point->x;
					temp.y = point->y;
					temp.z = point->z;
					source_cloud->push_back(temp);
					
				}
			}
			for (auto t_itr = t_sv_set.cbegin(); t_itr != t_sv_set.cend(); ++t_itr)
			{
				if (point->label == *t_itr)
				{
					temp.x = point->x;
					temp.y = point->y;
					temp.z = point->z;
					target_cloud->push_back(temp);
				}
			}
			for (auto s_bou_itr = s_sv_bou_set.cbegin(); s_bou_itr != s_sv_bou_set.cend(); ++s_bou_itr)
			{
				if (point->label == *s_bou_itr)
				{
					temp.x = point->x;
					temp.y = point->y;
					temp.z = point->z;
					source_bou_cloud->push_back(temp);
				}
			}
			for (auto t_bou_itr = t_sv_bou_set.cbegin(); t_bou_itr != t_sv_bou_set.cend(); ++t_bou_itr)
			{
				if (point->label == *t_bou_itr)
				{
					temp.x = point->x;
					temp.y = point->y;
					temp.z = point->z;
					target_bou_cloud->push_back(temp);
				}
			}
		}*/
	
		for (auto s_itr = s_sv_set.cbegin(); s_itr != s_sv_set.cend(); ++s_itr)
		{
			for (int i = 0; i < m_sv_label_to_supervoxel_map_[(*s_itr)]->voxels_->size(); i++)
			{
				temp.x = m_sv_label_to_supervoxel_map_[(*s_itr)]->voxels_->points[i].x;
				temp.y = m_sv_label_to_supervoxel_map_[(*s_itr)]->voxels_->points[i].y;
				temp.z = m_sv_label_to_supervoxel_map_[(*s_itr)]->voxels_->points[i].z;
				source_cloud->push_back(temp);
			}
		}

		for (auto t_itr = t_sv_set.cbegin(); t_itr != t_sv_set.cend(); ++t_itr)
		{
			for (int i = 0; i < m_sv_label_to_supervoxel_map_[(*t_itr)]->voxels_->size(); i++)
			{
				temp.x = m_sv_label_to_supervoxel_map_[(*t_itr)]->voxels_->points[i].x;
				temp.y = m_sv_label_to_supervoxel_map_[(*t_itr)]->voxels_->points[i].y;
				temp.z = m_sv_label_to_supervoxel_map_[(*t_itr)]->voxels_->points[i].z;
				target_cloud->push_back(temp);
			}
		}

		for (auto s_bou_itr = s_sv_bou_set.cbegin(); s_bou_itr != s_sv_bou_set.cend(); ++s_bou_itr)
		{
			for (int i = 0; i < m_sv_label_to_supervoxel_map_[(*s_bou_itr)]->voxels_->size(); i++)
			{
				temp.x = m_sv_label_to_supervoxel_map_[(*s_bou_itr)]->voxels_->points[i].x;
				temp.y = m_sv_label_to_supervoxel_map_[(*s_bou_itr)]->voxels_->points[i].y;
				temp.z = m_sv_label_to_supervoxel_map_[(*s_bou_itr)]->voxels_->points[i].z;
				source_bou_cloud->push_back(temp);
			}
		}

		for (auto t_bou_itr = t_sv_bou_set.cbegin(); t_bou_itr != t_sv_bou_set.cend(); ++t_bou_itr)
		{
			for (int i = 0; i < m_sv_label_to_supervoxel_map_[(*t_bou_itr)]->voxels_->size(); i++)
			{
				temp.x = m_sv_label_to_supervoxel_map_[(*t_bou_itr)]->voxels_->points[i].x;
				temp.y = m_sv_label_to_supervoxel_map_[(*t_bou_itr)]->voxels_->points[i].y;
				temp.z = m_sv_label_to_supervoxel_map_[(*t_bou_itr)]->voxels_->points[i].z;
				target_bou_cloud->push_back(temp);
			}
		}



		if (source_bou_cloud->size() < 2 || target_bou_cloud->size() < 2)
		{
			/*cout << s_sv_set.size() << endl;
			cout << t_sv_set.size() << endl;
			cout << s_sv_bou_set.size() << endl;
			cout << t_sv_bou_set.size() << endl;
			cout << *source_cloud << endl;
			cout << *target_cloud << endl;
			cout << *source_bou_cloud << endl;
			cout << *target_bou_cloud << endl;*/
			return 0;
		}
		/*pcl::io::savePCDFile("11.pcd", *source_cloud);
		pcl::io::savePCDFile("22.pcd", *target_cloud);
		pcl::io::savePCDFile("33.pcd", *source_bou_cloud);
		pcl::io::savePCDFile("44.pcd", *target_bou_cloud);*/

		//主成分分析求主方向和长度
		pcl::MomentOfInertiaEstimation <pcl::PointXYZ> s_bou_feature, t_bou_feature;
		Eigen::Vector3f major_s_bou, middle_s_bou, minor_s_bou, major_t_bou, middle_t_bou, minor_t_bou;
		s_bou_feature.setInputCloud(source_bou_cloud);
		t_bou_feature.setInputCloud(target_bou_cloud);
		s_bou_feature.compute(); t_bou_feature.compute();
		//求两个边界的主方向
		s_bou_feature.getEigenVectors(major_s_bou, middle_s_bou, minor_s_bou);
		t_bou_feature.getEigenVectors(major_t_bou, middle_t_bou, minor_t_bou);
		//求两个主方向所在的直线
		pcl::ModelCoefficients::Ptr coefficients_s_bou(new pcl::ModelCoefficients());
		pcl::ModelCoefficients::Ptr coefficients_t_bou(new pcl::ModelCoefficients());
		coefficients_s_bou->values.resize(6);
		coefficients_t_bou->values.resize(6);

		coefficients_s_bou->values[0] = source_bou_cloud->points[0].x;
		coefficients_s_bou->values[1] = source_bou_cloud->points[0].y;
		coefficients_s_bou->values[2] = source_bou_cloud->points[0].z;
		coefficients_s_bou->values[3] = major_s_bou[0];
		coefficients_s_bou->values[4] = major_s_bou[1];
		coefficients_s_bou->values[5] = major_s_bou[2];

		coefficients_t_bou->values[0] = target_bou_cloud->points[0].x;
		coefficients_t_bou->values[1] = target_bou_cloud->points[0].y;
		coefficients_t_bou->values[2] = target_bou_cloud->points[0].z;
		coefficients_t_bou->values[3] = major_t_bou[0];
		coefficients_t_bou->values[4] = major_t_bou[1];
		coefficients_t_bou->values[5] = major_t_bou[2];

		//source和target点云投影到直线上并求投影后点云的最大值
		pcl::ProjectInliers<pcl::PointXYZ> proj_sou;
		pcl::ProjectInliers<pcl::PointXYZ> proj_tar;
		pcl::ProjectInliers<pcl::PointXYZ> proj_s_bou;
		pcl::ProjectInliers<pcl::PointXYZ> proj_t_bou;
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected_sou(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected_tar(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected_s_bou(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_projected_t_bou(new pcl::PointCloud<pcl::PointXYZ>);

		proj_sou.setModelType(SACMODEL_LINE);
		proj_tar.setModelType(SACMODEL_LINE);
		proj_s_bou.setModelType(SACMODEL_LINE);
		proj_t_bou.setModelType(SACMODEL_LINE);

		proj_sou.setInputCloud(source_cloud);
		proj_tar.setInputCloud(target_cloud);
		proj_s_bou.setInputCloud(source_bou_cloud);
		proj_t_bou.setInputCloud(target_bou_cloud);

		proj_sou.setModelCoefficients(coefficients_t_bou);
		proj_tar.setModelCoefficients(coefficients_s_bou);
		proj_s_bou.setModelCoefficients(coefficients_s_bou);
		proj_t_bou.setModelCoefficients(coefficients_t_bou);

		proj_sou.filter(*cloud_projected_sou);
		proj_tar.filter(*cloud_projected_tar);
		proj_s_bou.filter(*cloud_projected_s_bou);
		proj_t_bou.filter(*cloud_projected_t_bou);


		/*pcl::io::savePCDFile("p1.pcd", *cloud_projected_sou);
		pcl::io::savePCDFile("p2.pcd", *cloud_projected_tar);
		pcl::io::savePCDFile("p3.pcd", *cloud_projected_s_bou);
		pcl::io::savePCDFile("p4.pcd", *cloud_projected_t_bou);*/


		//求两个投影后点云的长度并赋予权值
		pcl::PointXYZ s_min, s_max, t_min, t_max, s_bou_min, s_bou_max, t_bou_min, t_bou_max;
		auto result_s = pcl::getMaxSegment(*cloud_projected_sou, s_min, s_max);
		auto result_t = pcl::getMaxSegment(*cloud_projected_tar, t_min, t_max);
		auto result_s_bou = pcl::getMaxSegment(*cloud_projected_s_bou, s_bou_min, s_bou_max);
		auto result_t_bou = pcl::getMaxSegment(*cloud_projected_t_bou, t_bou_min, t_bou_max);
		if (abs(result_t_bou / result_s) > 0.7 && abs(result_s_bou / result_t) > 0.7)//超参数
		{
			return 1.0;
		}
		if (abs(result_s - result_t_bou) < 0.03 && abs(result_t - result_s_bou) < 0.03)//超参数
		{
			return 1.0;
		}
		else
		{
			return 0.0;
		}

	}
	

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <typename PoinT> bool
supersize::SuperSize<PoinT>::edge_is_valid(const std::uint32_t source_label_arg, const std::uint32_t target_label_arg, \
	                                       float& normal_angle, float& centroid_dis)
{
	SuperVoxelPtr& sv_source = m_sv_label_to_supervoxel_map_[source_label_arg];
	SuperVoxelPtr& sv_target = m_sv_label_to_supervoxel_map_[target_label_arg];
	const Eigen::Vector3f& source_centroid = sv_source->centroid_.getVector3fMap();
	const Eigen::Vector3f& target_centroid = sv_target->centroid_.getVector3fMap();
	const Eigen::Vector3f& source_normal = sv_source->normal_.getNormalVector3fMap().normalized();
	const Eigen::Vector3f& target_normal = sv_target->normal_.getNormalVector3fMap().normalized();

	if (m_angle_threshold_ < 0)
	{
		return (false);
	}
	bool is_valid = false;
	normal_angle = getAngle3D(source_normal, target_normal, true);
	Eigen::Vector3f vec_t_to_s;
	vec_t_to_s = source_centroid - target_centroid;
	centroid_dis = vec_t_to_s.norm();
	if (normal_angle < m_angle_threshold_)//如果两个相邻的超体素法向量夹角小于阈值，则valid
	{
		is_valid = true;
	}
	return (is_valid);
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template class supersize::SuperSize<PointXYZRGB>;