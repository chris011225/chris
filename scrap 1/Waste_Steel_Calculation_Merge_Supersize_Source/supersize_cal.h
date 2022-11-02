#pragma once
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/johnson_all_pairs_shortest.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/voxel_grid.h>


namespace supersize
{

	/** \brief 废钢的超尺寸大件提取类所在的命名空间，主要为了对废钢点云进行分割，提取具有最大尺寸的大件。并能够输出其尺寸
	*   \author 高梓成 郭浩等
	*/
	template <typename PointT>
	class SuperSize
	{
		/** \brief 超体素构成邻接图边的属性*/
		struct edge_properties
		{
			/** \brief 邻接超体素的法向量夹角*/
			float normal_difference;

			/** \brief 邻接超体素的重心距离*/
			float centroid_dis;

			/** \brief 是否进行合并的依据*/
			bool is_valid;

			edge_properties() :
			normal_difference(0), is_valid(false)
			{
			}
		};
		/** \brief 每个分割结果需要输出的属性*/
		struct supersize_information
		{
			pcl::PointXYZ aabb_min, aabb_max;

			pcl::PointXYZ obb_min, obb_max;

			float first_size, second_size, third_size;

			float area;
		};
		/** \brief 2个分割连接的边的权*/
		struct edge_Weight
		{
			float Weight;
			edge_Weight() :
				Weight(0)
			{

			}
		};
	    public: 
			using SupervoxelAdjacencyList = boost::adjacency_list<boost::setS, boost::setS, boost::undirectedS, std::uint32_t, edge_properties>;
			using VertexIterator = typename boost::graph_traits<SupervoxelAdjacencyList>::vertex_iterator;
			using AdjacencyIterator = typename boost::graph_traits<SupervoxelAdjacencyList>::adjacency_iterator;
			using VertexID = typename boost::graph_traits<SupervoxelAdjacencyList>::vertex_descriptor;
			using EdgeIterator = typename boost::graph_traits<SupervoxelAdjacencyList>::edge_iterator;
			using OutEdgeIterator = typename boost::graph_traits<SupervoxelAdjacencyList>::out_edge_iterator;
			using EdgeID = typename boost::graph_traits<SupervoxelAdjacencyList>::edge_descriptor;
			using SuperVoxelPtr = typename pcl::Supervoxel<PointT>::Ptr;

			//分割后的邻接图的一些句柄
			typedef boost::property <boost::edge_weight_t, float> Weight;
			using SegmentAdjacencyList= boost::adjacency_list<boost::setS, boost::setS, boost::undirectedS, std::uint32_t, edge_Weight>;
			using Seg_VertexIterator = typename boost::graph_traits<SegmentAdjacencyList>::vertex_iterator;
			using Seg_AdjacencyIterator = typename boost::graph_traits<SegmentAdjacencyList>::adjacency_iterator;
			using Seg_VertexID = typename boost::graph_traits<SegmentAdjacencyList>::vertex_descriptor;
			using Seg_EdgeIterator = typename boost::graph_traits<SegmentAdjacencyList>::edge_iterator;
			using Seg_OutEdgeIterator = typename boost::graph_traits<SegmentAdjacencyList>::out_edge_iterator;
			using Seg_EdgeID = typename boost::graph_traits<SegmentAdjacencyList>::edge_descriptor;
			using Seg_SuperVoxelPtr = typename pcl::Supervoxel<PointT>::Ptr;


			SuperSize();
			virtual ~SuperSize();

			/** \brief 重置*/
			void reset();

			/** \brief 导入超体素分割结果后进行初始化*/
			inline void set_input_supervoxels(const std::map<std::uint32_t, SuperVoxelPtr>& supervoxel_clusters_arg,
				                              const std::multimap<std::uint32_t, std::uint32_t>& label_adjacency_arg)
			{
				prepare_segmentation(supervoxel_clusters_arg, label_adjacency_arg);//初始化
				m_supervoxels_set_ = true;
			}

			/** \brief 进行合并*/
			void segment();

			/** \brief 对点云的标签进行更新*/
			void relabel_cloud(pcl::PointCloud<pcl::PointXYZL>& labeled_cloud_arg);

			/** \brief 对点云的颜色进行更新*/
			void get_recolor_cloud(pcl::PointCloud<pcl::PointXYZL>& relabeled_cloud,pcl::PointCloud<pcl::PointXYZRGB>& recolor_cloud);

			/** \brief 对聚类后的每个class边界的点云进行保存*/
			void get_color_bou_cloud(pcl::PointCloud<pcl::PointXYZL>& super_labeled_cloud, pcl::PointCloud<pcl::PointXYZRGB>& recolor_bou_cloud);

			/** \brief 对聚类后的每个class相邻class共有边界的点云进行保存*/
			void get_color_public_bou_cloud(pcl::PointCloud<pcl::PointXYZL>& super_labeled_cloud, pcl::PointCloud<pcl::PointXYZRGB>& recolor_public_bou_cloud);

			/** \brief 获得超体素结果的邻接图*/
			inline void getSVAdjacencyList(SupervoxelAdjacencyList& adjacency_list_arg) const
			{
				adjacency_list_arg = m_sv_adjacency_list_;
			}
			/** \brief 计算每个聚类的属性*/
			void cal_supersize_properties(pcl::PointCloud<pcl::PointXYZL>& relabeled_cloud);

			/** \brief 导入超体素标签的点云*/
			void input_labeled_cloud(const pcl::PointCloud<pcl::PointXYZL>::ConstPtr labeled_cloud);

	     protected:
			 /** \brief 进行分割合并前的准备，输入是进行超体素分割后的两个结果*/
			 void prepare_segmentation(const std::map<std::uint32_t, SuperVoxelPtr>& supervoxel_clusters_arg,\
				                       const std::multimap<std::uint32_t, std::uint32_t>& label_adjacency_arg);

			 /** \brief 计算邻接图的每条边的属性是否valid*/
			 void calculate_edges_isvalid(SupervoxelAdjacencyList& adjacency_list_arg);

			 /** \brief 计算一条边的属性是否valid，返回法向量夹角和超体素重心距离*/
			 bool edge_is_valid(const std::uint32_t source_label_arg, const std::uint32_t target_label_arg, \
				                float& normal_angle, float& centroid_dis);

			 /** \brief 进行第一次的区域增长来合并超体素，并且保存第一次和并后每个合并的边界超体素*/
			 void supersize_detection();

			 /** \brief 针对一个查询点进行区域增长*/
			 int segment_growing(const VertexID& query_pointid, const unsigned int group_label,std::set<VertexID>& temp_v);

			 /** \brief 求分割后的邻接图关系*/
			 void compute_segment_adjacency();

			 /** \brief 求分割后相邻聚类结果的边界*/
			 void compute_segments_public_boundary();

			 /** \brief 求两个相邻聚类结果的边界*/
			 std::set<std::uint32_t> computer_two_seg_bou(const std::uint32_t& seg_a, const std::uint32_t& seg_b);

			 /** \brief 求分割邻接图的边的权，这个结果可以进行另外的谱分割*/
			 void compute_sgement_adjacencyList_weight();

			 /** \brief 计算分割的边的权重用于之后合并用*/
			 float cal_seg_edge_weight(const std::uint32_t source_label_arg, const std::uint32_t target_label_arg);

			 /** \brief 合并class内超体素个数小于一定值的分割*/
			 void merge_small_segments();

			 /** \brief 根据分割邻接图的权值合并分割*/
			 void merge_segments();

			 /** \brief 针对一个class进行区域增长*/
			 void segment_growing_with_segment(const Seg_VertexID& query_pointid, const unsigned int group_label);

			



		 public:

			 /** \brief 每个聚类内的属性*/
			std::map<std::uint32_t, supersize_information> m_supersizes_properties;

	     protected:

			 /** \brief 进行了超体素处理的带有label的原始点云数据*/
			 pcl::PointCloud<pcl::PointXYZL>::ConstPtr m_sv_labeled_cloud;

			 /** \brief 判断相邻超体素法向量夹角时的角度阈值*/
			 float m_angle_threshold_;

			 /** \brief 如果进行分割前的准备函数执行没问题则为ture*/
			 bool m_supervoxels_set_;

			 /** \brief 一个聚类包含的最小超体素个数*/
			 std::uint32_t m_min_segment_size_;

			 /** \brief 一共几个聚类*/
			 int m_sum_segment_;

			 /** \brief 对于一个超体素是否已经执行了合并，即已经被遍历到，组成的一个map*/
			 std::map<std::uint32_t, bool> m_processed_;

			 /** \brief 由超体素分割结果构成的邻接图，顶点为超体素的label*/
			 SupervoxelAdjacencyList m_sv_adjacency_list_;

			 /** \brief 由第n次merge结果构成的邻接图，顶点为每个class的label*/
			 SegmentAdjacencyList m_seg_adjacency_list_;

			 /** \brief 超体素label对应的各自的超体素对象的指针*/
			 std::map<std::uint32_t, typename pcl::Supervoxel<PointT>::Ptr> m_sv_label_to_supervoxel_map_;

			 /** \brief 储存原始的超体素的label到新的聚类里，给予新的label
			  *  \note m_sv_label_to_seg_label_map_[old_labelID] = new_labelID */
			 std::map<std::uint32_t, std::uint32_t> m_sv_label_to_seg_label_map_;

			 /** \brief 一个聚类的label对应一各超体素的集合*/
			 std::map<std::uint32_t, std::set<std::uint32_t> > m_seg_label_to_sv_list_map_;


			 /** \brief 一个聚类的label对应边界的超体素的集合*/
			 std::map<std::uint32_t, std::set<std::uint32_t> > m_seg_label_to_bou_list_map_;


			 /** \brief 所有边界超体素所在顶点组成的集合*/
			 std::set<VertexID> all_bou_v_;

			 /** \brief 一个聚类的邻域周围的所有聚类label的集合,得到邻接图关系*/
			 std::map<std::uint32_t, std::set<std::uint32_t> > m_seg_label_to_neighbor_set_map_;

			 /** \brief 一个聚类与邻域的每个聚类检测后得到的对应的边界的子集*/
			 std::map<std::uint32_t, std::map<std::uint32_t, std::set<std::uint32_t>>> m_seg_label_to_neighbor_bou_label_list_map_;

			 /** \brief 进行一次分割后的merge时，区域生长得到的一个聚类label对应的class集合*/
			 std::map<std::uint32_t, std::set<std::uint32_t> > m_seg_label_to_seg_list_map_;

			 /** \brief 进行一次分割后的merge时，为了更新边界m_seg_label_to_bou_list_map_储存的一个集合*/
			 std::map<std::uint32_t, std::vector<std::pair<std::uint32_t, std::uint32_t>>> m_seg_label_to_bou_indexs_;

			 /** \brief 进行聚类merge的临时变量，最后赋给m_seg_label_to_sv_list_map_*/
			 std::map<std::uint32_t, std::set<std::uint32_t> > m_temp_seg_label_to_sv_list_map_;

			 /** \brief 进行聚类merge的临时变量，最后赋给m_seg_label_to_bou_list_map_*/
			 std::map<std::uint32_t, std::set<std::uint32_t> > m_temp_seg_label_to_bou_list_map_;

			 /** \brief 对于一个class是否已经执行了合并，即已经被遍历到，组成的一个map*/
			 std::map<std::uint32_t, bool> m_seg_processed_;
			 
	};
}