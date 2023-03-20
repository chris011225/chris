#pragma once
#include <iostream>
#include <deque>
#include <vector>
#include <string>
#include <pcl/common/common.h>
#include <pcl/memory.h>
#include <pcl/pcl_base.h>
#include <pcl/pcl_macros.h>
#include <pcl/search/search.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <list>
#include <cmath>
#include <ctime>
#include <memory>
#include <math.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/common/distances.h>
#include <pcl/common/pca.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/common/eigen.h>
#include <pcl/common/centroid.h>
#include<pcl/common/distances.h>
#include <pcl/search/organized.h>
#include <omp.h>//调用Openmp库函数

namespace thickness
{
	template <typename PointT>
	class Thickness : public pcl::PCLBase<PointT>
	{
	public:
		using PointCloud = pcl::PointCloud<PointT>;
		using KdTree = pcl::search::Search<PointT>;
		using KdTreePtr = typename KdTree::Ptr;
		using pointDirection = std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;
		using PointCloudPtr = typename PointCloud::Ptr;
		using PointCloudConstPtr = typename PointCloud::ConstPtr;
		using PointDirectionPtr = std::shared_ptr<pointDirection>;

	private:
		/** \brief 设置最小厚度聚类点个数*/
		int m_min_pts_per_cluster_;

		/** \brief 设置最大厚度聚类点个数*/
		int m_max_pts_per_cluster_;

		/** \brief 边缘点云数据*/
		PointCloudConstPtr m_A_cloud_;
		PointCloudConstPtr m_B_cloud_;

		/** \brief 原始点云数据*/
		PointCloudConstPtr m_original_cloud_;

		/** \brief 搜索方式*/
		KdTreePtr m_search_;

		/** \brief 输入的原始方向*/
		PointDirectionPtr m_A_cloud_direction_;
		PointDirectionPtr m_B_cloud_direction_;

		/** \brief 体素的结果索引*/
		std::vector<std::vector<int>> m_A_index;
		std::vector<std::vector<int>> m_B_index;

		/** \brief line_segment类*/
		class Node;

		/** \brief line_segment列表*/
		std::vector<Node*> m_node_list_A;
		std::vector<Node*> m_node_list_B;

		/** \brief 距离矩阵*/
		std::vector<std::vector<float>> m_dis_matrix_;

		/** \brief 距离计算的权*/
		float m_line_w1_ = 0.5;
		float m_line_w2_ = 0.5;

		float m_th_cen_ = 0.995;//进行聚类前判断最小质心间距是否大于阈值中的质心距离阈值





		class Node
		{
		public:
			Node* m_left_;
			Node* m_right_;
			Node* m_father_;


			/** \brief line_segment所属于的边界类型*/
			int m_type_;

			/** \brief line_segment的首尾点索引*/
			int m_begin_index_, m_end_index_;

			/** \brief line_segment的点索引的队列*/
			std::deque<int> m_node_index_list;

			/** \brief line_segment的点云*/
			pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloud_;

			/** \brief line_segment的长度*/
			float m_length_;

			/** \brief line_segment的粗度*/
			float m_roughness_;

			/** \brief line_segment的点密度*/
			float m_density_;

			/** \brief 聚类的层数*/
			int m_level_;

			/** \brief 聚类内点数*/
			int m_cluster_num_;

			/** \brief 点向式点坐标*/
			Eigen::Vector4f m_pca_point;

			/** \brief 点向式方向*/
			Eigen::Vector4f m_pca_dir;




			typename pcl::PointXYZ m_pmin_, m_pmax_;

		public:
			Node()
			{
				m_left_ = NULL;
				m_right_ = NULL;
				m_father_ = NULL;
			}
			Node(Node* A, Node* B)
			{
				this->m_left_ = A;
				this->m_right_ = B;
				this->m_left_->m_father_ = this;
				this->m_right_->m_father_ = this;
			}
		private:


		};
	public:

		/**\子node*/
		//std::vector<Node*> son_node;

		/** \brief 构造函数*/
		Thickness();

		/** \brief 析构函数*/
		~Thickness();

		/** \brief 设置构成厚度的最小点数量*/
		void set_min_cluster_size(int min_cluster_size);

		/** \brief 设置构成厚度的最大点数量*/
		void set_max_cluster_size(int max_cluster_size);

		/** \brief 设置进行邻域搜索的方法*/
		void set_search_method(const KdTreePtr& tree);

		/** \brief 输入边界点云*/
		void input_cloud(const PointCloudConstPtr& cloudA, const PointCloudConstPtr& cloudB);

		/** \brief 输入体素索引*/
		void input_index(const std::vector<std::vector<int>>& A_index, const std::vector<std::vector<int>>& B_index);

		/** \brief 输入原始点云*/
		void input_original_cloud(const PointCloudConstPtr& cloud);

		/** \brief 输入边界方向*/
		void input_cloud_direction(const PointDirectionPtr& CloudDirectionPtrA, const PointDirectionPtr& CloudDirectionPtrB);

		/** \brief 进行聚类*/
		void clustering();

		/** \brief 平面检测*/
		void plane_detection();

		/** \brief 平行检测*/
		void parallel_detection();

		/** \brief 点密度检测来确定是否为厚度*/
		void density_detection(float a, float b);

		/** \brief 统计厚度和aabb*/
		void statistics(std::vector<std::pair<float, int>>& S, std::vector<std::vector<float>>& aabb_xy, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

		/**\brief 子节点特征*/
		void find_son_node(Node* a);

		/**\提取特征做py脚本分类*/
		void feature_select();





	private:
		/** \brief 算法执行前的检测*/
		bool prepare();

		/** \brief 数据初始化，赋予每个node信息*/
		void initialization();
		void initialization1();
		/*计算视角11.8*/
		float calNodeAngle(std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>> corr);
		/** \brief 计算两个聚类端点间最短距离*/
		float cal_min_dis(Node* A, Node* B);


		/** \brief 聚类时求两个node之间的共线距离*/
		float cal_fit_line_dis(Node* A, Node* B);

		/** \brief 直线聚类时求两个node的最终加权距离*/
		float cal_line_weight_dis(float min_dis, float fit_dis);

		/** \brief 求两个node的对应点*/
		std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>> cal_node_corr(Node* A, Node* B);

		/** \brief 由node构成的平面*/
		struct  Plane
		{
			std::vector <Node*> node_list;
			pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud;

		};
		std::vector<Plane> m_plane_list_;

		/** \brief 由node构成的相互平行的node集合*/
		struct  Parallel
		{
			std::vector <Node*> node_list;
			pcl::PointCloud<pcl::PointXYZ>::Ptr parallel_cloud;

		};
		std::vector<Parallel> m_parallel_list_;

		/** \brief 符合厚度线对条件的node对*/
		struct  ThicknessPair//边
		{
			std::pair<Node*, Node*> thickness_pair;
			std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>> corr;//厚度点
			float min_x, min_y, max_x, max_y;
			int point_num;
			float thickness_val;
			std::pair<int, int>type;

		};
		double fake_test_angle(int num_cluster, pcl::PointCloud<pcl::PointXYZ>::Ptr all_thickness_one, pcl::PointCloud<pcl::PointXYZ>::Ptr all_thickness_two, pcl::PointXYZRGB midPoint, pcl::PointCloud<pcl::PointXYZ>::Ptr curvature_i, pcl::PCA<pcl::PointXYZ> pca, Eigen::Vector2f b, Eigen::RowVector3f V2, float dis, std::vector<int> pointIdxRadiusSearch, std::vector<float> pointRadiusSquaredDistance, std::vector<int> pointIdxRadiusSearch_new, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointXYZ depthPoint);





		/** \brief 倒角的厚度修正*/
		/** \brief 类内无方差滤除的厚度估计*/
		void thicknessCal_1(std::vector<ThicknessPair> ThicknessPair_list, std::vector<std::pair<float, int>>& thickresult, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
		/** \brief 类内无方差滤除的厚度估计*/
		float BinaryClassification(std::vector<float>& std_result);
		void thicknessCal(std::vector<ThicknessPair>& ThicknessPair_list, std::vector<ThicknessPair>& ThicknessPair_list_std, std::vector<std::pair<float, int>>& thickresult, std::vector<float>& std_result);
		void validation_fackAngle(std::vector<ThicknessPair> ThicknessPair_list, std::vector<std::pair<float, int>>& thickresult, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
		std::vector<ThicknessPair> m_ThicknessPair_list_;//边的集合
		std::vector<ThicknessPair> ThicknessPair_list_inclass;
		std::vector<ThicknessPair> ThicknessPair_list_difclass;

	};





}