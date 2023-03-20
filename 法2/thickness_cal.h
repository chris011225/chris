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
#include <omp.h>//����Openmp�⺯��

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
		/** \brief ������С��Ⱦ�������*/
		int m_min_pts_per_cluster_;

		/** \brief ��������Ⱦ�������*/
		int m_max_pts_per_cluster_;

		/** \brief ��Ե��������*/
		PointCloudConstPtr m_A_cloud_;
		PointCloudConstPtr m_B_cloud_;

		/** \brief ԭʼ��������*/
		PointCloudConstPtr m_original_cloud_;

		/** \brief ������ʽ*/
		KdTreePtr m_search_;

		/** \brief �����ԭʼ����*/
		PointDirectionPtr m_A_cloud_direction_;
		PointDirectionPtr m_B_cloud_direction_;

		/** \brief ���صĽ������*/
		std::vector<std::vector<int>> m_A_index;
		std::vector<std::vector<int>> m_B_index;

		/** \brief line_segment��*/
		class Node;

		/** \brief line_segment�б�*/
		std::vector<Node*> m_node_list_A;
		std::vector<Node*> m_node_list_B;

		/** \brief �������*/
		std::vector<std::vector<float>> m_dis_matrix_;

		/** \brief ��������Ȩ*/
		float m_line_w1_ = 0.5;
		float m_line_w2_ = 0.5;

		float m_th_cen_ = 0.995;//���о���ǰ�ж���С���ļ���Ƿ������ֵ�е����ľ�����ֵ





		class Node
		{
		public:
			Node* m_left_;
			Node* m_right_;
			Node* m_father_;


			/** \brief line_segment�����ڵı߽�����*/
			int m_type_;

			/** \brief line_segment����β������*/
			int m_begin_index_, m_end_index_;

			/** \brief line_segment�ĵ������Ķ���*/
			std::deque<int> m_node_index_list;

			/** \brief line_segment�ĵ���*/
			pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloud_;

			/** \brief line_segment�ĳ���*/
			float m_length_;

			/** \brief line_segment�Ĵֶ�*/
			float m_roughness_;

			/** \brief line_segment�ĵ��ܶ�*/
			float m_density_;

			/** \brief ����Ĳ���*/
			int m_level_;

			/** \brief �����ڵ���*/
			int m_cluster_num_;

			/** \brief ����ʽ������*/
			Eigen::Vector4f m_pca_point;

			/** \brief ����ʽ����*/
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

		/**\��node*/
		//std::vector<Node*> son_node;

		/** \brief ���캯��*/
		Thickness();

		/** \brief ��������*/
		~Thickness();

		/** \brief ���ù��ɺ�ȵ���С������*/
		void set_min_cluster_size(int min_cluster_size);

		/** \brief ���ù��ɺ�ȵ���������*/
		void set_max_cluster_size(int max_cluster_size);

		/** \brief ���ý������������ķ���*/
		void set_search_method(const KdTreePtr& tree);

		/** \brief ����߽����*/
		void input_cloud(const PointCloudConstPtr& cloudA, const PointCloudConstPtr& cloudB);

		/** \brief ������������*/
		void input_index(const std::vector<std::vector<int>>& A_index, const std::vector<std::vector<int>>& B_index);

		/** \brief ����ԭʼ����*/
		void input_original_cloud(const PointCloudConstPtr& cloud);

		/** \brief ����߽緽��*/
		void input_cloud_direction(const PointDirectionPtr& CloudDirectionPtrA, const PointDirectionPtr& CloudDirectionPtrB);

		/** \brief ���о���*/
		void clustering();

		/** \brief ƽ����*/
		void plane_detection();

		/** \brief ƽ�м��*/
		void parallel_detection();

		/** \brief ���ܶȼ����ȷ���Ƿ�Ϊ���*/
		void density_detection(float a, float b);

		/** \brief ͳ�ƺ�Ⱥ�aabb*/
		void statistics(std::vector<std::pair<float, int>>& S, std::vector<std::vector<float>>& aabb_xy, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

		/**\brief �ӽڵ�����*/
		void find_son_node(Node* a);

		/**\��ȡ������py�ű�����*/
		void feature_select();





	private:
		/** \brief �㷨ִ��ǰ�ļ��*/
		bool prepare();

		/** \brief ���ݳ�ʼ��������ÿ��node��Ϣ*/
		void initialization();
		void initialization1();
		/*�����ӽ�11.8*/
		float calNodeAngle(std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>> corr);
		/** \brief ������������˵����̾���*/
		float cal_min_dis(Node* A, Node* B);


		/** \brief ����ʱ������node֮��Ĺ��߾���*/
		float cal_fit_line_dis(Node* A, Node* B);

		/** \brief ֱ�߾���ʱ������node�����ռ�Ȩ����*/
		float cal_line_weight_dis(float min_dis, float fit_dis);

		/** \brief ������node�Ķ�Ӧ��*/
		std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>> cal_node_corr(Node* A, Node* B);

		/** \brief ��node���ɵ�ƽ��*/
		struct  Plane
		{
			std::vector <Node*> node_list;
			pcl::PointCloud<pcl::PointXYZ>::Ptr plane_cloud;

		};
		std::vector<Plane> m_plane_list_;

		/** \brief ��node���ɵ��໥ƽ�е�node����*/
		struct  Parallel
		{
			std::vector <Node*> node_list;
			pcl::PointCloud<pcl::PointXYZ>::Ptr parallel_cloud;

		};
		std::vector<Parallel> m_parallel_list_;

		/** \brief ���Ϻ���߶�������node��*/
		struct  ThicknessPair//��
		{
			std::pair<Node*, Node*> thickness_pair;
			std::vector<std::pair<pcl::PointXYZ, pcl::PointXYZ>> corr;//��ȵ�
			float min_x, min_y, max_x, max_y;
			int point_num;
			float thickness_val;
			std::pair<int, int>type;

		};
		double fake_test_angle(int num_cluster, pcl::PointCloud<pcl::PointXYZ>::Ptr all_thickness_one, pcl::PointCloud<pcl::PointXYZ>::Ptr all_thickness_two, pcl::PointXYZRGB midPoint, pcl::PointCloud<pcl::PointXYZ>::Ptr curvature_i, pcl::PCA<pcl::PointXYZ> pca, Eigen::Vector2f b, Eigen::RowVector3f V2, float dis, std::vector<int> pointIdxRadiusSearch, std::vector<float> pointRadiusSquaredDistance, std::vector<int> pointIdxRadiusSearch_new, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointXYZ depthPoint);





		/** \brief ���ǵĺ������*/
		/** \brief �����޷����˳��ĺ�ȹ���*/
		void thicknessCal_1(std::vector<ThicknessPair> ThicknessPair_list, std::vector<std::pair<float, int>>& thickresult, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
		/** \brief �����޷����˳��ĺ�ȹ���*/
		float BinaryClassification(std::vector<float>& std_result);
		void thicknessCal(std::vector<ThicknessPair>& ThicknessPair_list, std::vector<ThicknessPair>& ThicknessPair_list_std, std::vector<std::pair<float, int>>& thickresult, std::vector<float>& std_result);
		void validation_fackAngle(std::vector<ThicknessPair> ThicknessPair_list, std::vector<std::pair<float, int>>& thickresult, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);
		std::vector<ThicknessPair> m_ThicknessPair_list_;//�ߵļ���
		std::vector<ThicknessPair> ThicknessPair_list_inclass;
		std::vector<ThicknessPair> ThicknessPair_list_difclass;

	};





}