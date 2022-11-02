#pragma once
#include <iostream>
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
#include "geometry.cpp"

namespace thickness
{
	template <typename PointT>
	class Thickness : public pcl::PCLBase<PointT>
	{
	public:

		using PointCloud = pcl::PointCloud<PointT>;
		using KdTree = pcl::search::Search<PointT>;
		using KdTreePtr = typename KdTree::Ptr;
		using thicknessIndex = std::vector<std::pair<int, int>>;
		using pointDirection = std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;
		using PointCloudPtr = typename PointCloud::Ptr;
		using PointCloudConstPtr = typename PointCloud::ConstPtr;
		using PointDirectionPtr = std::shared_ptr<pointDirection>;
	public:

		/** \brief
		  * 构造函数
		  */
		Thickness();

		//析构函数
		~Thickness();

		//得到设置的厚度的点集的最小内点个数
		int getMinClusterSize();

		//得到设置的厚度的点集的最大内点个数
		int getMaxClusterSize();

		//设置构成厚度的最小点数量
		void setMinClusterSize(int min_cluster_size);

		//设置构成厚度的最大点数量
		void setMaxClusterSize(int max_cluster_size);

		//得到一个边界点的方向
		Eigen::Vector3f getDirection(int type, int point_index) const;

		//得到进行邻域搜索的方法
		KdTreePtr getSearchMethod() const;

		//设置进行邻域搜索的方法
		void setSearchMethod(const KdTreePtr& tree);

		//得到每个点的搜索半径1
		float getRadius1() const;

		//得到每个点的搜索半径2
		float getRadius2() const;

		//设置邻域搜索半径1
		void setRadius1(float search_radius1);

		//设置邻域搜索半径2
		void setRadius2(float search_radius2);

		//设置图距阈值
		void setGeodesicDis(float geodesic_distance);

		//设置点到平面的距离阈值
		void setDistanceToPlane(float dis);

		//得到一个点的邻域点数量
		unsigned int getNumberOfNeighbours() const;

		//设置p2寻找q2时候角度的阈值
		void setAngelThreshold(float angel);

		//设置p1与p2方向向量夹角阈值
		void setAngelThresholdWithDirection(float angle_with_direction);

		//输入两个的边缘点云（针对的是两个类别进行检测厚度的点云）
		void inputCloud(const PointCloudConstPtr& cloudA, const PointCloudConstPtr& cloudB);

		//输入一个边缘点云（针对的是在一个类别内部进行检测厚度的点云）
		void inputCloud(const PointCloudConstPtr& cloud);

		//输入原始点云 
		void inputOriginalCloud(const PointCloudConstPtr& cloud);

		//导入两个点云的方向
		void inputCloudDirection(const PointDirectionPtr& CloudADirectionPtr, const PointDirectionPtr& CloudBDirectionPtr);

		//导入一个点云的方向
		void inputCloudDirection(const PointDirectionPtr& CloudDirectionPtr);

		//进行厚度搜索，结果放到由pair组成的vector里，detection_type表示是进行内部搜索还是两个点的搜索
		void thicknessDetection(std::vector<thicknessIndex>& all_thickness_pair_index, int& detection_type);

		//去除不是厚度的点对（中间无点）std::vector<thicknessIndex>& all_thickness_pair_index
		std::vector<thicknessIndex> validation();
		
		std::vector<thicknessIndex> validation_fackThickness(std::vector<thicknessIndex> thickness_pair, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

		std::vector<std::pair<float, int>> validation_fackAngle(std::vector<thicknessIndex> thickness_pair, std::vector<std::pair<float, int>>& thickness_result, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);

		int fake_test(pcl::PointXYZRGB midPoint, pcl::PointCloud<pcl::PointXYZ>::Ptr curvature_i, pcl::PCA<pcl::PointXYZ> pca, Eigen::Vector2f b, Eigen::RowVector3f V2, float dis, std::vector<int> pointIdxRadiusSearch, std::vector<float> pointRadiusSquaredDistance, std::vector<int> pointIdxRadiusSearch_new, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int num);
		
		double fake_test_angle(int num_cluster, pcl::PointCloud<pcl::PointXYZ>::Ptr all_thickness_one, pcl::PointCloud<pcl::PointXYZ>::Ptr all_thickness_two, pcl::PointXYZRGB midPoint, pcl::PointCloud<pcl::PointXYZ>::Ptr curvature_i, pcl::PCA<pcl::PointXYZ> pca, Eigen::Vector2f b, Eigen::RowVector3f V2, float dis, std::vector<int> pointIdxRadiusSearch, std::vector<float> pointRadiusSquaredDistance, std::vector<int> pointIdxRadiusSearch_new, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, pcl::PointXYZ depthPoint);
		//去除两边主方向不平行的厚度对--类间
		std::vector<thicknessIndex> validationDirection_classBetween();

		//去除两边主方向不平行的厚度对--类内
		std::vector<thicknessIndex> validationDirection_classWithin();

		//进行厚度的合并
		void thicknessCombine(std::vector<thicknessIndex>& all_thickness_pair_index);

		//计算所有厚度的厚度值并进行统计
		void thicknessCal(std::vector<thicknessIndex>& thickness_index, std::vector<thicknessIndex>& thickness_index_std, std::vector<std::pair<float, int>>& thickness_result, std::vector<float>& std_result, int& detection_type);
		void thicknessCal_1(std::vector<thicknessIndex>& thickness_index_arg, std::vector<std::pair<float, int>>& thickness_result, int& detection_type, pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud);//不用方差滤除的结果

		//计算厚度方差滤除阈值
		float BinaryClassification(std::vector<float>& std_result);

		void SolvingQuadratics(double a, double b, double c, std::vector<double>& t);

		void LineIntersectSphere(Vector3dNew& O, Vector3dNew& E, Vector3dNew& Center, double R, std::vector<Vector3dNew>& points);

	protected:

		//进行处理之前的检测，检测是否可以执行算法
		bool prepareWithSegmentation();
		bool prepareWithSegmentationSelf();

		//进行邻域点查找（自己查自己）
		void findNeighbours(int type);

		//进行邻域点查找（A中每个点在B中有多少邻域点）
		void findNeighboursAWithB();

		//进行一次区域生长
		int growRegion(std::pair<int, int> seed, int segment_number, std::vector<std::pair<int, int>>& segment_pairs);

		//进行类内的一次区域生长
		int growRegionSelf(std::pair<int, int> seed, int segment_number, std::vector<std::pair<int, int>>& segment_pairs, std::vector<std::vector<float> >& geodesic_distances);
		int growRegionSelf1(std::pair<int, int> seed, int segment_number, std::vector<std::pair<int, int>>& segment_pairs);

		//得到点法式的平面方程的系数
		Eigen::Vector4f getPlane(Eigen::Vector3f point, Eigen::Vector3f direction);

		//得到每个厚度点对的欧氏距离
		float getPairDis(std::pair<int, int> seed_pair);

		//得到每个厚度点对的欧氏距离
		float getPairDisSelf(std::pair<int, int> seed_pair);


	protected:

		//设置最小厚度聚类点个数
		int min_pts_per_cluster_;

		//设置最大厚度聚类点个数
		int max_pts_per_cluster_;

		//点云中点的方向
		PointDirectionPtr point_direction_a_;
		PointDirectionPtr point_direction_b_;
		PointDirectionPtr point_direction_c_;

		//点云数据
		PointCloudConstPtr original_cloud; //add
		PointCloudConstPtr margin_a;
		PointCloudConstPtr margin_b;
		PointCloudConstPtr margin_c;

		//搜索方式
		KdTreePtr search_;

		//p1寻找q1的邻域搜索的半径
		float radius_1_;

		//p1q1中点搜索p2的搜索半径
		float radius_2_;

		//计算图距时进行一次邻域搜索周围点的个数
		unsigned int neighbour_number_;

		//设置图距阈值
		float geodesic_distances_threshold_;

		//每个点的邻域点(以自身的每个点搜索自己的点云）
		std::vector<std::vector<int> > point_neighbours_a_;
		std::vector<std::vector<int> > point_neighbours_b_;
		std::vector<std::vector<int> > point_neighbours_c_;

		//每个点的邻域点（以A点云每个点为搜索点来搜索B）
		std::vector<std::vector<int> > point_neighbours_aWithb_;

		//每个点的邻域点间距（以自身的每个带你搜索自己的其他点云）
		std::vector<std::vector<float>> point_neighbours_dis_a_;
		std::vector<std::vector<float>> point_neighbours_dis_b_;
		std::vector<std::vector<float>> point_neighbours_dis_c_;

		//每个点对b点云的邻域点（以A点云的每个店为搜索点来搜索B）
		std::vector<std::vector<float>> point_neighbours_aWithb_dis_;

		//设置p2寻找q2时候角度的阈值
		float  angel_;

		//设置p1与p2方向向量夹角阈值
		float angle_with_direction_;

		//点到平面的距离阈值
		float dis_threshold_;

		//厚度检测的类别
		bool detection_type_;

		//检测所有的厚度点对索引
		std::vector<thicknessIndex> thickness_index_;

		//validation后的厚度点对索引
		std::vector<thicknessIndex> thickness_index_validation;

		//validation_direction后的厚度点对索引
		std::vector<thicknessIndex> thickness_index_validation_direction;

		//每个厚度的内点个数
		std::vector<int> num_pts_in_thickness_;

		//每个点属于哪一个厚度特征
		std::vector<int> point_labels_a_;
		std::vector<int> point_labels_b_;
		std::vector<int> point_labels_c_;

		//输出厚度和个数
		std::vector<std::pair<float, int>> thickness_result_;

		//记录厚度方差
		std::vector<float> std_result_;

		//检测出的厚度个数
		int number_of_thicknesses_;

		//每个分割结果包含多少个pair
		std::vector<int> num_pairs_in_segment;

	public:
		PCL_MAKE_ALIGNED_OPERATOR_NEW

	};
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	//定义厚度聚类的类
	template <typename PointT>
	class Cluster : public pcl::PCLBase<PointT>
	{
		public:
			using PointCloud = pcl::PointCloud<PointT>;
			using KdTree = pcl::search::Search<PointT>;
			using KdTreePtr = typename KdTree::Ptr;
			using thicknessIndex = std::vector<std::pair<int, int>>;
			using pointDirection = std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;
			using PointCloudPtr = typename PointCloud::Ptr;
			using PointCloudConstPtr = typename PointCloud::ConstPtr;
			using PointDirectionPtr = std::shared_ptr<pointDirection>;


		private:
			PointCloudConstPtr margin_A_;
			PointCloudConstPtr margin_B_;
			PointCloudConstPtr margin_C_;
			std::vector<thicknessIndex> thickness_index_;
			std::vector<thicknessIndex> thickness_index_after_clustering_;
			int Type_;//0表示类内聚类，1表示类间聚类
			class Node;
			friend class Node_;
			std::vector<Node*> node_list_;
			float th_cen_;//进行聚类前判断最小质心间距是否大于阈值中的质心距离阈值
			float th_cri_;
			float th_fit_dis;//两个聚类的端点距离阈值
			float line_w1_, line_w2_, line_w3_;
			float cricle_w1_, cricle_w2_, cricle_w3_;
			std::vector<std::vector<float>> dis_matrix_;
			double DataMap[10000][10000];
			float z_th_;
			float view_ang_th_;

		public:
			/** \brief 构造*/
			Cluster();

			/** \brief 析构*/
			~Cluster();

			/** \brief 输入一个边缘点云*/
			void inputMargin(const PointCloudConstPtr& cloudC);

			/** \brief 输入两个边缘点云*/
			void inputMargin(const PointCloudConstPtr& cloudA, const PointCloudConstPtr& cloudB);

			/** \brief 输入类间检测得到的厚度点索引*/
			void inputThicknessIndex(const std::vector<thicknessIndex>& thickness_index,int type);
			
			/** \brief 进行聚类*/
			void clustering(std::vector<thicknessIndex>& thickness_index_arg);

			/** \brief 设置质心z坐标阈值*/
			void setMaxZ(float max_z);

			/** \brief 设置视点角度阈值*/
			void setMaxViewAng(float ang);
		private:
			/** \brief 更新node的质心*/
			Eigen::Vector4f updateCenter(Node* A);

			/** \brief 得到当前node的点云数据*/
			pcl::PointCloud<pcl::PointXYZ>::Ptr getNodePointcloud(Node* A);

			/** \brief 求两个node的质心距离*/
			float calCenterDis(Node* A, Node* B);

			/** \brief 直线聚类时求两个node的主方向共线距离*/
			float calFitLineDis(Node* A, Node* B);
			float calFitLineDisSelf(Node* A, Node* B);

			/** \brief 圆聚类时求两个node的距拟合圆的距离*/
			float calFitCricleDis_classWithin(Node* A, Node* B); 
			float calFitCricleDis_classBetween(Node* A, Node* B); 

			/** \brief 求两个node的厚度差距离*/
			float calThicknessDis(Node* A, Node* B);
			float calThicknessDisSelf(Node* A, Node* B);

			/** \brief 直线聚类时求两个node的最终加权距离*/
			float calLineWeightingDis(float cen_dis, float fit_dis, float hou_dis);

			/** \brief 圆聚类时求两个node的最终加权距离*/
			float calCricleWeightingDis(float cen_dis, float cri_dis, float hou_dis);

			/** \brief 合并前的数据准备，判断数据是否导入成功*/
			bool prepareWithClustering();

			/** \brief 数据初始化，赋予每个node信息*/
			void initialization();

			/** \brief 得到一对pair的距离*/
			float calPairDis(std::pair<int, int> seed_pair);
			float calPairDisSelf(std::pair<int, int> seed_pair);

			/** \brief 计算node的点云法向量与质心到原点的向量的夹角*/
			float calNodeAngle(Node* A);

			class Node
			{
			  public:
				Node* left_;
				Node* right_;
				Node* father_;
				typename pcl::PointXYZ center_;
				typename pcl::PointXYZ pmin_, pmax_;
				std::vector<std::pair<int, int>> node_thickness_index_;
				pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_;
				float l_;
				int level_;
				int cluster_id_;
				int cluster_num;
			  public:
				Node()
				{
					left_ = NULL;
					right_ = NULL;
					father_ = NULL;
				}
				Node(Node* A, Node* B)
				{
					this->left_ = A;
					this->right_ = B;
					this->left_->father_ = this;
					this->right_->father_ = this;
				}
			  private:
				 

			};

		
	};
}