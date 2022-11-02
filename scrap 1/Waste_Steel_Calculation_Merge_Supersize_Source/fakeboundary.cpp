#include <omp.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/point_tests.h> 
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/eigen.h>
#include <pcl/common/centroid.h>

using namespace std;
using namespace pcl;

class PointNew
{
private:
    int pointId, clusterId;
    int dimensions;
    vector<double> values;

    vector<double> lineToVec(string& line)
    {
        vector<double> values;
        string tmp = "";

        for (int i = 0; i < (int)line.length(); i++)
        {
            if ((48 <= int(line[i]) && int(line[i]) <= 57) || line[i] == '.' || line[i] == '+' || line[i] == '-' || line[i] == 'e')
            {
                tmp += line[i];
            }
            else if (tmp.length() > 0)
            {

                values.push_back(stod(tmp));
                tmp = "";
            }
        }
        if (tmp.length() > 0)
        {
            values.push_back(stod(tmp));
            tmp = "";
        }

        return values;
    }

public:
    PointNew(int id, string line)
    {
        pointId = id;
        values = lineToVec(line);
        dimensions = values.size();
        clusterId = 0;
    }

    int getDimensions() { return dimensions; }

    int getCluster() { return clusterId; }

    int getID() { return pointId; }

    void setCluster(int val) { clusterId = val; }

    double getVal(int pos) { return values[pos]; }
};

class ClusterNew
{
private:
    int clusterId;
    vector<double> centroid;
    vector<PointNew> points;

public:
    ClusterNew(int clusterId, PointNew centroid)
    {
        this->clusterId = clusterId;
        for (int i = 0; i < centroid.getDimensions(); i++)
        {
            this->centroid.push_back(centroid.getVal(i));
        }
        this->addPoint(centroid);
    }

    void addPoint(PointNew p)
    {
        p.setCluster(this->clusterId);
        points.push_back(p);
    }

    bool removePoint(int pointId)
    {
        int size = points.size();

        for (int i = 0; i < size; i++)
        {
            if (points[i].getID() == pointId)
            {
                points.erase(points.begin() + i);
                return true;
            }
        }
        return false;
    }

    void removeAllPoints() { points.clear(); }

    int getId() { return clusterId; }

    PointNew getPoint(int pos) { return points[pos]; }

    int getSize() { return points.size(); }

    double getCentroidByPos(int pos) { return centroid[pos]; }

    void setCentroidByPos(int pos, double val) { this->centroid[pos] = val; }
};

class KMeans_fake
{
private:
    int K, iters, dimensions, total_points;
    vector<ClusterNew> clusters;

    void clearClusters()
    {
        for (int i = 0; i < K; i++)
        {
            clusters[i].removeAllPoints();
        }
    }

    int getNearestClusterId(PointNew point)
    {
        double sum = 0.0, min_dist;
        int NearestClusterId;
        if (dimensions == 1) {
            min_dist = abs(clusters[0].getCentroidByPos(0) - point.getVal(0));
        }
        else
        {
            for (int i = 0; i < dimensions; i++)
            {
                sum += pow(clusters[0].getCentroidByPos(i) - point.getVal(i), 2.0);
            }
            min_dist = sqrt(sum);
        }
        NearestClusterId = clusters[0].getId();

        for (int i = 1; i < K; i++)
        {
            double dist;
            sum = 0.0;

            if (dimensions == 1) {
                dist = abs(clusters[i].getCentroidByPos(0) - point.getVal(0));
            }
            else {
                for (int j = 0; j < dimensions; j++)
                {
                    sum += pow(clusters[i].getCentroidByPos(j) - point.getVal(j), 2.0);
                }

                dist = sqrt(sum);
            }
            if (dist < min_dist)
            {
                min_dist = dist;
                NearestClusterId = clusters[i].getId();
            }
        }

        return NearestClusterId;
    }

public:
    KMeans_fake(int K, int iterations)
    {
        this->K = K;
        this->iters = iterations;
    }

    double run(vector<PointNew>& all_points)
    {
        total_points = all_points.size();
        dimensions = all_points[0].getDimensions();

        vector<int> used_pointIds;
        int t = 0;
        for (int i = 1; i <= K; i++)
        {
            while (true)
            {
                int index = t;

                if (find(used_pointIds.begin(), used_pointIds.end(), index) ==
                    used_pointIds.end())
                {
                    used_pointIds.push_back(index);
                    all_points[index].setCluster(i);
                    ClusterNew cluster(i, all_points[index]);
                    clusters.push_back(cluster);
                    t++;
                    break;
                }
            }
        }

        int iter = 1;
        while (true)
        {

            bool done = true;

            for (int i = 0; i < total_points; i++)
            {
                int currentClusterId = all_points[i].getCluster();
                int nearestClusterId = getNearestClusterId(all_points[i]);

                if (currentClusterId != nearestClusterId)
                {
                    all_points[i].setCluster(nearestClusterId);
                    done = false;
                }
            }

            clearClusters();

            for (int i = 0; i < total_points; i++)
            {
                clusters[all_points[i].getCluster() - 1].addPoint(all_points[i]);
            }

            for (int i = 0; i < K; i++)
            {
                int ClusterSize = clusters[i].getSize();

                for (int j = 0; j < dimensions; j++)
                {
                    double sum = 0.0;
                    if (ClusterSize > 0)
                    {
                        for (int p = 0; p < ClusterSize; p++)
                        {
                            sum += clusters[i].getPoint(p).getVal(j);
                        }
                        clusters[i].setCentroidByPos(j, sum / ClusterSize);
                    }
                }
            }

            if (done || iter >= iters)
            {
                break;
            }
            iter++;
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster_one(new pcl::PointCloud<pcl::PointXYZ>);
        for (int i = 1; i < clusters[0].getSize(); i++)
        {
        	pcl::PointXYZ temp;
            temp.x = clusters[0].getPoint(i).getVal(0);
        	temp.y = clusters[0].getPoint(i).getVal(1);
        	temp.z = clusters[0].getPoint(i).getVal(2);
            cloud_cluster_one->push_back(temp);
        }
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster_two(new pcl::PointCloud<pcl::PointXYZ>);
        for (int i = 1; i < clusters[1].getSize(); i++)
        {
            pcl::PointXYZ temp;
            temp.x = clusters[1].getPoint(i).getVal(0);
            temp.y = clusters[1].getPoint(i).getVal(1);
            temp.z = clusters[1].getPoint(i).getVal(2);
            cloud_cluster_two->push_back(temp);
        }
        Eigen::Vector4d centroid;                    
        Eigen::Matrix3d covariance_matrix;          
        pcl::computeMeanAndCovarianceMatrix(*cloud_cluster_one, covariance_matrix, centroid);
        Eigen::Matrix3d eigenVectors;
        Eigen::Vector3d eigenValues;
        pcl::eigen33(covariance_matrix, eigenVectors, eigenValues);
        Eigen::Vector3d::Index minRow, minCol;
        eigenValues.minCoeff(&minRow, &minCol);
        Eigen::Vector3d normal = eigenVectors.col(minCol);
        Eigen::Vector4d centroid_2;
        Eigen::Matrix3d covariance_matrix_2;
        pcl::computeMeanAndCovarianceMatrix(*cloud_cluster_two, covariance_matrix_2, centroid_2);
        Eigen::Matrix3d eigenVectors_2;
        Eigen::Vector3d eigenValues_2;
        pcl::eigen33(covariance_matrix_2, eigenVectors_2, eigenValues_2);
        Eigen::Vector3d::Index minRow_2, minCol_2;
        eigenValues_2.minCoeff(&minRow_2, &minCol_2);
        Eigen::Vector3d normal_2 = eigenVectors_2.col(minCol_2);
        double angle;
        angle = abs((normal[0] * normal_2[0] + normal[1] * normal_2[1] + normal[2] * normal_2[2]) / (sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]) * sqrt(normal_2[0] * normal_2[0] + normal_2[1] * normal_2[1] + normal_2[2] * normal_2[2])));
        return angle;
    }
};
