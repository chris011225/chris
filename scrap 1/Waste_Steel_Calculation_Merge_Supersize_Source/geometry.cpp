#include <iostream>
#include <string>
#include <vector>

using namespace std;

const double EPSILON = 0.0000000001;
// 3D vector
struct Vector3dNew
{
public:
	Vector3dNew()
	{
	}

	~Vector3dNew()
	{
	}

	Vector3dNew(double dx, double dy, double dz)
	{
		x = dx;
		y = dy;
		z = dz;
	}

	// 矢量赋值
	void set(double dx, double dy, double dz)
	{
		x = dx;
		y = dy;
		z = dz;
	}

	// 矢量相加
	Vector3dNew operator + (const Vector3dNew& v) const
	{
		return Vector3dNew(x + v.x, y + v.y, z + v.z);
	}

	// 矢量相减
	Vector3dNew operator - (const Vector3dNew& v) const
	{
		return Vector3dNew(x - v.x, y - v.y, z - v.z);
	}

	//矢量数乘
	Vector3dNew Scalar(double c) const
	{
		return Vector3dNew(c * x, c * y, c * z);
	}

	// 矢量点积
	double Dot(const Vector3dNew& v) const
	{
		return x * v.x + y * v.y + z * v.z;
	}

	// 矢量叉积
	Vector3dNew Cross(const Vector3dNew& v) const
	{
		return Vector3dNew(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
	}

	bool operator == (const Vector3dNew& v) const
	{
		if (abs(x - v.x) < EPSILON && abs(y - v.y) < EPSILON && abs(z - v.z) < EPSILON)
		{
			return true;
		}
		return false;
	}

	double x, y, z;
};