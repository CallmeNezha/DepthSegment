#include "ransac.h"
#include <set>
#include <random>

#include <Eigen/Core>
#include <Eigen/SVD>


using vec3f = Eigen::Vector3f;
using vec4f = Eigen::Vector4f;
using matXf = Eigen::MatrixXf;
using mat3f = Eigen::Matrix3f;


std::set<int> GenRandomSet(int _min, int _max, int _num)
{
    assert((_max - _min) > _num);

    std::set<int> set;
    if ((_max - _min) <= _num) {
        for (int i = _min; i < _max; ++i)
            set.insert(i);
        return set;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(_min, _max);
    while (set.size() < _num) {
        int k = dis(gen);
        while (set.end() != set.find(k))
            k = ++k > _max ? _min : k;

        set.insert(k);
    }
    return set;
}

template<typename V1, typename V2>
inline float PointPlaneDisSqr(const V1& point, const V2& plane)
{
    return std::pow(point[0] * plane[0] + point[1] * plane[1] + point[2] * plane[2] + plane[3], 2) /
        (plane[0] * plane[0] + plane[1] * plane[1] + plane[2] * plane[2]);
}

/*
@return: Sum of squared distance
*/
float FitPlaneError(const matXf& points, const vec4f& plane)
{
    float total = 0;
    for (int i = 0; i < points.rows(); ++i) {
        total += PointPlaneDisSqr(points.row(i), plane);
    }
    return total;
}


vec4f FitPlane(const matXf& points)
{
    Eigen::VectorXf b(points.rows());
    b.setConstant(-1);

    vec3f x;
    x = points.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    return vec4f(x(0), x(1), x(2), 1);
}

vec4f FitPlane(const matXf& points, const std::set<int>& indices)
{
    const int rows = (int)indices.size();
    matXf A(rows, 3);
    Eigen::VectorXf b(rows);
    vec3f x;

    b.setConstant(-1);

    int count = 0;
    for (auto iter = indices.begin(); iter != indices.end(); ++iter) {
        int idx = *iter;
        A.row(count++) = points.row(idx);
    }
    x = A.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(b);
    return vec4f(x(0), x(1), x(2), 1);
}



/**
@return: Plane equation by ax+by+cz+d (a,b,c,d) <::Vector4d>
@param: data - nx3 Matrix contains n points
@param: nIter - Number of iterations : find how many candidate models
@param: nSample - Number of sample point assumed fitting the plane
@param: thold - Threashold distance that decide whether sample fit the model
*/
void RANSAC3D_Internal(const matXf& data, unsigned int nIter, unsigned int nSample, float thold, vec4f& model, float& error)
{
    /** Goes the ransac */

    uint nMax = 0; // Maxinum number threshold of samples that fit model
    thold = std::abs(thold); //Threshold that decide whether sample fit the model.
    vec4f bestModel = vec4f::Zero();

    for (uint i = 0; i < nIter; ++i) 
    {
        auto maybeInlier = GenRandomSet(0, (int)data.rows() - 1, nSample);
        auto maybeModel = FitPlane(data, maybeInlier);

        // Include all points that fits the model
        maybeInlier.clear();
        for (int j = 0; j < data.rows(); ++j) {
            float dis = PointPlaneDisSqr(vec3f(data(j, 0), data(j, 1), data(j, 2)), maybeModel);
            if (dis < thold) {
                maybeInlier.insert(j);
            }
        }

        if (maybeInlier.size() > nMax) {
            nMax = (int)maybeInlier.size();
            bestModel = FitPlane(data, maybeInlier); //Maybe better
        }
    }
    model = bestModel;
    error = FitPlaneError(data, bestModel);
}

void RANSAC3D(std::vector<cv::Vec3f>& _data, uint _niter, uint _nsample, float _throld, cv::Vec4f& _model, float& _error)
{
    float* ptr = &_data[0][0];
    Eigen::Map<Eigen::MatrixXf> mtx(ptr, _data.size(), 3);
    vec4f plane = vec4f::Zero();
    float error = 0;
    RANSAC3D_Internal(mtx, _niter, _nsample, _throld, plane, error);
    _model = cv::Vec4f(plane[0], plane[1], plane[2], plane[3]);
    _error = error;
}

void MeanVar(std::vector<cv::Vec3f>& _data, cv::Vec3f& _mean, cv::Vec3f& _var)
{
    cv::Vec3f mean = cv::Vec3f(0, 0, 0);
    cv::Vec3f var = cv::Vec3f(0, 0, 0);
    for (auto& vec : _data)
    {
        mean += vec;
    }
    mean /= (float)_data.size();

    for (auto& vec : _data)
    {
        cv::Vec3f tmp = (mean - vec);
        var += cv::Vec3f(abs(tmp[0]), abs(tmp[1]), abs(tmp[2]));
    }
    var /= (float)_data.size();

    _mean = mean;
    _var = var;
}