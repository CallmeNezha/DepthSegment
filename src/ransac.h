#include <opencv2/core.hpp>

void RANSAC3D(std::vector<cv::Vec3f>& _data, uint _niter, uint _nsample, float _throld, cv::Vec4f& _model, float& _error);
void MeanVar(std::vector<cv::Vec3f>& _data, cv::Vec3f& _mean, cv::Vec3f& _var);