#include <opencv2/imgproc.hpp>
#include <random>
#include <map>

template<typename T>
inline T Clip(T _val, T _min, T _max) {
    // Row clip
    if (_val < _min) _val = _min;
    if (_val > _max) _val = _max;
    return _val;
};

cv::Mat ClipDistance(const cv::Mat& _img, int _near, int _far) 
{
    if (CV_16UC1 != _img.type())
    {
        throw std::exception("Type error, MedianFilter accepts CV_16UC1");
    }
    cv::Mat img_processed = _img.clone();
    for (size_t r = 0; r < _img.rows; ++r)
    {
        for (size_t c = 0; c < _img.cols; ++c)
        {
            ushort depth = img_processed.at<ushort>(r, c);
            if (depth < _near || depth > _far) img_processed.at<ushort>(r, c) = 0;
        }
    }
    return img_processed;
}

cv::Mat MedianFilter(const cv::Mat& _img)
{
    if (CV_16UC1 != _img.type())
    {
        throw std::exception("Type error, MedianFilter accepts CV_16UC1");
    }

    const ushort kernelsize = (3 - 1) / 2;


    cv::Mat img_processed(_img.rows, _img.cols, _img.type());
    for (size_t r = 0; r < _img.rows; ++r)
    {
        for (size_t c = 0; c < _img.cols; ++c)
        {
            const int pixldepth = (int)_img.at<ushort>(r, c);
            int sumvar = 0;
            for (int h = - kernelsize; h <= kernelsize; ++h)
            {
                for (int v = - kernelsize; v <= kernelsize; ++v)
                {
                    if ( ! (r + v >= 0 && r + v < _img.rows && c + h >= 0 && c + h < _img.cols)) continue;
                    sumvar += std::abs(pixldepth - (int)_img.at<ushort>(r + v, c + h));
                }
            }

            if (sumvar > 50)
                img_processed.at<ushort>(r, c) = 0;
            else
                img_processed.at<ushort>(r, c) = pixldepth;
        }
    }
    return img_processed;
}


cv::Mat Depth2Color(const cv::Mat& _img)
{
    if (CV_16UC1 != _img.type())
    {
        throw std::exception("Type error, MedianFilter accepts CV_16UC1");
    }

    const ushort minreliabledepth = 255;

    cv::Mat color(_img.rows, _img.cols, CV_8UC3);
    for (size_t r = 0; r < _img.rows; ++r)
    {
        for (size_t c = 0; c < _img.cols; ++c)
        {
            const ushort pixldepth = _img.at<ushort>(r, c) - minreliabledepth;

            if (pixldepth >= 512 && pixldepth < 768)
            {
                color.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 767 - pixldepth, pixldepth - 512);
            }
            else if (pixldepth >= 256 && pixldepth < 512) 
            {
                color.at<cv::Vec3b>(r, c) = cv::Vec3b(511 - pixldepth, pixldepth - 256, 0);
            }
            else if (pixldepth < 256)
            {
                color.at<cv::Vec3b>(r, c) = cv::Vec3b(pixldepth, 0, 0);
            }
            else 
            {
                color.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
            }
        }
    }
    return color;
}

cv::Mat CalcuNormal(const cv::Mat& _img)
{
    if (CV_16UC1 != _img.type())
    {
        throw std::exception("Type error, CalcuNormal accepts CV_16UC1");
    }

    const int offsetsize = 2;
    const cv::Vec2i offset[3]{
          cv::Vec2i(-offsetsize, 0)
        , cv::Vec2i(offsetsize, -offsetsize)
        , cv::Vec2i(offsetsize,  offsetsize)
    };

    const cv::Vec2i offset_dual[3]{
          cv::Vec2i(offsetsize, 0)
        , cv::Vec2i(-offsetsize, offsetsize)
        , cv::Vec2i(-offsetsize, -offsetsize)
    };


    cv::Mat normal(_img.rows, _img.cols, CV_32FC3);

    //TODO: Nezha Test more on this normal compute.
    for (int i = 0; i < 1; ++i)
    {
        const cv::Vec2i* offset_seleted = (i == 0 ? offset : offset_dual);

        for (size_t r = 0; r < _img.rows; ++r)
        {
            for (size_t c = 0; c < _img.cols; ++c)
            {
                cv::Vec2i up = cv::Vec2i(r, c) + offset_seleted[0];
                cv::Vec2i leftdown = cv::Vec2i(r, c) + offset_seleted[1];
                cv::Vec2i rightdown = cv::Vec2i(r, c) + offset_seleted[2];
                {
                    up[0] = Clip(up[0], 0, _img.rows - 1);
                    leftdown[0] = Clip(leftdown[0], 0, _img.rows - 1);
                    leftdown[1] = Clip(leftdown[1], 0, _img.cols - 1);
                    rightdown[0] = Clip(rightdown[0], 0, _img.rows - 1);
                    rightdown[1] = Clip(rightdown[1], 0, _img.cols - 1);
                }

                cv::Vec3f c_p = cv::Vec3f(float(up[0]), float(up[1]), _img.at<ushort>(up[0], up[1]));
                cv::Vec3f a_p = cv::Vec3f(float(leftdown[0]), float(leftdown[1]), _img.at<ushort>(leftdown[0], leftdown[1]));
                cv::Vec3f b_p = cv::Vec3f(float(rightdown[0]), float(rightdown[1]), _img.at<ushort>(rightdown[0], rightdown[1]));

                cv::Vec3f vec_1 = c_p - a_p;
                cv::Vec3f vec_2 = b_p - a_p;
                cv::Vec3f vec_normal = vec_2.cross(vec_1);
                assert(cv::norm(vec_normal) > std::numeric_limits<float>::epsilon());
                vec_normal = cv::normalize(vec_normal);
                if (1 == i)
                    normal.at<cv::Vec3f>(r, c) = (vec_normal + normal.at<cv::Vec3f>(r, c)) / 2.f;
                else
                    normal.at<cv::Vec3f>(r, c) = vec_normal;
            }
        }
    }
    
    return normal;
}

//! Attention: Nezha, _average do more smooth effect and shapen(noise sensitive) when turned off.
cv::Mat EdgeDetect(const cv::Mat& _img, ushort _kernelsize, float _throld, bool _average)
{
    if (CV_32FC3 != _img.type())
    {
        throw std::exception("Type error, EdgeDetect accepts CV_32FC3");
    }

    cv::Mat edge(_img.rows, _img.cols, CV_8UC1);

    auto IsEdge = [&](int row, int col) -> bool {

        const cv::Vec3f normal_1 = _img.at<cv::Vec3f>(row, col);


        /*std::vector<float> adjvals(8, 0);*/
        for(int idx_dir = 0; idx_dir < 8; ++idx_dir)
        {
            int rsign = 0;
            int csign = 0;
            switch (idx_dir) 
            {
            case 0:
                rsign = 0;
                csign = -1;
                break;
            case 1:
                rsign = -1;
                csign = -1;
                break;
            case 2:
                rsign = -1;
                csign = 0;
                break;
            case 3:
                rsign = -1;
                csign = 1;
                break;
            case 4:
                rsign = 0;
                csign = 1;
                break;
            case 5:
                rsign = 1;
                csign = 1;
                break;
            case 6:
                rsign = 1;
                csign = 0;
                break;
            case 7:
                rsign = 1;
                csign = -1;
                break;
            }

            float sum = 0.f;
            for (int i = 0; i < _kernelsize; ++i)
            {
                const int r = Clip(row + rsign * i, 0, _img.rows - 1);
                const int c = Clip(col + csign * i, 0, _img.cols - 1);
                const cv::Vec3f normal_2 = _img.at<cv::Vec3f>(r, c);
                const float val = normal_1.dot(normal_2);
                if(_average)
                    sum += val;
                else
                    if (val < _throld) return true;
            }
            if (_average)
            {
                //adjvals[idx_dir] = sum / (float)_kernelsize;
                //if (adjvals[idx_dir] < _throld) return true;
                if (sum / (float)_kernelsize < _throld) return true;
            }
        }
        return false;
    };

    for (size_t r = 0; r < _img.rows; ++r)
    {
        for (size_t c = 0; c < _img.cols; ++c)
        {
            bool isedge = IsEdge(r, c);
            if (isedge)
                edge.at<uchar>(r, c) = 0;
            else
                edge.at<uchar>(r, c) = 255;
        }
    }
    return edge;
}


//TODO: Nezha when patches dramatically increase, performance will drop simutanously
cv::Mat SegmentPatches(const cv::Mat& _img)
{
    assert(_img.type() == CV_8UC1);
    cv::Mat patches = _img.clone();

    int idx_patch = 0;


    cv::Mat mask;
    for (size_t r = 0; r < patches.rows; ++r)
    {
        for (size_t c = 0; c < patches.cols; ++c)
        {
            uchar id = patches.at<uchar>(r, c);
            if(255 == id)
            {
                //std::random_device rd;
                //std::mt19937       gen(rd());
                //std::uniform_int_distribution<> dis(0, 255);
                floodFill(patches, mask, cv::Point(c, r), cv::Scalar(++idx_patch)); // From index 1
            }
        }
    }
    return patches;
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

void GetPlanesMask(const cv::Mat& _normal, const cv::Mat& _label, cv::Mat& _mask)
{
    using planepair = std::pair<int, cv::Vec4f>;

    std::vector<std::pair<int, cv::Vec4f>> _planes;
    cv::Mat _debugmat;

    assert(_label.type() == CV_8UC1 && _normal.type() == CV_32FC3 && _label.size == _normal.size);
    _planes.clear();

    std::vector<std::vector<cv::Vec3f>> normals;
    std::map<uchar, uint>               labelidxpair;
    std::map<uint, uchar>               labelidxpair_inverse;

    uint idx = -1;
    for (size_t r = 0; r < _label.rows; ++r)
    {
        for (size_t c = 0; c < _label.cols; ++c)
        {
            uchar label = _label.at<uchar>(r, c);
            if (0 == label) continue;
            if (labelidxpair.end() == labelidxpair.find(label))
            {
                normals.resize(++idx + 1);
                labelidxpair.insert(std::pair<uchar, uint>(label, idx));
                labelidxpair_inverse.insert(std::pair<uint, uchar>(idx, label));
            }
            cv::Vec3f normal = _normal.at<cv::Vec3f>(r, c);

            normals[labelidxpair[label]].push_back(normal);
        }
    }
    for (size_t i = 0; i < normals.size(); ++i)
    {
        if (normals[i].size() < 300)
        {
            _planes.push_back(planepair(-1, cv::Vec4f(0.f, 0.f, 0.f, 0.f)));
        }
        else
        {
            /*          cv::Vec4f plane;
            float error = 0;
            RANSAC3D(points[i], 10, 100, 5, plane, error);
            printf("Total plane fit error is %f\n", error);*/
            // Use variance is more efficient
            cv::Vec3f mean, var;
            MeanVar(normals[i], mean, var);
            printf("Label %d Variance: %f, %f, %f \n", labelidxpair_inverse[i], var[0], var[1], var[2]);
            cv::Vec4f plane = cv::Vec4f(mean[0], mean[1], mean[2], 0.f);
            _planes.push_back(planepair(labelidxpair_inverse[i], plane));
        }
    }

    _debugmat = cv::Mat::zeros(_label.rows, _label.cols, CV_32FC3);

    for (size_t r = 0; r < _label.rows; ++r)
    {
        for (size_t c = 0; c < _label.cols; ++c)
        {
            int label = (int)_label.at<uchar>(r, c);
            int idx = labelidxpair[label];
            if (label == _planes[idx].first && 0 != label)
            {
                cv::Vec3f plane = cv::Vec3f(_planes[idx].second[0], _planes[idx].second[1], _planes[idx].second[2]);
                _debugmat.at<cv::Vec3f>(r, c) = cv::Vec3f(abs(plane[0]) , abs(plane[1]) , abs(plane[2]) );
            }
        }
    }

    _mask = _debugmat;
}