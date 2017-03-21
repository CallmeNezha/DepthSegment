#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <map>
#include "types.h"
//#include "ransac.h"

extern cv::Mat MedianFilter(const cv::Mat& _img);
extern cv::Mat Depth2Color(const cv::Mat& _img);
extern cv::Mat CalcuNormal(const cv::Mat& _img);
extern cv::Mat EdgeDetect(const cv::Mat& _img, ushort _kernelsize, float _throld, bool _average);
extern cv::Mat SegmentPatches(const cv::Mat& _img);
extern cv::Mat ClipDistance(const cv::Mat& _img, int _near, int _far);
extern void GetPlanesMask(const cv::Mat& _normal, const cv::Mat& _label, cv::Mat& _mask);

#if defined USE_KINECT2
static const int width_depth = 512;
static const int height_depth = 424;
IKinectSensor*      kinect;
IDepthFrameReader*  dfr;
#endif //!USE_KINECT2

#if defined USE_KINECT1
static const int width_depth = 640;
static const int height_depth = 480;
INuiSensor* nuisensor = NULL;
HANDLE	    stream_depth = NULL;
HANDLE      event_nextdepthframe = NULL;
#endif //!USE_KINECT1

#if defined USE_REALSENSE
static const int width_depth = 640;
static const int height_depth = 480;

PXCSenseManager*    pp = NULL;
PXCImage*           depthIm = NULL;
PXCImage::ImageData depth_data;
PXCImage::ImageInfo depth_info;

#endif


template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease)
{
    if (pInterfaceToRelease != NULL)
    {
        pInterfaceToRelease->Release();
        pInterfaceToRelease = NULL;
    }
}


#if defined USE_KINECT2

void ProcessDepth(long long _time, const unsigned short* _buffer, int _width, int _height, unsigned short _mindepth, unsigned short _maxdepth)
{
    cv::Mat depth = cv::Mat(_height, _width, CV_16UC1, (void*)_buffer);
    cv::Mat dst = cv::Mat(_height, _width, CV_16UC1);
    cv::Mat normal = cv::Mat(_height, _width, CV_32FC3);
    cv::Mat normal_smoothed = cv::Mat(_height, _width, CV_32FC3);
    cv::Mat edgemask, edgemask_smoothed, depth_smoothed;
    cv::Mat edgemask_erode;
    cv::Mat patches;
    cv::Mat patch_dialate;
    cv::Mat normalmask;

    //depth = MedianFilter(depth);
    //depth.convertTo(depth_smoothed, CV_32FC1);


    // Only contains 256cm ~ 1024cm depths
    
    depth = ClipDistance(depth, 0, 3000);

    normal = CalcuNormal(depth);

    //const int kernel_size = 3;
    //cv::bilateralFilter(normal, normal_smoothed, kernel_size, kernel_size * 2, kernel_size / 2);
    //cv::imshow("bilateral smoothed", normal_smoothed);
   /* const int kernel_size = 3;
    cv::bilateralFilter(normal, normal_smoothed, kernel_size, kernel_size * 2, kernel_size / 2);
    cv::swap(normal, normal_smoothed);*/
    for (int i = 0; i < 3; ++i)
    {
        const int kernel_size = 3;
        cv::bilateralFilter(normal, normal_smoothed, kernel_size, kernel_size * 2, kernel_size / 2);
        cv::swap(normal, normal_smoothed);
    }
    edgemask = EdgeDetect(normal, 5, 0.9f, true);

    const int dilation_size = 7;
    const int erode_size = 2;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS,
        cv::Size(2 * erode_size + 1, 2 * erode_size + 1),
        cv::Point(erode_size, erode_size));
    
    erode(edgemask, edgemask_erode, element);
    cv::swap(edgemask, edgemask_erode);

    for (int i = 0; i < 3; ++i)
    {
        /*const int kernel_size = 3;
        cv::bilateralFilter(edgemask, edgemask_smoothed, kernel_size, kernel_size * 2, kernel_size / 2);*/
        cv::medianBlur(edgemask, edgemask_smoothed, 3);
        cv::swap(edgemask, edgemask_smoothed);
    }

    patches = SegmentPatches(edgemask);


    /// Apply the dilation operation
    element = cv::getStructuringElement(cv::MORPH_CROSS,
        cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        cv::Point(dilation_size, dilation_size));
    dilate(patches, patch_dialate, element);
    GetPlanesMask(normal, patch_dialate, normalmask);

    //mat = Depth2Color(dst);
    cv::imshow("Depth", depth * 10);
    cv::imshow("Planes Masks", normalmask);
}


HRESULT InitDefaultSensor()
{
    HRESULT hr;
    hr = GetDefaultKinectSensor(&kinect);

    if (FAILED(hr))
    {
        return hr;
    }

    if (kinect)
    {
        IDepthFrameSource* dfs = NULL;
        hr = kinect->Open();

        if (SUCCEEDED(hr))
        {
            hr = kinect->get_DepthFrameSource(&dfs);
        }
        if (SUCCEEDED(hr))
        {
            hr = dfs->OpenReader(&dfr);
        }
        SafeRelease(dfs);

    }

    if (!kinect || FAILED(hr))
    {
        printf("No ready Kinect found!\n");
        return E_FAIL;
    }
    return hr;
}

void Update()
{
    if (!dfr)
    {
        printf("Depth frame reader not ready! \n");
        return;
    }
    IDepthFrame* df = NULL;

    HRESULT hr = dfr->AcquireLatestFrame(&df);
    if (SUCCEEDED(hr))
    {
        IFrameDescription* fd = NULL;

        long long       time;
        unsigned short  minreliabledis_depth = 0;
        unsigned short  maxdis_depth = 0;
        unsigned int    buffersize = 0;
        unsigned short* buffer = NULL;

        int             width = 0;
        int             height = 0;

        hr = df->get_RelativeTime(&time);

        if (SUCCEEDED(hr))
        {
            hr = df->get_FrameDescription(&fd);
        }
        if (SUCCEEDED(hr))
        {
            hr = fd->get_Width(&width);
        }
        if (SUCCEEDED(hr))
        {
            hr = fd->get_Height(&height);
        }

        if (SUCCEEDED(hr))
        {
            hr = df->get_DepthMinReliableDistance(&minreliabledis_depth);
        }
        if (SUCCEEDED(hr))
        {
            // In order to see the full range of depth (including the less reliable far field depth)
            // we are setting nDepthMaxDistance to the extreme potential depth threshold
            maxdis_depth = USHRT_MAX;

            // Note:  If you wish to filter by reliable depth distance, uncomment the following line.
            //// hr = pDepthFrame->get_DepthMaxReliableDistance(&nDepthMaxDistance);
        }
        if (SUCCEEDED(hr))
        {
            hr = df->AccessUnderlyingBuffer(&buffersize, &buffer);
        }
        if (SUCCEEDED(hr))
        {
            ProcessDepth(time, buffer, width, height, minreliabledis_depth, maxdis_depth);
        }
        SafeRelease(fd);
    }
    SafeRelease(df);
}
#endif //!USE_KINECT2

#if defined USE_KINECT1

void ProcessDepth(long long _time, const cv::Mat& _img, int _width, int _height, unsigned short _mindepth, unsigned short _maxdepth)
{
    cv::Mat depth = _img;
    cv::Mat dst = cv::Mat(_height, _width, CV_16UC1);
    cv::Mat normal = cv::Mat(_height, _width, CV_32FC3);
    cv::Mat normal_smoothed = cv::Mat(_height, _width, CV_32FC3);
    cv::Mat edgemask, edgemask_smoothed, depth_smoothed;
    //depth = MedianFilter(depth);
    //depth.convertTo(depth_smoothed, CV_32FC1);


    // Only contains 256cm ~ 1024cm depths
    
    depth = ClipDistance(depth, 0, 3000);

    normal = CalcuNormal(depth);

    //const int kernel_size = 3;
    //cv::bilateralFilter(normal, normal_smoothed, kernel_size, kernel_size * 2, kernel_size / 2);
    //cv::imshow("bilateral smoothed", normal_smoothed);
   /* const int kernel_size = 3;
    cv::bilateralFilter(normal, normal_smoothed, kernel_size, kernel_size * 2, kernel_size / 2);
    cv::swap(normal, normal_smoothed);*/
    for (int i = 0; i < 3; ++i)
    {
        const int kernel_size = 3;
        cv::bilateralFilter(normal, normal_smoothed, kernel_size, kernel_size * 2, kernel_size / 2);
        cv::swap(normal, normal_smoothed);
    }
    edgemask = EdgeDetect(normal, 5, 0.9f, true);
    for (int i = 0; i < 3; ++i)
    {
        /*const int kernel_size = 3;
        cv::bilateralFilter(edgemask, edgemask_smoothed, kernel_size, kernel_size * 2, kernel_size / 2);*/
        cv::medianBlur(edgemask, edgemask_smoothed, 3);
        cv::swap(edgemask, edgemask_smoothed);
    }
    const int dilation_size = 6;
    cv::Mat patches = SegmentPatches(edgemask);
    cv::Mat patch_dialate;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS,
        cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        cv::Point(dilation_size, dilation_size));
    /// Apply the dilation operation
    dilate(patches, patch_dialate, element);

    cv::Mat normalmask;
    GetPlanesMask(normal, patch_dialate, normalmask);

    //mat = Depth2Color(dst);
    cv::imshow("Depth", normal);
    cv::imshow("Edge Mask", edgemask);
    cv::imshow("Patches", patch_dialate);
    cv::imshow("Planes Masks", normalmask);
}

HRESULT InitDefaultSensor()
{
    INuiSensor * pNuiSensor;
    HRESULT hr;

    int sensorcount = 0;
    hr = NuiGetSensorCount(&sensorcount);
    if (FAILED(hr))
    {
        return hr;
    }
    // Look at each Kinect sensor
    for (int i = 0; i < sensorcount; ++i)
    {
        // Create the sensor so we can check status, if we can't create it, move on to the next
        hr = NuiCreateSensorByIndex(i, &pNuiSensor);
        if (FAILED(hr))
        {
            continue;
        }

        // Get the status of the sensor, and if connected, then we can initialize it
        hr = pNuiSensor->NuiStatus();
        if (S_OK == hr)
        {
            nuisensor = pNuiSensor;
            break;
        }

        // This sensor wasn't OK, so release it since we're not using it
        pNuiSensor->Release();
    }

    if (NULL != nuisensor)
    {
        // Initialize the Kinect and specify that we'll be using depth
        hr = nuisensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_DEPTH);
        if (SUCCEEDED(hr))
        {
            // Create an event that will be signaled when depth data is available
            event_nextdepthframe = CreateEvent(NULL, TRUE, FALSE, NULL);

            // Open a depth image stream to receive depth frames
            hr = nuisensor->NuiImageStreamOpen(
                NUI_IMAGE_TYPE_DEPTH,
                NUI_IMAGE_RESOLUTION_640x480,
                0,
                2,
                event_nextdepthframe,
                &stream_depth);
        }
    }

    if (NULL == nuisensor || FAILED(hr))
    {
        printf("No ready Kinect found!\n");
        return E_FAIL;
    }

    return hr;
}


void Update()
{
    if (NULL == nuisensor)
    {
        return;
    }

    if (WAIT_OBJECT_0 == WaitForSingleObject(event_nextdepthframe, 0))
    {
        HRESULT hr;
        NUI_IMAGE_FRAME imgframe;
        // Attempt to get the depth frame
        do
        {
            hr = nuisensor->NuiImageStreamGetNextFrame(stream_depth, 0, &imgframe);
            if (FAILED(hr))
            {
                return;
            }
            BOOL nearmode = true;
            INuiFrameTexture* texture;
            // Get the depth image pixel texture
            hr = nuisensor->NuiImageFrameGetDepthImagePixelFrameTexture(
                stream_depth, &imgframe, &nearmode, &texture);
            if (FAILED(hr))
            {
                break;
            }

            NUI_LOCKED_RECT LockedRect;

            // Lock the frame data so the Kinect knows not to modify it while we're reading it
            texture->LockRect(0, &LockedRect, NULL, 0);
            // Make sure we've received valid data
            if (LockedRect.Pitch != 0)
            {
                // Get the min and max reliable depth for the current frame
                int minDepth = (nearmode ? NUI_IMAGE_DEPTH_MINIMUM_NEAR_MODE : NUI_IMAGE_DEPTH_MINIMUM) >> NUI_IMAGE_PLAYER_INDEX_SHIFT;
                int maxDepth = (nearmode ? NUI_IMAGE_DEPTH_MAXIMUM_NEAR_MODE : NUI_IMAGE_DEPTH_MAXIMUM) >> NUI_IMAGE_PLAYER_INDEX_SHIFT;

                const NUI_DEPTH_IMAGE_PIXEL * pBufferRun = reinterpret_cast<const NUI_DEPTH_IMAGE_PIXEL *>(LockedRect.pBits);

                // end pixel is start + width*height - 1
                const NUI_DEPTH_IMAGE_PIXEL * pBufferEnd = pBufferRun + (width_depth * height_depth);

                cv::Mat mat_depth(height_depth, width_depth, CV_16UC1);

                size_t index = 0;
                while (pBufferRun < pBufferEnd)
                {
                    // discard the portion of the depth that contains only the player index
                    USHORT depth = pBufferRun->depth;

                    // To convert to a byte, we're discarding the most-significant
                    // rather than least-significant bits.
                    // We're preserving detail, although the intensity will "wrap."
                    // Values outside the reliable depth range are mapped to 0 (black).

                    // Note: Using conditionals in this loop could degrade performance.
                    // Consider using a lookup table instead when writing production code.
                    depth = (depth >= minDepth && depth <= maxDepth ? depth : 0);

                    mat_depth.at<ushort>(index / width_depth, index % width_depth) = depth;


                    // We're outputting BGR, the last byte in the 32 bits is unused so skip it
                    // If we were outputting BGRA, we would write alpha here.
                    ++index;
                    // Increment our index into the Kinect's depth buffer
                    ++pBufferRun;
                }

                ProcessDepth(0, mat_depth, width_depth, height_depth, minDepth, maxDepth);

                // We're done with the texture so unlock it
                texture->UnlockRect(0);
                texture->Release();
            }
        } while (0);
        nuisensor->NuiImageStreamReleaseFrame(stream_depth, &imgframe);
    }
}

#endif //!USE_KINECT1

#if defined USE_REALSENSE

void ProcessDepth(long long _time, const cv::Mat& _img, int _width, int _height, unsigned short _mindepth, unsigned short _maxdepth)
{
    cv::Mat depth = _img;
    cv::Mat dst = cv::Mat(_height, _width, CV_16UC1);
    cv::Mat normal = cv::Mat(_height, _width, CV_32FC3);
    cv::Mat normal_smoothed = cv::Mat(_height, _width, CV_32FC3);
    cv::Mat edgemask, edgemask_smoothed, depth_smoothed;
    //depth = MedianFilter(depth);
    //depth.convertTo(depth_smoothed, CV_32FC1);


    // Only contains 256cm ~ 1024cm depths
    
    depth = ClipDistance(depth, 0, 3000);

    normal = CalcuNormal(depth);

    //const int kernel_size = 3;
    //cv::bilateralFilter(normal, normal_smoothed, kernel_size, kernel_size * 2, kernel_size / 2);
    //cv::imshow("bilateral smoothed", normal_smoothed);
   /* const int kernel_size = 3;
    cv::bilateralFilter(normal, normal_smoothed, kernel_size, kernel_size * 2, kernel_size / 2);
    cv::swap(normal, normal_smoothed);*/
    for (int i = 0; i < 3; ++i)
    {
        const int kernel_size = 3;
        cv::bilateralFilter(normal, normal_smoothed, kernel_size, kernel_size * 2, kernel_size / 2);
        cv::swap(normal, normal_smoothed);
    }
    edgemask = EdgeDetect(normal, 5, 0.9f, true);
    for (int i = 0; i < 3; ++i)
    {
        /*const int kernel_size = 3;
        cv::bilateralFilter(edgemask, edgemask_smoothed, kernel_size, kernel_size * 2, kernel_size / 2);*/
        cv::medianBlur(edgemask, edgemask_smoothed, 3);
        cv::swap(edgemask, edgemask_smoothed);
    }
    const int dilation_size = 6;
    cv::Mat patches = SegmentPatches(edgemask);
    cv::Mat patch_dialate;
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS,
        cv::Size(2 * dilation_size + 1, 2 * dilation_size + 1),
        cv::Point(dilation_size, dilation_size));
    /// Apply the dilation operation
    cv::dilate(patches, patch_dialate, element);

    cv::Mat normalmask;
    GetPlanesMask(normal, patch_dialate, normalmask);

    //mat = Depth2Color(dst);
    cv::imshow("Depth", depth * 100);
    //cv::imshow("Edge Mask", edgemask);
    //cv::imshow("Patches", patch_dialate);
    cv::imshow("Planes Masks", normalmask);
}

int InitDefaultSensor(void)
{
    pp = PXCSenseManager::CreateInstance();

    if (!pp) {
        printf("Unable to create the SenseManager\n");
        return -1;
    }

    pp->EnableStream(PXCCapture::STREAM_TYPE_DEPTH, width_depth, height_depth);

    auto sts = pp->Init();
    if (sts != PXC_STATUS_NO_ERROR)
    {
        printf("Unabel to Initializes the pipeline\n");
        return -1;
    }

    return 0;
}

void ConvertPXCImageToOpenCVMat(PXCImage *inImg, cv::Mat *outImg) {
    int cvDataType;
    int cvDataWidth;

    PXCImage::ImageData data;
    inImg->AcquireAccess(PXCImage::ACCESS_READ, &data);
    PXCImage::ImageInfo imgInfo = inImg->QueryInfo();

    switch (data.format) {
        /* STREAM_TYPE_COLOR */
    case PXCImage::PIXEL_FORMAT_YUY2: /* YUY2 image  */
    case PXCImage::PIXEL_FORMAT_NV12: /* NV12 image */
        throw(0); // Not implemented
    case PXCImage::PIXEL_FORMAT_RGB32: /* BGRA layout on a little-endian machine */
        cvDataType = CV_8UC4;
        cvDataWidth = 4;
        break;
    case PXCImage::PIXEL_FORMAT_RGB24: /* BGR layout on a little-endian machine */
        cvDataType = CV_8UC3;
        cvDataWidth = 3;
        break;
    case PXCImage::PIXEL_FORMAT_Y8:  /* 8-Bit Gray Image, or IR 8-bit */
        cvDataType = CV_8U;
        cvDataWidth = 1;
        break;

        /* STREAM_TYPE_DEPTH */
    case PXCImage::PIXEL_FORMAT_DEPTH: /* 16-bit unsigned integer with precision mm. */
    case PXCImage::PIXEL_FORMAT_DEPTH_RAW: /* 16-bit unsigned integer with device specific precision (call device->QueryDepthUnit()) */
        cvDataType = CV_16U;
        cvDataWidth = 2;
        break;
    case PXCImage::PIXEL_FORMAT_DEPTH_F32: /* 32-bit float-point with precision mm. */
        cvDataType = CV_32F;
        cvDataWidth = 4;
        break;

        /* STREAM_TYPE_IR */
    case PXCImage::PIXEL_FORMAT_Y16:          /* 16-Bit Gray Image */
        cvDataType = CV_16U;
        cvDataWidth = 2;
        break;
    case PXCImage::PIXEL_FORMAT_Y8_IR_RELATIVE:    /* Relative IR Image */
        cvDataType = CV_8U;
        cvDataWidth = 1;
        break;
    }

    // suppose that no other planes
    if (data.planes[1] != NULL) throw(0); // not implemented
    // suppose that no sub pixel padding needed
    if (data.pitches[0] % cvDataWidth != 0) throw(0); // not implemented

    outImg->create(imgInfo.height, data.pitches[0] / cvDataWidth, cvDataType);

    memcpy(outImg->data, data.planes[0], imgInfo.height*imgInfo.width*cvDataWidth * sizeof(pxcBYTE));

    inImg->ReleaseAccess(&data);
}

void Update()
{
    /* Stream Data */
    if (pp->AcquireFrame(true) >= PXC_STATUS_NO_ERROR) 
    {
        /* Render streams, unless -noRender is selected */
        const PXCCapture::Sample *sample = pp->QuerySample();
        if (sample) 
        {
            depthIm = sample->depth;
            if (depthIm->AcquireAccess(PXCImage::ACCESS_READ, &depth_data) < PXC_STATUS_NO_ERROR)
            {
                printf("Cant access depth image\n");
                return;
            }
            depth_info = sample->depth->QueryInfo();

            /* Releases lock so pipeline can process next frame */
            cv::Mat depth(cv::Size(depth_info.width, depth_info.height), CV_16UC1, (void*)depth_data.planes[0], depth_data.pitches[0] / sizeof(uchar));
            ProcessDepth(0, depth, width_depth, height_depth, 0, USHRT_MAX);
            depthIm->ReleaseAccess(&depth_data);
            pp->ReleaseFrame();

        }
    }
}


#endif


int main(int _argc, char** _argv)
{
    int hr = (int)InitDefaultSensor();
    if (hr < 0)
    {
        return -1;
    }

    while (1)
    {
        Update();

        int keyPressed = cv::waitKey(2);
        if (keyPressed == 32) // space bar to quit
            break;
    }

#if defined USE_KINECT2
    SafeRelease(kinect);
    SafeRelease(dfr);
#endif //!USE_KINECT2
#if defined USE_KINECT1
    nuisensor->NuiShutdown();
    CloseHandle(event_nextdepthframe);
    SafeRelease(nuisensor);
#endif //!USE_KINECT1
    return 0;
}