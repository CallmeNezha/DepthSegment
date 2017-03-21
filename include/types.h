/* #undef USE_KINECT2 */
/* #undef USE_KINECT1 */
#define USE_REALSENSE

#if defined USE_KINECT1
#include <windows.h>
#include <NuiApi.h>
#elif defined USE_KINECT2
#include <Kinect.h>
#elif defined USE_REALSENSE
#include <pxcsensemanager.h>
#include <pxcmetadata.h>
#endif


