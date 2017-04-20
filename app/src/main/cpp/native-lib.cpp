#include <jni.h>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include<vector>

using namespace std;
using namespace cv;

extern "C"
JNIEXPORT void JNICALL Java_com_example_muneeb_mobandweb_MainActivity_FindMatches(JNIEnv*, jobject, jlong objectAddress, jlong sceneAddress,jint detectorID, jint descriptorID,jint matcherID,jlong matchingResult)
{
    cv::Mat& object  = *(cv::Mat*)objectAddress;
    cv::Mat& scene = *(cv::Mat*)sceneAddress;
    cv::Mat& result = *(cv::Mat*)matchingResult;
    cv::Mat grayObject;
    cv::Mat grayScene;

    //Convert the object and scene image to grayscale
    cv::cvtColor(object,grayObject,cv::COLOR_RGBA2GRAY);
    cv::cvtColor(scene,grayScene,cv::COLOR_RGBA2GRAY);

    std::vector<cv::KeyPoint> objectKeyPoints;
    std::vector<cv::KeyPoint> sceneKeyPoints;
    cv::Mat objectDescriptor;
    cv::Mat scenceDescriptor;
    cv::Ptr<Feature2D> descriptor;
    Ptr<cv::Feature2D> detector;

    //Construct a detector object based on the input ID
    if(detectorID==1)//FAST
    {

        detector = FastFeatureDetector::create();
    }

    else if(detectorID == 2)
    {
        detector = AgastFeatureDetector::create();
    }

   else if (detectorID == 3)
    {
        detector = MSER::create();
    }

    else if (detectorID == 4)
    {
        detector = GFTTDetector::create();
    }


    else if(detectorID==5)
    {
         detector = cv::ORB::create();
    }

    else if(detectorID==6)
    {
        detector = KAZE::create();

    }

    else if(detectorID == 7)
    {
        detector = AKAZE::create();

    }

    else if(detectorID == 8)
    {
         detector = xfeatures2d::StarDetector::create();
    }

    else if(detectorID == 9)
    {
        detector = xfeatures2d::SiftFeatureDetector::create(200);
    }

    else if(detectorID == 10)
    {
        detector = xfeatures2d::SurfFeatureDetector::create(200);
    }

    //For whichever detector is selected
    detector->detect(grayObject, objectKeyPoints);
    detector->detect(grayScene, sceneKeyPoints);

    if(descriptorID==1)
    {
        descriptor = KAZE::create();

    }

    if(descriptorID == 2)
    {
        descriptor = AKAZE::create();
    }

    //Construct a descriptor object based on the input ID
    if(descriptorID==3)//ORB
    {
        descriptor = cv::ORB::create();
    }


    else if(descriptorID==4)
    {
        descriptor = xfeatures2d::FREAK::create();
    }

    else if(descriptorID==5)//BRISK
    {
        descriptor = BRISK::create();
    }

    else if(descriptorID==6)
    {
        descriptor = xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if(descriptorID == 7)
    {
        descriptor = xfeatures2d::LUCID::create();
    }
    else if(descriptorID==8)
    {
        descriptor = xfeatures2d::LATCH::create();
    }
    else if(descriptorID==9)
    {
        descriptor = xfeatures2d::DAISY::create();
    }
    else if(descriptorID==10)
    {
        descriptor = xfeatures2d::SiftDescriptorExtractor::create();

    }
    else if(descriptorID==11)
    {
         descriptor = xfeatures2d::SurfDescriptorExtractor::create();
    }

    descriptor->compute(grayObject,objectKeyPoints,objectDescriptor);
    descriptor->compute(grayScene,sceneKeyPoints,scenceDescriptor);

    std::vector< DMatch > good_matches;


    // SIFT AND SURF
if(matcherID == 1) //FlannBased
{
    //cv::BFMatcher bfmatcher(cv::NormTypes::NORM_L1, false);

    cv::FlannBasedMatcher fmatcher;
    std::vector<std::vector< cv::DMatch > > matches;

    fmatcher.knnMatch(objectDescriptor, scenceDescriptor,matches,2);
    //bfmatcher.knnMatch( objectDescriptor, scenceDescriptor, matches,2);

    string retVal;
    ostringstream convert;
    convert << objectDescriptor.rows;
    retVal = convert.str();


    //-- Step 4: Select only goot matches

    for (int k = 0; k < std::min(scenceDescriptor.rows - 1, (int)matches.size()); k++)
    {
        if ( (matches[k][0].distance < 0.6*(matches[k][1].distance)) &&
             ((int)matches[k].size() <= 2 && (int)matches[k].size()>0) )
        {
            // take the first result only if its distance is smaller than 0.6*second_best_dist
            // that means this descriptor is ignored if the second distance is bigger or of similar
            good_matches.push_back( matches[k][0] );
        }
    }
}


    if(matcherID == 2) //Brute L1
    {
        cv::BFMatcher bfmatcher(cv::NormTypes::NORM_L1, false);

        //cv::FlannBasedMatcher fmatcher;
        std::vector<std::vector< cv::DMatch > > matches;

        //fmatcher.knnMatch(objectDescriptor, scenceDescriptor,matches,2);
        bfmatcher.knnMatch( objectDescriptor, scenceDescriptor, matches,2);

        string retVal;
        ostringstream convert;
        convert << objectDescriptor.rows;
        retVal = convert.str();


        //-- Step 4: Select only goot matches

        for (int k = 0; k < std::min(scenceDescriptor.rows - 1, (int)matches.size()); k++)
        {
            if ( (matches[k][0].distance < 0.6*(matches[k][1].distance)) &&
                 ((int)matches[k].size() <= 2 && (int)matches[k].size()>0) )
            {
                // take the first result only if its distance is smaller than 0.6*second_best_dist
                // that means this descriptor is ignored if the second distance is bigger or of similar
                good_matches.push_back( matches[k][0] );
            }
        }
    }

    if(matcherID == 3) //Brute L2
    {
        cv::BFMatcher bfmatcher(cv::NormTypes::NORM_L2, false);

        //cv::FlannBasedMatcher fmatcher;
        std::vector<std::vector< cv::DMatch > > matches;

        //fmatcher.knnMatch(objectDescriptor, scenceDescriptor,matches,2);
        bfmatcher.knnMatch( objectDescriptor, scenceDescriptor, matches,2);

        string retVal;
        ostringstream convert;
        convert << objectDescriptor.rows;
        retVal = convert.str();


        //-- Step 4: Select only goot matches

        for (int k = 0; k < std::min(scenceDescriptor.rows - 1, (int)matches.size()); k++)
        {
            if ( (matches[k][0].distance < 0.6*(matches[k][1].distance)) &&
                 ((int)matches[k].size() <= 2 && (int)matches[k].size()>0) )
            {
                // take the first result only if its distance is smaller than 0.6*second_best_dist
                // that means this descriptor is ignored if the second distance is bigger or of similar
                good_matches.push_back( matches[k][0] );
            }
        }
    }

    if(matcherID == 4) //Hamming
    {
        cv::BFMatcher bfmatcher(cv::NormTypes::NORM_HAMMING, false);
        vector<DMatch> matches;
        bfmatcher.match ( objectDescriptor, scenceDescriptor, matches );
        double min_dist=10000, max_dist=0;
        for ( int i = 0; i < objectDescriptor.rows; i++ )
        {
            double dist = matches[i].distance;
            if ( dist < min_dist ) min_dist = dist;
            if ( dist > max_dist ) max_dist = dist;
        }
        for ( int i = 0; i < objectDescriptor.rows; i++ )
        {
            if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) )
            {
                good_matches.push_back ( matches[i] );
            }
        }
    }

    if(matcherID == 5) //HammingLUT
    {
        cv::BFMatcher bfmatcher(cv::NormTypes::NORM_HAMMING2, false);
        vector<DMatch> matches;
        bfmatcher.match ( objectDescriptor, scenceDescriptor, matches );
        double min_dist=10000, max_dist=0;
        for ( int i = 0; i < objectDescriptor.rows; i++ )
        {
            double dist = matches[i].distance;
            if ( dist < min_dist ) min_dist = dist;
            if ( dist > max_dist ) max_dist = dist;
        }
        for ( int i = 0; i < objectDescriptor.rows; i++ )
        {
            if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) )
            {
                good_matches.push_back ( matches[i] );
            }
        }
    }

    drawMatches( grayObject, objectKeyPoints, grayScene, sceneKeyPoints,good_matches, result, cv::Scalar::all(-1), cv::Scalar::all(-1),std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS+cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

}

extern "C"
JNIEXPORT void JNICALL Java_com_example_muneeb_mobandweb_MainActivity_Stitch(JNIEnv*, jobject, jlong sceneOneAddress, jlong sceneTwoAddress,jlong stitchingResult) {
    cv::Mat& sceneOne  = *(cv::Mat*)sceneOneAddress;
    cv::Mat& sceneTwo = *(cv::Mat*)sceneTwoAddress;
    cv::Mat& result = *(cv::Mat*)stitchingResult;
    /* The core stitching calls: */
    //a list to store all the images that need to be stitched
    std::vector<cv::Mat> natImgs;
    natImgs.push_back(sceneOne);
    natImgs.push_back(sceneTwo);
    //create a stitcher object with the default pipeline
    cv::Stitcher stitcher = cv::Stitcher::createDefault();
    //stitch and return the result
    stitcher.stitch(natImgs, result);
}


extern "C"
JNIEXPORT void JNICALL Java_com_example_muneeb_mobandweb_MainActivity_NativeStich(JNIEnv*, jobject, jlong objectAddress, jlong sceneAddress,jint detectorID, jint descriptorID,jlong matchingResult)
{

    cv::Mat& object  = *(cv::Mat*)objectAddress;
    cv::Mat& scene = *(cv::Mat*)sceneAddress;
    cv::Mat& result = *(cv::Mat*)matchingResult;
    cv::Mat grayObject;
    cv::Mat grayScene;

    //Convert the object and scene image to grayscale
     cv::cvtColor(object,grayObject,cv::COLOR_RGBA2GRAY);
     cv::cvtColor(scene,grayScene,cv::COLOR_RGBA2GRAY);

    int minHessianVal = 400;

    cv::Ptr<Feature2D> detector = xfeatures2d::SurfFeatureDetector::create(minHessianVal);

    std::vector<KeyPoint> keypoints_object, keypoints_scene;

    detector->detect(grayObject, keypoints_object);
    detector->detect(grayScene, keypoints_scene);


    cv::Ptr<Feature2D> descriptor = xfeatures2d::SurfDescriptorExtractor::create();

    Mat descriptors_object, descriptors_scene;

    descriptor->compute(grayObject, keypoints_object, descriptors_object);
    descriptor->compute(grayScene, keypoints_scene, descriptors_scene);

    FlannBasedMatcher matcher;
    std::vector<DMatch> matches;
    matcher.match(descriptors_object, descriptors_scene, matches);

    double max_dist = 0;
    double min_dist = 100;

    for (int i = 0; i < descriptors_object.rows; i++) {
        double dist = matches[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    std::vector<DMatch> good_matches;

    for (int i = 0; i < descriptors_object.rows; i++) {
        if (matches[i].distance < 3 * min_dist) { good_matches.push_back(matches[i]); }
    }



    std::vector<Point2f> obj;
    std::vector<Point2f> sceneP;

    for (int i = 0; i < good_matches.size(); i++) {

        obj.push_back(keypoints_object[good_matches[i].queryIdx].pt);
        sceneP.push_back(keypoints_scene[good_matches[i].trainIdx].pt);
    }


    Mat H = findHomography(obj, sceneP, CV_RANSAC);

// Use the Homography Matrix to warp the images
    warpPerspective(object, result, H, cv::Size(object.cols + scene.cols, object.rows));
    //cv::Mat half(result, cv::Rect(0, 0, scene.cols, scene.rows));
    //scene.copyTo(half);
}

extern "C"
JNIEXPORT void JNICALL Java_com_example_muneeb_mobandweb_MainActivity_FindHarrisCorners(JNIEnv*, jobject, jlong addrGray, jlong addrRgba)
{
    cv::Mat& mGr  = *(cv::Mat*)addrGray;
    cv::Mat& mRgb = *(cv::Mat*)addrRgba;

    cv::Mat dst_norm;
    cv::Mat dst = cv::Mat::zeros(mGr.size(),CV_32FC1);

    //the size of the neighbor in which we will check
    //the existence of a corner
    int blockSize = 2;
    //used for the Sobel kernel to detect edges before
    //checking for corners
    int apertureSize = 3;
    // a free constant used in Harris mathematical formula
    double k = 0.04;
    //corners response threshold
    float threshold=150;

    cv::cornerHarris( mGr, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT );

    cv::normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );

    for( int i = 0; i < dst_norm.rows; i++ )
    {
        float * row=dst_norm.ptr<float>(i);
        for(int j=0;j<dst_norm.cols;j++)
        {
            if(row[j]>=threshold)
            {
                cv::circle(mRgb, cv::Point(j, i), 10, cv::Scalar(255,0,0,255));
            }
        }
    }
}

extern "C"
JNIEXPORT void JNICALL Java_com_example_muneeb_mobandweb_MainActivity_FindFastFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba)
{
    cv::Mat& mGr  = *(cv::Mat*)addrGray;
    cv::Mat& mRgb = *(cv::Mat*)addrRgba;

    vector<KeyPoint> keypointsD;
    Ptr<FastFeatureDetector> detector=FastFeatureDetector::create();
    detector->detect(mGr, keypointsD);
    for( int i = 0; i < keypointsD.size(); i++ )
    {
        const cv::KeyPoint& kp = keypointsD[i];
        cv::circle(mRgb, cv::Point(kp.pt.x, kp.pt.y), 10, cv::Scalar(255,0,0,255));
    }
}

extern "C"
JNIEXPORT void JNICALL Java_com_example_muneeb_mobandweb_MainActivity_FindORBFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba)
{
    cv::Mat& object = *(cv::Mat*)addrRgba;
    cv::Mat& grayObject = *(cv::Mat*)addrGray;
    Ptr<Feature2D> detector= ORB::create();
    std::vector<KeyPoint> keypoints_object;
    cv::cvtColor(object,grayObject,cv::COLOR_RGBA2GRAY);
    detector->detect(grayObject, keypoints_object);

    for( int i = 0; i < keypoints_object.size(); i++ )
    {
        const cv::KeyPoint& kp = keypoints_object[i];
        cv::circle(object, cv::Point(kp.pt.x, kp.pt.y), 10, cv::Scalar(255,0,0,255));
    }

}

extern "C"
JNIEXPORT void JNICALL Java_com_example_muneeb_mobandweb_MainActivity_AgastFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba)
{
    cv::Mat& object = *(cv::Mat*)addrRgba;
    cv::Mat& grayObject = *(cv::Mat*)addrGray;
    Ptr<Feature2D> detector= AgastFeatureDetector::create();
    std::vector<KeyPoint> keypoints_object;
    cv::cvtColor(object,grayObject,cv::COLOR_RGBA2GRAY);
    detector->detect(grayObject, keypoints_object);

    for( int i = 0; i < keypoints_object.size(); i++ )
    {
        const cv::KeyPoint& kp = keypoints_object[i];
        cv::circle(object, cv::Point(kp.pt.x, kp.pt.y), 10, cv::Scalar(255,0,0,255));
    }

}

extern "C"
JNIEXPORT void JNICALL Java_com_example_muneeb_mobandweb_MainActivity_MserFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba)
{
    cv::Mat& object = *(cv::Mat*)addrRgba;
    cv::Mat& grayObject = *(cv::Mat*)addrGray;
    Ptr<Feature2D> detector= MSER::create();
    std::vector<KeyPoint> keypoints_object;
    cv::cvtColor(object,grayObject,cv::COLOR_RGBA2GRAY);
    detector->detect(grayObject, keypoints_object);

    for( int i = 0; i < keypoints_object.size(); i++ )
    {
        const cv::KeyPoint& kp = keypoints_object[i];
        cv::circle(object, cv::Point(kp.pt.x, kp.pt.y), 10, cv::Scalar(255,0,0,255));
    }

}

extern "C"
JNIEXPORT void JNICALL Java_com_example_muneeb_mobandweb_MainActivity_GfttdFeatures(JNIEnv*, jobject, jlong addrGray, jlong addrRgba)
{
    cv::Mat& object = *(cv::Mat*)addrRgba;
    cv::Mat& grayObject = *(cv::Mat*)addrGray;
    Ptr<Feature2D> detector= GFTTDetector::create();
    std::vector<KeyPoint> keypoints_object;
    cv::cvtColor(object,grayObject,cv::COLOR_RGBA2GRAY);
    detector->detect(grayObject, keypoints_object);

    for( int i = 0; i < keypoints_object.size(); i++ )
    {
        const cv::KeyPoint& kp = keypoints_object[i];
        cv::circle(object, cv::Point(kp.pt.x, kp.pt.y), 10, cv::Scalar(255,0,0,255));
    }

}

extern "C"
JNIEXPORT void JNICALL Java_com_example_muneeb_mobandweb_MainActivity_Kaze(JNIEnv*, jobject, jlong addrGray, jlong addrRgba)
{
    cv::Mat& object = *(cv::Mat*)addrRgba;
    cv::Mat& grayObject = *(cv::Mat*)addrGray;
    Ptr<Feature2D> detector= KAZE::create();
    std::vector<KeyPoint> keypoints_object;
    cv::cvtColor(object,grayObject,cv::COLOR_RGBA2GRAY);
    detector->detect(grayObject, keypoints_object);

    for( int i = 0; i < keypoints_object.size(); i++ )
    {
        const cv::KeyPoint& kp = keypoints_object[i];
        cv::circle(object, cv::Point(kp.pt.x, kp.pt.y), 10, cv::Scalar(255,0,0,255));
    }

}

extern "C"
JNIEXPORT void JNICALL Java_com_example_muneeb_mobandweb_MainActivity_AKaze(JNIEnv*, jobject, jlong addrGray, jlong addrRgba)
{
    cv::Mat& object = *(cv::Mat*)addrRgba;
    cv::Mat& grayObject = *(cv::Mat*)addrGray;
    Ptr<Feature2D> detector= AKAZE::create();
    std::vector<KeyPoint> keypoints_object;
    cv::cvtColor(object,grayObject,cv::COLOR_RGBA2GRAY);
    detector->detect(grayObject, keypoints_object);

    for( int i = 0; i < keypoints_object.size(); i++ )
    {
        const cv::KeyPoint& kp = keypoints_object[i];
        cv::circle(object, cv::Point(kp.pt.x, kp.pt.y), 10, cv::Scalar(255,0,0,255));
    }

}

extern "C"
JNIEXPORT void JNICALL Java_com_example_muneeb_mobandweb_MainActivity_Star(JNIEnv*, jobject, jlong addrGray, jlong addrRgba)
{
    cv::Mat& object = *(cv::Mat*)addrRgba;
    cv::Mat& grayObject = *(cv::Mat*)addrGray;
    Ptr<Feature2D> detector= xfeatures2d::StarDetector::create();
    std::vector<KeyPoint> keypoints_object;
    cv::cvtColor(object,grayObject,cv::COLOR_RGBA2GRAY);
    detector->detect(grayObject, keypoints_object);

    for( int i = 0; i < keypoints_object.size(); i++ )
    {
        const cv::KeyPoint& kp = keypoints_object[i];
        cv::circle(object, cv::Point(kp.pt.x, kp.pt.y), 10, cv::Scalar(255,0,0,255));
    }

}

extern "C"
JNIEXPORT void JNICALL Java_com_example_muneeb_mobandweb_MainActivity_Sift(JNIEnv*, jobject, jlong addrGray, jlong addrRgba)
{
    cv::Mat& object = *(cv::Mat*)addrRgba;
    cv::Mat& grayObject = *(cv::Mat*)addrGray;
    Ptr<Feature2D> detector= xfeatures2d::SIFT::create();
    std::vector<KeyPoint> keypoints_object;
    cv::cvtColor(object,grayObject,cv::COLOR_RGBA2GRAY);
    detector->detect(grayObject, keypoints_object);

    for( int i = 0; i < keypoints_object.size(); i++ )
    {
        const cv::KeyPoint& kp = keypoints_object[i];
        cv::circle(object, cv::Point(kp.pt.x, kp.pt.y), 10, cv::Scalar(255,0,0,255));
    }

}

extern "C"
JNIEXPORT void JNICALL Java_com_example_muneeb_mobandweb_MainActivity_Surf(JNIEnv*, jobject, jlong addrGray, jlong addrRgba)
{
    cv::Mat& object = *(cv::Mat*)addrRgba;
    cv::Mat& grayObject = *(cv::Mat*)addrGray;
    Ptr<Feature2D> detector= xfeatures2d::SURF::create();
    std::vector<KeyPoint> keypoints_object;
    cv::cvtColor(object,grayObject,cv::COLOR_RGBA2GRAY);
    detector->detect(grayObject, keypoints_object);

    for( int i = 0; i < keypoints_object.size(); i++ )
    {
        const cv::KeyPoint& kp = keypoints_object[i];
        cv::circle(object, cv::Point(kp.pt.x, kp.pt.y), 10, cv::Scalar(255,0,0,255));
    }

}
