#include <jni.h>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/xfeatures2d.hpp>

#include<vector>

using namespace std;
using namespace cv;

extern "C"
JNIEXPORT void JNICALL Java_com_example_muneeb_mobandweb_MainActivity_FindMatches(JNIEnv*, jobject, jlong objectAddress, jlong sceneAddress,jint detectorID, jint descriptorID,jlong matchingResult)
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

    //Construct a detector object based on the input ID
    if(detectorID==1)//FAST
    {

        Ptr<cv::Feature2D> detector = FastFeatureDetector::create();
        detector->detect(grayObject, objectKeyPoints);
        detector->detect(grayScene, sceneKeyPoints);
    }

    else if(detectorID == 2)
    {
        Ptr<cv::Feature2D> detector = AgastFeatureDetector::create();
        detector->detect(grayObject, objectKeyPoints);
        detector->detect(grayScene, sceneKeyPoints);
    }

   else if (detectorID == 3)
    {
        Ptr<cv::Feature2D> detector = MSER::create();
        detector->detect(grayObject, objectKeyPoints);
        detector->detect(grayScene, sceneKeyPoints);
    }

    else if (detectorID == 4)
    {
        Ptr<cv::Feature2D> detector = GFTTDetector::create();
        detector->detect(grayObject, objectKeyPoints);
        detector->detect(grayScene, sceneKeyPoints);
    }

    else if(detectorID==5)
    {
        cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
        detector->detect(grayObject, objectKeyPoints);
        detector->detect(grayScene, sceneKeyPoints);
    }

    else if(detectorID==6)
    {
        cv::Ptr<Feature2D> detector = KAZE::create();
        detector->detect(grayObject, objectKeyPoints);
        detector->detect(grayScene, sceneKeyPoints);
    }

    else if(detectorID == 7)
    {
        cv::Ptr<Feature2D> detector = AKAZE::create();
        detector->detect(grayObject, objectKeyPoints);
        detector->detect(grayScene, sceneKeyPoints);
    }

    else if(detectorID == 8)
    {
        cv::Ptr<Feature2D> detector = xfeatures2d::StarDetector::create();
        detector->detect(grayObject, objectKeyPoints);
        detector->detect(grayScene, sceneKeyPoints);
    }

    else if(detectorID == 9)
    {
        cv::Ptr<Feature2D> detector = xfeatures2d::SiftFeatureDetector::create(200);
        detector->detect(grayObject, objectKeyPoints);
        detector->detect(grayScene, sceneKeyPoints);
    }

    else if(detectorID == 10)
    {
        cv::Ptr<Feature2D> detector = xfeatures2d::SurfFeatureDetector::create(200);
        detector->detect(grayObject, objectKeyPoints);
        detector->detect(grayScene, sceneKeyPoints);
    }

    if(descriptorID==1)
    {
        cv::Ptr<cv::Feature2D> descriptor = KAZE::create();
        descriptor->compute(grayObject,objectKeyPoints,objectDescriptor);
        descriptor->compute(grayScene,sceneKeyPoints,scenceDescriptor);
    }

    if(descriptorID == 2)
    {
        cv::Ptr<cv::Feature2D> descriptor = AKAZE::create();
        descriptor->compute(grayObject,objectKeyPoints,objectDescriptor);
        descriptor->compute(grayScene,sceneKeyPoints,scenceDescriptor);
    }

    //Construct a descriptor object based on the input ID
    if(descriptorID==3)//ORB
    {
        cv::Ptr<cv::Feature2D> descriptor = cv::ORB::create();
        descriptor->compute(grayObject,objectKeyPoints,objectDescriptor);
        descriptor->compute(grayScene,sceneKeyPoints,scenceDescriptor);
    }


    else if(descriptorID==4)
    {
        cv::Ptr<Feature2D> descriptor = xfeatures2d::FREAK::create();
        descriptor->compute(grayObject,objectKeyPoints,objectDescriptor);
        descriptor->compute(grayScene,sceneKeyPoints,scenceDescriptor);

    }

    else if(descriptorID==5)//BRISK
    {
        cv::Ptr<Feature2D> descriptor = BRISK::create();
        descriptor->compute(grayObject,objectKeyPoints,objectDescriptor);
        descriptor->compute(grayScene,sceneKeyPoints,scenceDescriptor);

    }

    else if(descriptorID==6)
    {
        cv::Ptr<Feature2D> descriptor = xfeatures2d::BriefDescriptorExtractor::create();
        descriptor->compute(grayObject,objectKeyPoints,objectDescriptor);
        descriptor->compute(grayScene,sceneKeyPoints,scenceDescriptor);

    }
    else if(descriptorID == 7)
    {
        cv::Ptr<Feature2D> descriptor = xfeatures2d::LUCID::create();
        descriptor->compute(grayObject,objectKeyPoints,objectDescriptor);
        descriptor->compute(grayScene,sceneKeyPoints,scenceDescriptor);

    }
    else if(descriptorID==8)
    {
        cv::Ptr<Feature2D> descriptor = xfeatures2d::LATCH::create();
        descriptor->compute(grayObject,objectKeyPoints,objectDescriptor);
        descriptor->compute(grayScene,sceneKeyPoints,scenceDescriptor);

    }
    else if(descriptorID==9)
    {
        cv::Ptr<Feature2D> descriptor = xfeatures2d::DAISY::create();
        descriptor->compute(grayObject,objectKeyPoints,objectDescriptor);
        descriptor->compute(grayScene,sceneKeyPoints,scenceDescriptor);

    }
    else if(descriptorID==10)
    {
        cv::Ptr<Feature2D> descriptor = xfeatures2d::SiftDescriptorExtractor::create();
        descriptor->compute(grayObject,objectKeyPoints,objectDescriptor);
        descriptor->compute(grayScene,sceneKeyPoints,scenceDescriptor);

    }
    else if(descriptorID==11)
    {
        cv::Ptr<Feature2D> descriptor = xfeatures2d::SurfDescriptorExtractor::create();
        descriptor->compute(grayObject,objectKeyPoints,objectDescriptor);
        descriptor->compute(grayScene,sceneKeyPoints,scenceDescriptor);

    }

    // SIFT AND SURF
/*

    //Construct a brute force matcher object using the
    //Hamming distance as the distance function
    cv::BFMatcher bfmatcher(cv::NormTypes::NORM_L1, false);

    cv::FlannBasedMatcher fmatcher;
    std::vector<std::vector< cv::DMatch > > matches;

    fmatcher.knnMatch(objectDescriptor, scenceDescriptor,matches,2);
    //bfmatcher.knnMatch( objectDescriptor, scenceDescriptor, matches,2);

    string retVal;
    ostringstream convert;
    convert << objectDescriptor.rows;
    retVal = convert.str();


    std::vector< cv::DMatch> good_matches;
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

*/


    // ORB
    /*
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
    std::vector< DMatch > good_matches;
    for ( int i = 0; i < objectDescriptor.rows; i++ )
    {
        if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            good_matches.push_back ( matches[i] );
        }
    }

     */



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