# mobandweb
This is academic project done as part of Mobile and Web Development course at HS Fulda. The project revolves around 
Features Matching in images.

Following algorithms (feature detectors) have been implemented to generate keypoints in a particular image and display it: 
Harris, FAST, ORB, AGAST, MSER, GFTTD, KAZE, AKAZE, STAR, SIFT and SURF

Following algorithms have been implemented as a descriptor selection choice for feature matching: 
KAZE, AKAZE, ORB, FREAK, BRISK, BRIEF, LUCID, LATCH, DAISY, SIFT and SURF

FlannBased, BruteForce L1 and BruteForce L2 matchers have been implemented to match float descriptors 
while BruteForce Hamming and BruteForce HammingLUT have been implemented to match uchar descriptors.

Image stitching has been implemented using openCV function as well as using custom technique. 
The custom implementation of the image stitcher is achieved by using SURF feature detector and descriptor, 
along with FlannBased matcher to find the matches between the images. After the matches are found, we find the homography 
by using findhomography function of openCV. We use homography matrix to warp the matching images. 
Functions to load, process and display images using mat object has also been implemented as part of the app.
