package com.example.muneeb.mobandweb;

import android.content.Context;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Point;
import android.media.ExifInterface;
import android.media.audiofx.AudioEffect;
import android.net.Uri;
import android.provider.MediaStore;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Display;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.DMatch;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.xfeatures2d.*;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }

    //A Tag to filter the log messages
    private static final String  TAG = "Example::MobAndWeb::Activity";

    private static int SELECT_PICTURE = 0;
    private static int SELECT_TARGET_PICTURE = 0;
    private Mat sampledImage;
    private Mat imgToMatch;
    private int descriptorID = 1;
    private int detectorID = 1;
    private String selectedImagePath;

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        try
        {


        if (id == R.id.fast_check || id == R.id.agastD_check || id == R.id.mser_check  || id == R.id.gfttd_check  ||
                id == R.id.orbD_check || id == R.id.kaze_check || id == R.id.akaze_check || id == R.id.star_check ||
                id == R.id.siftD_check || id == R.id.surfD_check)
        {
            item.setChecked(true);

            if(id == R.id.fast_check)
                detectorID = 1;

            else if(id == R.id.agastD_check)
                detectorID = 2;
            else if(id == R.id.mser_check)
                detectorID = 3;
            else if(id == R.id.gfttd_check)
                detectorID = 4;
            else if(id == R.id.orbD_check)
                detectorID = 5;
            else if(id == R.id.kaze_check)
                detectorID = 6;
            else if(id == R.id.akaze_check)
                detectorID = 7;
            else if(id == R.id.star_check)
                detectorID = 8;
            else if(id == R.id.siftD_check)
                detectorID = 9;
            else if(id == R.id.surfD_check)
                detectorID = 10;

            return true;
        }

        if (id == R.id.KAZEDes_check || id == R.id.AKAZEDes_check || id == R.id.ORBDes_check || id == R.id.FREAK_check
                || id == R.id.BRISK_check || id == R.id.BRIEF_check || id == R.id.LUCID_check || id == R.id.LATCH_check
                || id == R.id.DAISY_check || id == R.id.SIFTDes_check || id == R.id.SURFDes_check)
        {
            item.setChecked(true);

            if(id == R.id.KAZEDes_check)
                descriptorID = 1;

            else if(id == R.id.AKAZEDes_check)
                descriptorID = 2;

            else if(id == R.id.ORBDes_check)
                descriptorID = 3;

            else if(id == R.id.FREAK_check)
                descriptorID = 4;
            else if(id == R.id.BRISK_check)
                descriptorID = 5;

            else if(id == R.id.BRIEF_check)
                descriptorID = 6;

            else if(id == R.id.LUCID_check)
                descriptorID = 7;
            else if(id == R.id.LATCH_check)
                descriptorID = 8;

            else if(id == R.id.DAISY_check)
                descriptorID = 9;

            else if(id == R.id.SIFTDes_check)
                descriptorID = 10;
            else if(id == R.id.SURFDes_check)
                descriptorID = 11;


            return true;
        }


        if (id == R.id.action_openGallary) {
            Intent intent = new Intent();
            intent.setType("image/*");
            intent.setAction(Intent.ACTION_GET_CONTENT);
            SELECT_PICTURE = 1;
            SELECT_TARGET_PICTURE = 0;
            startActivityForResult(Intent.createChooser(intent,"Select Picture"), SELECT_PICTURE);
            return true;
        }

        if (id == R.id.action_selectImgToMatch) {
            Intent intent = new Intent();
            intent.setType("image/*");
            intent.setAction(Intent.ACTION_GET_CONTENT);
            SELECT_TARGET_PICTURE = 1;
            SELECT_PICTURE = 0;
            startActivityForResult(Intent.createChooser(intent,"Select Picture"), SELECT_TARGET_PICTURE);
            return true;
        }

        else if(id==R.id.action_native_stitcher)
        {
            if(sampledImage==null || imgToMatch==null)
            {
                Context context = getApplicationContext();
                CharSequence text = "You need to load an two scenes!";
                int duration = Toast.LENGTH_SHORT;

                Toast toast = Toast.makeText(context, text, duration);
                toast.show();
                return true;
            }
            Mat finalImg=new Mat();
            Stitch(imgToMatch.getNativeObjAddr(),sampledImage.getNativeObjAddr(),finalImg.getNativeObjAddr());
            displayImage(finalImg);
        }

        else if (id == R.id.action_fast || id == R.id.action_harris || id == R.id.action_orb)
        {
            if(sampledImage ==null)
            {
                Context context = getApplicationContext();
                CharSequence text = "You need to load an image first!";
                int duration = Toast.LENGTH_SHORT;

                Toast toast = Toast.makeText(context, text, duration);
                toast.show();
                return true;
            }

            if(id==R.id.action_fast)
            {
                Mat greyImage=new Mat();
                Imgproc.cvtColor(sampledImage, greyImage, Imgproc.COLOR_RGB2GRAY);

                MatOfKeyPoint keyPoints=new MatOfKeyPoint();
                FeatureDetector detector=FeatureDetector.create(FeatureDetector.FAST);

                detector.detect(greyImage, keyPoints);
                Features2d.drawKeypoints(greyImage, keyPoints, greyImage);
                displayImage(greyImage);
            }

            else if(id== R.id.action_harris)
            {
                Mat greyImage=new Mat();

                MatOfKeyPoint keyPoints=new MatOfKeyPoint();
                Imgproc.cvtColor(sampledImage, greyImage, Imgproc.COLOR_RGB2GRAY);

                SIFT detector = SIFT.create();
                //FeatureDetector detector = FeatureDetector.create(FeatureDetector.SIFT);
                detector.detect(greyImage, keyPoints);

                Features2d.drawKeypoints(greyImage, keyPoints, greyImage);

                displayImage(greyImage);

            }

            else if(id==R.id.action_orb)
            {
                Mat greyImage=new Mat();
                Imgproc.cvtColor(sampledImage, greyImage, Imgproc.COLOR_RGB2GRAY);
                MatOfKeyPoint keyPoints=new MatOfKeyPoint();

                FeatureDetector detector = FeatureDetector.create(FeatureDetector.ORB);

                detector.detect(greyImage, keyPoints);
                Features2d.drawKeypoints(greyImage, keyPoints, greyImage);
                displayImage(greyImage);

            }

        }

        else if(id==R.id.action_match)
        {
            if(sampledImage==null || imgToMatch==null)
            {
                Context context = getApplicationContext();
                CharSequence text = "You need to load an object and a scene to match!";
                int duration = Toast.LENGTH_SHORT;
                Toast toast = Toast.makeText(context, text, duration);
                toast.show();
                return true;
            }

            int maximumNuberOfMatches=50;
            Mat greyImage=new Mat();
            Mat greyImageToMatch=new Mat();

            Imgproc.cvtColor(sampledImage, greyImage, Imgproc.COLOR_RGB2GRAY);
            Imgproc.cvtColor(imgToMatch, greyImageToMatch, Imgproc.COLOR_RGB2GRAY);

            MatOfKeyPoint keyPoints=new MatOfKeyPoint();
            MatOfKeyPoint keyPointsToMatch=new MatOfKeyPoint();

            FeatureDetector detector=FeatureDetector.create(detectorID);
            detector.detect(greyImage, keyPoints);
            detector.detect(greyImageToMatch, keyPointsToMatch);


            DescriptorExtractor dExtractor = DescriptorExtractor.create(DescriptorExtractor.BRISK);
            Mat descriptors=new Mat();
            Mat descriptorsToMatch=new Mat();

            dExtractor.compute(greyImage, keyPoints, descriptors);
            dExtractor.compute(greyImageToMatch, keyPointsToMatch, descriptorsToMatch);

            DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
            MatOfDMatch matches=new MatOfDMatch();
            matcher.match(descriptorsToMatch,descriptors,matches);

            ArrayList<DMatch> goodMatches=new ArrayList<DMatch>();
            List<DMatch> allMatches=matches.toList();

            double minDist = 100;
            for( int i = 0; i < descriptorsToMatch.rows(); i++ )
            {
                double dist = (double)(allMatches.get(i)).distance;
                if( dist < minDist )
                    minDist = dist;
            }

            for( int i = 0; i < descriptorsToMatch.rows() && goodMatches.size()<maximumNuberOfMatches; i++ )
            {
                if(allMatches.get(i).distance<= 2*minDist)
                {
                    goodMatches.add(allMatches.get(i));
                }
            }

            MatOfDMatch goodEnough=new MatOfDMatch();
            goodEnough.fromList(goodMatches);
            Mat finalImg=new Mat();
            Features2d.drawMatches(greyImageToMatch, keyPointsToMatch, greyImage, keyPoints, goodEnough, finalImg, Scalar.all(-1), Scalar.all(-1),new MatOfByte(), Features2d.DRAW_RICH_KEYPOINTS + Features2d.NOT_DRAW_SINGLE_POINTS);

            displayImage(finalImg);
        }

        else if(id==R.id.action_native_match)
        {
            //FeatureDetector.AKAZE;

            if(detectorID==FeatureDetector.HARRIS)
            {
                Context context = getApplicationContext();
                CharSequence text = "Not a valid option for native matching";
                int duration = Toast.LENGTH_SHORT;

                Toast toast = Toast.makeText(context, text, duration);
                toast.show();
                return true;
            }
            if(sampledImage==null || imgToMatch==null)
            {
                Context context = getApplicationContext();
                CharSequence text = "You need to load an object and a scene to match!";
                int duration = Toast.LENGTH_SHORT;

                Toast toast = Toast.makeText(context, text, duration);
                toast.show();
                return true;
            }
            Mat finalImg=new Mat();

            FindMatches(imgToMatch.getNativeObjAddr(),sampledImage.getNativeObjAddr(),detectorID,descriptorID,finalImg.getNativeObjAddr());

            displayImage(finalImg);
        }
        }
        catch (Throwable e)
        {
            Context context = getApplicationContext();
            CharSequence text = "Sorry, the request can not be completed!";
            int duration = Toast.LENGTH_SHORT;
            Toast toast = Toast.makeText(context, text, duration);
            toast.show();
            return true;

        }

        return super.onOptionsItemSelected(item);
    }

    public boolean onCreateOptionsMenu(Menu menu) {

        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.main_menu, menu);
        return true;
    }

    public void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (resultCode == RESULT_OK) {
            if (requestCode == SELECT_PICTURE) {
                Uri selectedImageUri = data.getData();
                selectedImagePath = getPath(selectedImageUri);
                loadImage(selectedImagePath, 1);
                displayImage(sampledImage);
            }
            if (requestCode == SELECT_TARGET_PICTURE) {
                Uri selectedImageUri = data.getData();
                selectedImagePath = getPath(selectedImageUri);
                loadImage(selectedImagePath, 2);
                displayImage(imgToMatch);
            }
        }
    }

    public Mat loadImageFromFile(String filePath) {

        Mat rgbLoadedImage = null;

        //File root = Environment.getExternalStorageDirectory();
        File file = new File(filePath);

        // this should be in BGR format according to the
        // documentation.
        Mat image = Imgcodecs.imread(file.getAbsolutePath());

        if (image.width() > 0) {

            rgbLoadedImage = new Mat(image.size(), image.type());

            Imgproc.cvtColor(image, rgbLoadedImage, Imgproc.COLOR_BGR2RGB);


            image.release();
            image = null;
        }

        return rgbLoadedImage;

    }

    private void loadImage(String path, int option)
    {
//        Mat originalImage = loadImageFromFile(path);

        Mat rgbImage=loadImageFromFile(path);
        //      int chan = originalImage.channels();

        //    Imgproc.cvtColor(originalImage, rgbImage, Imgproc.COLOR_BGR2RGB);

        Display display = getWindowManager().getDefaultDisplay();
        //This is "android graphics Point" class
        Point size = new Point();
        display.getSize(size);

        int width = size.x;
        int height = size.y;

        Mat imageInp =new Mat();

        //imgToMatch = new Mat();

        double downSampleRatio= calculateSubSampleSize(rgbImage,width,height);

        Imgproc.resize(rgbImage, imageInp, new Size(),downSampleRatio,downSampleRatio,Imgproc.INTER_AREA);

        try {
            ExifInterface exif = new ExifInterface(selectedImagePath);
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, 1);

            switch (orientation)
            {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    //get the mirrored image
                    imageInp = imageInp.t();
                    //flip on the y-axis
                    Core.flip(imageInp, imageInp, 1);
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    //get up side down image
                    imageInp = imageInp.t();
                    //Flip on the x-axis
                    Core.flip(imageInp, imageInp, 0);
                    break;
            }
            if (option == 1)
                sampledImage = imageInp;
            else if(option == 2)
                imgToMatch = imageInp;

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static double calculateSubSampleSize(Mat srcImage, int reqWidth, int reqHeight) {
        // Raw height and width of image
        final int height = srcImage.height();
        final int width = srcImage.width();
        double inSampleSize = 1;

        if (height > reqHeight || width > reqWidth) {

            // Calculate ratios of requested height and width to the raw
            //height and width
            final double heightRatio = (double) reqHeight / (double) height;
            final double widthRatio = (double) reqWidth / (double) width;

            // Choose the smallest ratio as inSampleSize value, this will
            //guarantee final image with both dimensions larger than or
            //equal to the requested height and width.
            inSampleSize = heightRatio<widthRatio ? heightRatio :widthRatio;
        }
        return inSampleSize;
    }

    private String getPath(Uri uri) {
        // just some safety built in
        if(uri == null ) {
            return null;
        }
        // try to retrieve the image from the media store first
        // this will only work for images selected from gallery
        String[] projection = { MediaStore.Images.Media.DATA };
        Cursor cursor = getContentResolver().query(uri, projection, null, null, null);
        if(cursor != null ){
            int column_index = cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATA);
            cursor.moveToFirst();
            return cursor.getString(column_index);
        }

        return uri.getPath();
    }


    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    private void displayImage(Mat image)
    {
        // create a bitMap
        Bitmap bitMap = Bitmap.createBitmap(image.cols(), image.rows(),Bitmap.Config.RGB_565);
        // convert to bitmap:
        Utils.matToBitmap(image, bitMap);

        // find the imageview and draw it!
        ImageView iv = (ImageView) findViewById(R.id.ImageView);
        iv.setImageBitmap(bitMap);
    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
    }

    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public native void FindMatches(long objectAddress, long sceneAddress,int detectorID, int descriptorID,long matchingResult);
    public native void Stitch(long sceneOneAddress, long sceneTwoAddress,long stitchingResult);


}
