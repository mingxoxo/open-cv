#include <iostream>
#include <stdio.h>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>


using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

#define DEBUG 1

Mat panorama(Mat matLeftImage, Mat  matRightImage) {
	Mat matGrayLImage;
	Mat matGrayRImage;

	imshow("L", matLeftImage);
	imshow("R", matRightImage);
	waitKey(1);

	//Gray 이미지로 변환
	cvtColor(matLeftImage, matGrayLImage, CV_RGB2GRAY);
	cvtColor(matRightImage, matGrayRImage, CV_RGB2GRAY);

	//step 1 SURF이용해서 특징점 결정
	int nMinHessian = 400; // threshold (한계점)???????
	Ptr<SurfFeatureDetector> Detector = SURF::create(nMinHessian);

	vector <KeyPoint> vtKeypointsObject, vtKeypointsScene;

	Detector->detect(matGrayLImage, vtKeypointsObject);
	Detector->detect(matGrayRImage, vtKeypointsScene);

	Mat matLImageKeypoints;
	Mat matRImageKeypoints;
	drawKeypoints(matGrayLImage, vtKeypointsObject, matLImageKeypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
	drawKeypoints(matGrayRImage, vtKeypointsScene, matRImageKeypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	imshow("LK", matLImageKeypoints);
	imshow("RK", matRImageKeypoints);
	waitKey(1);

	//step 2 기술자
	Ptr<SurfDescriptorExtractor> Extractor = SURF::create();

	Mat matDescriptorsObject, matDescriptorsScene;

	Extractor->compute(matGrayLImage, vtKeypointsObject, matDescriptorsObject);
	Extractor->compute(matGrayRImage, vtKeypointsScene, matDescriptorsScene);
	//descriptor(기술자)들 사이의 매칭 결과를 matches에 저장한다.
	FlannBasedMatcher Matcher; //kd트리를 사용하여 매칭을 빠르게 수행
	vector <DMatch> matches;
	Matcher.match(matDescriptorsObject, matDescriptorsScene, matches);

	Mat matGoodMatches1;
	drawMatches(matGrayLImage, vtKeypointsObject, matGrayRImage, vtKeypointsScene, matches, matGoodMatches1, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("allmatches", matGoodMatches1);
	waitKey(1);
	double dMaxDist = matches[0].distance;
	double dMinDist = matches[0].distance;
	double dDistance;

	// 두 개의 keypoint 사이에서 min-max를 계산한다
	for (int i = 0; i < matDescriptorsObject.rows; i++) {
		dDistance = matches[i].distance;

		if (dDistance < dMinDist) dMinDist = dDistance;
		if (dDistance < dMaxDist) dMaxDist = dDistance;
	}
	printf("max_dist : %f \n", dMaxDist);
	printf("max_dist : %f \n", dMinDist);
	// 값이 작을수록 matching이 잘 된 것
	//min의 값의 3.5배까지만 goodmatch로 인정해주겠다

	vector<DMatch>good_matches;

	for (int i = 0; i < matDescriptorsObject.rows; i++) {
		if (matches[i].distance < 3.5 * dMinDist)
			good_matches.push_back(matches[i]);
	}

	//keypoint들과 matching 결과 ("good" matched point)를 선으로 연결하여 이미지에 그려 표시
	Mat matGoodMatches;
	drawMatches(matGrayLImage, vtKeypointsObject, matGrayRImage, vtKeypointsScene, good_matches, matGoodMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	imshow("good-matches", matGoodMatches);
	waitKey(3000);
	//Point2f형으로 변환
	vector <Point2f> obj;
	vector <Point2f> scene;

	for (int i = 0; i < good_matches.size();i++) {
		obj.push_back(vtKeypointsObject[good_matches[i].queryIdx].pt);
		scene.push_back(vtKeypointsScene[good_matches[i].trainIdx].pt);
	}
	Mat HomoMatrix = findHomography(scene, obj, CV_RANSAC);
	//RANSAC기법을 이용하여 첫 번째 매개변수와 두번째 매개변수 사이의 3*3 크기의 투영행렬변환 H를 구한다
	cout << HomoMatrix << endl;

	//Homograpy matrix를 사용하여 이미지를 삐뚤게
	Mat matResult;

	warpPerspective(matRightImage, matResult, HomoMatrix, Size(matRightImage.cols * 2, matRightImage.rows), INTER_CUBIC);

	Mat matPanorama;
	matPanorama = matResult.clone();
	//matResult = imagecut(matGrayLImage, matResult,  HomoMatrix);
	imshow("wrap", matResult);
	waitKey(3000);

	Mat matROI(matPanorama, Rect(0, 0, matLeftImage.cols, matLeftImage.rows));
	matLeftImage.copyTo(matROI);
	
	//imshow("Panorama", matPanorama);
	//waitKey(3000);
	return matResult;
}

/*Mat imagecut(Mat img1, Mat img2, Mat HomoMatrix) {

std::vector<Point2f> model_corner(4);
model_corner[0] = cvPoint(0, 0);
model_corner[1] = cvPoint(img1.cols, 0);
model_corner[2] = cvPoint(img1.cols, img1.rows);
model_corner[3] = cvPoint(0, img1.rows);

std::vector<Point2f> scene_corner(4);
perspectiveTransform(model_corner, scene_corner, HomoMatrix);
Mat img3 = img2(Range(scene_corner[0].y, scene_corner[1].y), Range(scene_corner[2].x, scene_corner.size[1].x));
return img3;
}*/

int main()
{
	Mat img1;
	Mat img2;
	Mat img3;
	Mat panoram;

	img1 = imread("C:/opencv-2-4-13-6/S1.jpg", IMREAD_COLOR);
	img2 = imread("C:/opencv-2-4-13-6/S2.jpg", IMREAD_COLOR);
	img3 = imread("C:/opencv-2-4-13-6/S3.jpg", IMREAD_COLOR);

	if (img1.empty() || img2.empty() || img3.empty()) return -1;

	panoram = panorama(img1, img2);
	Mat img_gray;
	cvtColor(panoram, img_gray, COLOR_BGR2GRAY);
	threshold(img_gray, img_gray, 25, 255, THRESH_BINARY); //Threshold the gray
	vector<vector<Point> > contours; // Vector for storing contour
	vector<Vec4i> hierarchy;
	findContours(img_gray, contours, hierarchy, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image
	int largest_area = 0;
	int largest_contour_index = 0;
	Rect bounding_rect;

	for (int i = 0; i< contours.size(); i++) // iterate through each contour. 
	{
		double a = contourArea(contours[i], false);  //  Find the area of contour
		if (a>largest_area) {
			largest_area = a;
			largest_contour_index = i;                //Store the index of largest contour
			bounding_rect = boundingRect(contours[i]); // Find the bounding rectangle for biggest contour
		}

	}

	// Scalar color( 255,255,255);
	panoram = panoram(Rect(bounding_rect.x, bounding_rect.y, bounding_rect.width, bounding_rect.height));
	imshow("Panorama", panoram);
	waitKey(3000);

	Mat img_gray2;
	Mat panoram2;
	panoram2 = panorama(img2, img3);
	cvtColor(panoram2, img_gray2, COLOR_BGR2GRAY);

	//Finding the largest contour i.e remove the black region from image
	threshold(img_gray2, img_gray2, 25, 255, THRESH_BINARY); //Threshold the gray
	vector<vector<Point> > contours2; // Vector for storing contour
	vector<Vec4i> hierarchy2;
	findContours(img_gray2, contours2, hierarchy2, CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE); // Find the contours in the image
	int largest_area2 = 0;
	int largest_contour_index2 = 0;
	Rect bounding_rect2;

	for (int i = 0; i< contours2.size(); i++) // iterate through each contour. 
	{
		double a = contourArea(contours2[i], false);  //  Find the area of contour
		if (a>largest_area2) {
			largest_area2 = a;
			largest_contour_index2 = i;                //Store the index of largest contour
			bounding_rect2 = boundingRect(contours2[i]); // Find the bounding rectangle for biggest contour
		}

	}
	Mat pa;
	panoram2 = panoram2(Rect(bounding_rect2.x, bounding_rect2.y, bounding_rect2.width, bounding_rect2.height));
	imshow("Panorama2", panoram2);
	waitKey(3000);
	pa = panorama(panoram, panoram2);
	imshow("최종 파노라마", panoram);
	waitKey();
	return 0;
}
