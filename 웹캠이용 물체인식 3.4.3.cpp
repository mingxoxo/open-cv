//잘 실행 되다가 실행 도중 오류 발생 -> inlier의 부족으로 생긴 문제 (해결하지 )

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <stdio.h>
#include <opencv2/videoio.hpp>

//특징 추출과 매칭에 필요한 헤더파일
using namespace cv;
using namespace cv::xfeatures2d;
#define RED Scalar(0,0,255)
#define COLOR Scalar(0,0,0)

int main()
{
	Mat img1 = imread("C:/opencv-2-4-13-6/image/opencv.png", IMREAD_COLOR);
	Mat matGrayImage;
	assert(img1.data);
	cvtColor(img1, matGrayImage, CV_RGB2GRAY);
	Ptr<SiftFeatureDetector> detector = SIFT::create(800); //SIFT를 검출
	//Sift 클래스 안에 create 함수
	//create안의 숫자는 스케일을 의미
	std::vector<KeyPoint> keypoint1, vtKeypointVideo; //vector은 데이터 집합을 저장하는데 사용
	detector->detect(matGrayImage, keypoint1);//멤버 함수 detect()는 첫 번째 매개변수에 있는 영상에서 키포인트를 검출하여 
	//두번째 매개변수인 vector형 변수에 저장해준다.
	Mat disp;
	drawKeypoints(img1, keypoint1, disp, RED, DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//drawKeypoints()함수는 첫번째 매개변수에 있는 Mat 영상에
	//두 번째 매개변수의 키포인트를 그려 세번째 매개변수의 영상에 저장해준다
	namedWindow("키포인트"); imshow("키포인트", disp);
	waitKey(1);

	//기술자 추출
	Ptr<SiftDescriptorExtractor> extractor = SIFT::create();
	Mat descriptor1;
	extractor->compute(img1, keypoint1, descriptor1);
	//compute() 첫번째 매개변수의 영상에 대해 두번째 매개변수의 키포인트 위치에서 기술자를 추출 후
	//세번째 매개변수에 저장해준다
	
	VideoCapture cap(0);

	UMat video_frame;
	namedWindow("camera", 1);
	for (;;) {
		
		Mat matGrayvideo;
		Mat matDescriptorsVideo;
		cap.read(video_frame); // retrieve
		imshow("camera", video_frame);
		cvtColor(video_frame, matGrayvideo, CV_RGB2GRAY);

		detector->detect(video_frame, vtKeypointVideo);
		drawKeypoints(video_frame, vtKeypointVideo, disp, RED, DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
		// DrawMatchesFlags::DRAW_RICH_KEYPOINTS : 각 키포인트에 대해 키포인트 크기 및 방향이있는 원 중심의 키포인트가 그려집니다.
		//namedWindow("키포인트"); imshow("키포인트", disp);
		//waitKey(1);

		extractor->compute(video_frame, vtKeypointVideo, matDescriptorsVideo);

		FlannBasedMatcher matcher; // kd트리를 사용하여 매칭을 수행
		std::vector<DMatch> match; //매칭결과를 저장할 vector형 변수
		//Dmatch는 매칭 목록을 표현하는 클래스로 정렬목적으로 사용된다.
		matcher.match(descriptor1, matDescriptorsVideo,  match);
		//기술자 매칭한 것을 match에 저장
		//goodmatch를 선별하기 위한 과정

		double maxd = 0; double mind = match[0].distance; //매칭 점수의 최소와 최대를 구해줌
		for (int i = 0; i < descriptor1.rows; i++) {
			double dist = match[i].distance;
			if (dist < mind) mind = dist;
			if (dist > maxd) maxd = dist;
		}
		printf("%lf %lf\n", mind, maxd);

		std::vector<DMatch> good_match;
		for (int i = 0; i < descriptor1.rows; i++)
			if (match[i].distance <= max(2 * mind, 0.02)) good_match.push_back(match[i]);
		//매칭점수가 최소값의 두배(만약 이 값이 0.02보다 작으면 0.02) 이내의 매칭쌍을 골라낸다.

		Mat img_match;
		drawMatches(img1, keypoint1, video_frame, vtKeypointVideo, good_match, img_match, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
		//마스크 인수가 empty이므로(std::vector< char >()) 이미지 전 영역에서 매칭을 그린다
		//namedWindow("매칭 결과"); imshow("매칭 결과", img_match);
		//waitKey(1);
		for (int i = 0; i < (int)good_match.size(); i++)
			printf("키포인트 %d~%d\n", good_match[i].queryIdx, good_match[i].trainIdx);
		//매칭쌍의 첨자를 출력

		//findHomography를 사용하기 위해 Point2f형으로 변환해준다
		std::vector<Point2f> model_pt;
		std::vector<Point2f> scene_pt;
		for (int i = 0; i < good_match.size(); i++) {
			model_pt.push_back(keypoint1[good_match[i].queryIdx].pt);
			scene_pt.push_back(vtKeypointVideo[good_match[i].trainIdx].pt);
		}
		Mat H = findHomography(model_pt, scene_pt, CV_RANSAC);

		std::vector<Point2f> model_corner(4);
		model_corner[0] = cvPoint(0, 0);
		model_corner[1] = cvPoint(img1.cols, 0);
		model_corner[2] = cvPoint(img1.cols, img1.rows);
		model_corner[3] = cvPoint(0, img1.rows);

		std::vector<Point2f> scene_corner(4);
		perspectiveTransform(model_corner, scene_corner, H);
		//좌표에 투영변환 H를 적용하여 장면 영상에 나타날 좌표 scene_corner를 계산한다

		Point2f p(img1.cols, 0); //4개의 선분영상 그려넣음으로써 인식 결과 표시
		//하나의 윈도우에 모델 영상과 장면 영상을 같이 그려넣었기때문에 모델 영상의 너비만큼 이동시켜 그리기 위해 
		//Point2f 형의 p를 더해주었다.
		line(img_match, scene_corner[0] + p, scene_corner[1] + p, RED, 3);
		line(img_match, scene_corner[1] + p, scene_corner[2] + p, RED, 3);
		line(img_match, scene_corner[2] + p, scene_corner[3] + p, RED, 3);
		line(img_match, scene_corner[3] + p, scene_corner[0] + p, RED, 3);

		namedWindow("인식 결과", WINDOW_NORMAL); imshow("인식 결과", img_match);

		
		if (waitKey(30) >= 0) break;
	}
	return 0;
}
