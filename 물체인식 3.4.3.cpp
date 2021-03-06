#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>

//특징 추출과 매칭에 필요한 헤더파일
using namespace cv;
using namespace cv::xfeatures2d;
#define RED Scalar(0,0,255)

int main()
{
	Mat img1 = imread("C:/opencv-2-4-13-6/image/model3.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread("C:/opencv-2-4-13-6/image/scene.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	// CV_LOAD_IMAGE_GRAYSCALE 영상을 읽어들일 때 자동으로 명암 영상으로 변환
	assert(img1.data && img2.data);
	//인자로 들어가는 조건이 참이면 그냥 지나가고 아니면 에러

	Ptr<SiftFeatureDetector> detector = SIFT::create(300); //SIFT를 검출
	//Sift 클래스 안에 create 함수
	//create안의 숫자는 스케일 의미
	std::vector<KeyPoint> keypoint1, keypoint2;
	//vector은 데이터 집합을 저장하는데 사용

	detector->detect(img1, keypoint1);
	detector->detect(img2, keypoint2);
	//멤버 함수 detect()는 첫 번째 매개변수에 있는 영상에서 키포인트를 검출하여 
	//두번째 매개변수인 vector형 변수에 저장해준다.
	

	Mat disp;
	drawKeypoints(img1, keypoint1, disp, RED, DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	//drawKeypoints()함수는 첫번째 매개변수에 있는 Mat 영상에
	//두 번째 매개변수의 키포인트를 그려 세번째 매개변수의 영상에 저장해준다
	namedWindow("키포인트"); imshow("키포인트", disp);
	waitKey(1);

	//기술자 추출
	Ptr<SiftDescriptorExtractor> extractor = SIFT::create();
	Mat descriptor1, descriptor2;
	extractor->compute(img1, keypoint1, descriptor1);
	extractor->compute(img2, keypoint2, descriptor2);
	//compute() 첫번째 매개변수의 영상에 대해 두번째 매개변수의 키포인트 위치에서 기술자를 추출 후
	//세번째 매개변수에 저장해준다.
	FlannBasedMatcher matcher; // kd트리를 사용하여 매칭을 수행
	std::vector<DMatch> match; //매칭결과를 저장할 vector형 변수
							   //Dmatch는 매칭 목록을 표현하는 클래스로 정렬목적으로 사용된다.
	matcher.match(descriptor1, descriptor2, match);
	//기술자 매칭한 것을 match에 저장

	//goodmatch를 선별하기 위한 과정
	double maxd = 0; double mind = 100; //매칭 점수의 최소와 최대를 구해줌
	for (int i = 0; i < descriptor1.rows; i++) {
		double dist = match[i].distance;
		if (dist < mind) mind = dist;
		if (dist > maxd) maxd = dist;
	}

	std::vector<DMatch> good_match;
	for (int i = 0; i < descriptor1.rows; i++) 
		if (match[i].distance <= max(2 * mind, 0.02)) good_match.push_back(match[i]);
	//매칭점수가 최소값의 두배(만약 이 값이 0.02보다 작으면 0.02) 이내의 매칭쌍을 골라낸다.

	Mat img_match;
	drawMatches(img1, keypoint1, img2, keypoint2, good_match, img_match, Scalar::all(-1), Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
	//마스크 인수가 empty이므로(std::vector< char >()) 이미지 전 영역에서 매칭을 그린다
	namedWindow("매칭 결과"); imshow("매칭 결과", img_match);
	waitKey(1);
	for (int i = 0; i < (int)good_match.size(); i++)
		printf("키포인트 %d~%d\n", good_match[i].queryIdx, good_match[i].trainIdx);
	//매칭쌍의 첨자를 출력

	//findHomography를 사용하기 위해 Point2f형으로 변환해준다
	std::vector<Point2f> model_pt;
	std::vector<Point2f> scene_pt;
	for (int i = 0; i < good_match.size(); i++) {
		model_pt.push_back(keypoint1[good_match[i].queryIdx].pt);
		scene_pt.push_back(keypoint2[good_match[i].trainIdx].pt);
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
	waitKey();
	return 0;
}
