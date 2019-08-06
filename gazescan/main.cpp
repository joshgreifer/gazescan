#include <opencv2/core.hpp>
#include "opencv2/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/dnn/dnn.hpp>


#include <iostream>
#include <sys/stat.h>
#include <fstream>
#include <algorithm>
#include <boost/filesystem.hpp>

#include "Eyedetector.h"
#include "cvUtils.h"
#include "GazeDetector.h"

//using namespace cv;
using namespace std;

const string RESOURCE_DIR = "D:/projects/gazescan/";

const string OUTPUT_DIR = "D:/projects/gazescan/output/";
const string MAIN_WINDOW_NAME = "GazeScan";

//const string LABELS_FILENAME = OUTPUT_DIR + "Labels.txt";
//const string FEATURES_FILENAME = OUTPUT_DIR + "Features.png";
//

//const int FEATURE_VECTOR_SIZE = EyeDetector::EYE_RECT_SIZE * EyeDetector::EYE_RECT_SIZE * 2;



EyeDetector eye_detector;

GazeDetector gaze_detector(RESOURCE_DIR + "gazescan.onnx");


cv::Mat feature_mat;		// each row is a feature vector, created by reshaping the found eyes, of length FEATURE_VECTOR_SIZE
vector<int> labels_vec;	// each element is a number (1-9) representing the quadrant the eyes were looking at:
// +---+---+---+
// | 0 | 3 | 6 |
// +---+---+---+
// | 1 | 4 | 7 |
// +---+---+---+
// | 2 | 5 | 8 |
// +---+---+---+


int cursor_pos_x;
int cursor_pos_y;

bool is_recording;
bool is_predicting;
unsigned int num_images_saved;



const int GRID_NUM = 3;  // we use 3 X 3
const int NUM_CATEGORIES = GRID_NUM * GRID_NUM;

// Set in setup_and_display_main_window
int SCREEN_W;
int SCREEN_H;

const int RECORDING_BATCH_SIZE = 100; // 

enum Category
{
	top_left = 0,
	left = 1,
	bottom_left = 2,
	top = 3,
	centre = 4,
	bottom = 5,
	top_right = 6,
	right = 7,
	bottom_right = 8,

};

int current_category = centre;

std::vector<cv::Mat>calibration_images;

auto coord2label(const int x, const int y)
{
	return GRID_NUM * y / SCREEN_H + GRID_NUM * (GRID_NUM * x / SCREEN_W);

}
string label_name(const int label)
{
	static const char *label_names[] = { "c0tl", "c1ml", "c2bl", "c3tm", "c4mm", "c5bm", "c6tr", "c7mr", "c8br" };
	return label_names[label];

}

cv::Rect label2rect(int label)
{

	int col = label / GRID_NUM;
	int row = label % GRID_NUM;

	return cv::Rect(col * SCREEN_W / GRID_NUM, row * SCREEN_H / GRID_NUM, SCREEN_W / GRID_NUM, SCREEN_H / GRID_NUM);
}

void detect_and_display(cv::Mat input_frame)
{
	static int next_category = NUM_CATEGORIES;

	cv::Mat output_frame(SCREEN_H,SCREEN_W,CV_8UC3);

	output_frame.setTo(cv::Scalar(255, 255, 255));
	cv::Rect face_rect;
	std::vector<cv::Rect> eye_rects;

	if (eye_detector.detect(input_frame, face_rect, eye_rects)) {
		cv::Mat eyes_combined;

	
		if (eye_detector.create_eyes_image(face_rect, eye_rects, eyes_combined)) {
	// show region we're supposed to be looking at
			static bool calibrating;
			if (calibration_images.empty()) {
				cout << "Calibration start\n";
				calibrating = true;
				current_category = centre;
				if (++next_category >= NUM_CATEGORIES)
					next_category = 0;

			} else if (calibration_images.size() >= RECORDING_BATCH_SIZE)  {
				cout << RECORDING_BATCH_SIZE << " calibration images saved. Calibration ended\n";
				calibrating = false;
				current_category = next_category;
				cout << "Category set to " << current_category << "\n";
			}
							
			
			auto focus_rect = label2rect(current_category);
			rectangle(output_frame, focus_rect, cv::Scalar(0, 0, calibrating ? 128 : 255), 3);

			if (calibrating)
					calibration_images.push_back(eyes_combined);
			
			else  {
				cv::vconcat(eyes_combined, calibration_images.back(), eyes_combined);

				if (is_predicting) {

					auto results = gaze_detector.detect(eyes_combined);

					auto label = 0;

					const auto num_labels = results.size().width;
					for (auto i = 0; i < num_labels; ++i)
					{
						const auto conf = results.at<float>(i);
						auto label = num_labels - i - 1;
						auto predicted_rect = label2rect(label);
						rectangle(output_frame, predicted_rect, cv::Scalar(255-255*conf, 255, 255), cv::FILLED);
						char buf[100];
						sprintf_s(buf, sizeof(buf), "%2.2f", conf * 100);
						cv::putText(output_frame, buf,
							{ predicted_rect.x + predicted_rect.width /2, predicted_rect.y + predicted_rect.height / 2 }, cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(0, 0, 0));
					}

				}
				if (is_recording) {
					calibration_images.pop_back();

					char img_file_name[255];
					auto img_dir(OUTPUT_DIR + "img/" + label_name(current_category) + "/");
					boost::filesystem::create_directories(img_dir);
					sprintf_s(img_file_name, sizeof(img_file_name), "Img%6.6u.png", ++num_images_saved);

					cv::imwrite(img_dir + img_file_name, eyes_combined);
					cv::putText(output_frame, img_file_name,
						{ 30, SCREEN_H - 30 }, cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0));
					
				}
			
			}
			cvtColor(eyes_combined, eyes_combined, cv::COLOR_GRAY2BGR);
			// Show region cursor is in
			// rectangle(output_frame, label2rect(coord2label(cursor_pos_x, cursor_pos_y)), cv::Scalar(0, 0, 255), 1);

			// display eyes at cursor pos
			try {
				cv::Mat eyesOverlayROI = output_frame({ focus_rect.x + (focus_rect.width - 2 * EyeDetector::EYE_RECT_SIZE)/2,
					focus_rect.y + (focus_rect.height - 2 * EyeDetector::EYE_RECT_SIZE)/2, 2 * EyeDetector::EYE_RECT_SIZE, 2 * EyeDetector::EYE_RECT_SIZE });
				eyes_combined.copyTo(eyesOverlayROI);
	
				//cv::Mat inputFrameROI = output_frame({  (SCREEN_W - 640)/2, (SCREEN_H - 480)/2, 640, 480 });
				//input_frame.copyTo(inputFrameROI);


			} catch (...) {}
		}


	}


	imshow(MAIN_WINDOW_NAME, output_frame);
}


void onMouse(int event, int x, int y, int flags, void *userdata)
{
	cursor_pos_x = x;
	cursor_pos_y = y;
}

void setup_and_display_main_window()
{
	cv::namedWindow(MAIN_WINDOW_NAME, cv::WINDOW_NORMAL );
	cv::setWindowProperty(MAIN_WINDOW_NAME, cv::WND_PROP_FULLSCREEN , cv::WINDOW_FULLSCREEN);
	auto r = cv::getWindowImageRect(MAIN_WINDOW_NAME);
	SCREEN_H = r.height;
	SCREEN_W = r.width;

	cv::setMouseCallback(MAIN_WINDOW_NAME, onMouse);

}

bool file_exists(const char *filename)
{
	struct stat buffer;
	return stat(filename, &buffer) == 0;
}
int main(int argc, char** argv)
{

	try {
		// If filename passed, just run predictor on image file
		if (argc > 1) {
			boost::filesystem::path p(argv[1]);
			if (exists(p)) {
				if (is_regular_file(p)) {
					auto path = p.string();
					auto eyes_combined = cv::imread(path, cv::IMREAD_GRAYSCALE);
					if (eyes_combined.empty())
						return 1;
					float confidence;
					auto result = gaze_detector.predict(eyes_combined, confidence);
					cout << path << '\t' << result << '\t' << confidence << '\n';


				} else if (is_directory(p)) {

					for (auto& x : boost::filesystem::directory_iterator(p)) {
						auto path = x.path().string();
						auto eyes_combined = cv::imread(path, cv::IMREAD_GRAYSCALE);
						if (eyes_combined.empty())
							continue;
					float confidence;
					auto result = gaze_detector.predict(eyes_combined, confidence);
					cout << path << '\t' << result << '\t' << confidence << '\n';

					}
				} else
					cout << p << " exists, but is not a regular file or directory\n";
			} else {
				cout << p << " is not a filename or a directory\n";
				return 1;
			}

			return 0;
		}

		boost::filesystem::create_directories(OUTPUT_DIR);
		
		for (int label = 0; label < NUM_CATEGORIES; ++label)
		{
			boost::filesystem::path p(OUTPUT_DIR + "img/" + label_name(label));
			for (auto& x : boost::filesystem::directory_iterator(p)) {
					auto path = x.path().string();
					const char *number_part_of_filename = path.c_str()+path.length()-10;
						// parse out the number 
						auto img_file_num = static_cast<unsigned>(atol(number_part_of_filename));
						if (num_images_saved < img_file_num)
							num_images_saved = img_file_num;

			}
			
		}

		setup_and_display_main_window();

		cv::VideoCapture vid_cap;
		vid_cap.open(0);
		if (!vid_cap.isOpened()) {
			cerr << "Couldn't open video capture device.";
		}

		cv::Mat frame;
		int pressed_key = 0;
		while (vid_cap.read(frame) && !frame.empty() && pressed_key != 27) {
			detect_and_display(frame);
			pressed_key = cv::waitKey(1);
			switch (pressed_key) {
			case 32: // space bar
				is_recording = !is_recording;
				break;
			case 13:
				is_predicting = !is_predicting;

			default:
				;
			}

		}

	} catch (std::exception &ex) {
		std::cerr << "main: " << ex.what();
	}

}
