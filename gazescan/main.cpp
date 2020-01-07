

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaobjdetect.hpp>

#include <iostream>
#include <sys/stat.h>
#include <fstream>
#include <algorithm>
#include <boost/filesystem.hpp>

#include "GazeROI.h"
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



GazeROI eye_detector;

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

bool is_acquiring_training_data;
bool is_predicting;
bool use_face_image = true;
bool add_calibration_images_to_training_images = false;

unsigned int num_images_saved;



const int GRID_NUM = 3;  // we use 3 X 3
const int NUM_CATEGORIES = GRID_NUM * GRID_NUM;

// Set in setup_and_display_main_window
int SCREEN_W;
int SCREEN_H;

const int RECORDING_BATCH_SIZE = 50;

// only start recording after INITIAL_IMAGES_TO_IGNORE images obtained, to give user time to move gaze
const int INITIAL_IMAGES_TO_IGNORE = 50;

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

// These are up to RECORDING_BATCH_SIZE images of eyes looking at the centre of the screen
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
	static bool calibrating;
	static int num_times_gazed_at_current_category = -INITIAL_IMAGES_TO_IGNORE;

	cv::Mat output_frame(SCREEN_H,SCREEN_W,CV_8UC3);

	output_frame.setTo(cv::Scalar(255, 255, 255));

	
	cv::Mat roi;
	// set the face rect from detector if we're gazing at centre
	if (current_category == centre && num_times_gazed_at_current_category < 0)
		eye_detector.find_roi(input_frame,  roi, use_face_image);
	else
		eye_detector.get_last_roi(input_frame, roi);

	if (roi.size().width) {

		gaze_detector.accumulate_image_for_average(roi);

		if (add_calibration_images_to_training_images) {
			
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
		} else {
			if (++num_times_gazed_at_current_category >= RECORDING_BATCH_SIZE)
			{
				num_times_gazed_at_current_category = -INITIAL_IMAGES_TO_IGNORE;
				if (++current_category >= NUM_CATEGORIES)
					current_category = 0;
			}
		}
	// show region we're supposed to be looking at

		const auto focus_rect = label2rect(current_category);
		rectangle(output_frame, focus_rect, cv::Scalar(num_times_gazed_at_current_category < 0 ? 255 : 0, 0, calibrating ? 128 : 255), 3);

		if (calibrating)
				calibration_images.push_back(roi);
		
		else  {
			if (add_calibration_images_to_training_images) {
				cv::vconcat(roi, calibration_images.back(), roi);
				calibration_images.pop_back();
			}
			if (is_predicting) {

				auto results = gaze_detector.detect(roi);

				auto label = 0;

				const auto num_labels = results.size().width;
				for (auto i = 0; i < num_labels; ++i)
				{
					const auto conf = results.at<float>(i);
					auto label = num_labels - i - 1;
					const auto predicted_rect = label2rect(label);
					rectangle(output_frame, predicted_rect, cv::Scalar(255.0 - 255.0 * conf, 255.0, 255.0), cv::FILLED);
					char buf[100];
					sprintf_s(buf, sizeof(buf), "%2.2f", 100.0f * conf);
					cv::putText(output_frame, buf,
						{ predicted_rect.x + predicted_rect.width /2, predicted_rect.y + predicted_rect.height / 2 }, cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(0, 0, 0));
				}

			}
			if (is_acquiring_training_data) {
				// only start recording after INITIAL_IMAGES_TO_IGNORE images obtained, to give user time to move gaze
				if (num_times_gazed_at_current_category >= 0) {
					char img_file_name[255];
					auto img_dir(OUTPUT_DIR + "img/" + label_name(current_category) + "/");
					boost::filesystem::create_directories(img_dir);
					sprintf_s(img_file_name, sizeof(img_file_name), "Img%6.6u.png", ++num_images_saved);

					cv::resize(roi, roi, GazeDetector::ImageSize());
					cv::imwrite(img_dir + img_file_name, roi);
					cv::putText(output_frame, img_file_name,
						{ focus_rect.x + 30, focus_rect.y  + 30 }, cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0, 0, 0));
						
				}
				
			}
		
		}
		cvtColor(roi, roi, cv::COLOR_GRAY2BGR);
		// Show region cursor is in
		// rectangle(output_frame, label2rect(coord2label(cursor_pos_x, cursor_pos_y)), cv::Scalar(0, 0, 255), 1);

		// display roi at cursor pos
		try {

			roi.copyTo(output_frame(
				{ focus_rect.x + (focus_rect.width - roi.size().width)/2,
				focus_rect.y + (focus_rect.height - roi.size().height)/2, 
					roi.size().width, 
					roi.size().height 
				}));
			//cv::Mat test; gaze_detector.average_img(test);
			//// std::cerr << "average: "  << sel::opencv::utils::GetMatType(test) << std::endl;

			//cvtColor(test, test, cv::COLOR_GRAY2BGR);

			//test.copyTo(output_frame(
			//	{ focus_rect.x + (focus_rect.width - test.size().width)/2,
			//	focus_rect.y + (focus_rect.height - test.size().height)/2, 
			//		test.size().width, 
			//		test.size().height 
			//	}));




		} catch (...) {}
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

// run prediction on file or set of files
int predict(const char *file_or_directory_path)
{
	const boost::filesystem::path p(file_or_directory_path);
	if (exists(p)) {
		if (is_regular_file(p)) {
			const auto path = p.string();
			auto detection_image = cv::imread(path, cv::IMREAD_GRAYSCALE);
			if (detection_image.empty())
				return 1;
//			gaze_detector.accumulate_image_for_average(detection_image);
//			float confidence;
//			const auto result = gaze_detector.predict(detection_image, confidence);
			const auto result = gaze_detector.detect(detection_image);
			cout << path << '\t' << result << '\n';


		} else if (is_directory(p)) {

			for (auto& x : boost::filesystem::directory_iterator(p)) {
				auto path = x.path().string();
				auto detection_image = cv::imread(path, cv::IMREAD_GRAYSCALE);
				if (detection_image.empty())
					continue;
//			gaze_detector.accumulate_image_for_average(detection_image);
			float confidence;
			const auto result = gaze_detector.predict(detection_image, confidence);
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

unsigned int get_training_set_size()
{
	unsigned int training_set_size = 0;
	for (int label = 0; label < NUM_CATEGORIES; ++label)
	{
		boost::filesystem::path p(OUTPUT_DIR + "img/" + label_name(label));
		if (exists(p))
			for (auto& x : boost::filesystem::directory_iterator(p)) {
					auto path = x.path().string();
					// ReSharper disable once CommentTypo
					auto *number_part_of_filename = path.c_str()+path.length()-10; // last 10 chars are 'DDDDDD.png' , where D is a digit
					// parse out the number 
					char *end_ptr;
					const auto img_file_num = static_cast<unsigned>(strtol(number_part_of_filename, &end_ptr, 10));
					if (training_set_size < img_file_num)
						training_set_size = img_file_num;

			}
		
	}
	return training_set_size;
	
}

int main(int argc, char** argv)
{
//	GazeDetector::unit_test();

	try {
		// If filename passed, just run predictor on image file
		if (argc > 1) {
			return predict(argv[1]);
		}

		boost::filesystem::create_directories(OUTPUT_DIR);

		num_images_saved = get_training_set_size();
	
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
				is_acquiring_training_data = !is_acquiring_training_data;
				break;
			case 13:
				is_predicting = !is_predicting;

			default:
				;
			}

		}

	} catch (std::exception &ex) {
		std::cerr << "main: " << ex.what();
	} catch (...) {
		std::cerr << "main: Unknown error";		
	}

}

