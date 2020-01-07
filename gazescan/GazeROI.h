#pragma once
#include "cvUtils.h"
#include <opencv2/objdetect.hpp>


/*
 * Use opencv supplied Harr cascades to detect eyes in image
 *
 */
class GazeROI 

{
	double sigmoid(const double v) const
	{
		return 1.0 / (1.0 + exp(-v));
	}

	const std::string HARR_CASCADES_DIR = "D:/opencv-4.1.0/etc/haarcascades/";
	cv::CascadeClassifier face_cascade;
	cv::CascadeClassifier  eyes_cascade;

	cv::Mat detect_frame;				//  
	cv::Rect face_rect_ = { 0, 0, 0, 0 };

	cv::Rect& face_rect(cv::Rect& new_rect) 
	{
		static double successes;

		 

		// initialized yet?
		if (face_rect_.width == 0) {
			++successes;
			face_rect_ = new_rect;
		}
		// check overlap between new face and current
		const auto overlap = sel::opencv::utils::rect_overlap(new_rect, face_rect_);
		// good overlap
		successes += overlap - 0.1;

		return sigmoid(successes) < 0.5 ? face_rect_ : new_rect;
	}

	static const int EYE_RECT_SIZE = 32;

	void crop_face_to_only_eyes(cv::Rect& face) const
	{
		face.y += face.height / 4;
		face.height /= 2;
	}
public:

	GazeROI()
	{
		face_cascade.load(HARR_CASCADES_DIR + "haarcascade_frontalface_alt.xml");
		eyes_cascade.load(HARR_CASCADES_DIR + "haarcascade_eye_tree_eyeglasses.xml");
		
	}
	virtual ~GazeROI() = default;

	/*
	 *  Given a face rectangle and a vector (length 2) of eye rectangles,  create a combined image of them suitable for input to a ML net
	 *  The output image will be 2*EYE_RECT_SIZE width, EYE_RECT_SIZE height, and will be 8-bit gray scale (CV_8UC1) depth;
	 *  No validation is done on the input rect and vector.
	 */
	bool get_last_roi(cv::Mat& frame, cv::Mat& roi) const
	{
		cvtColor(frame, detect_frame, cv::COLOR_BGR2GRAY);
		cv::Rect r = face_rect_; 
		crop_face_to_only_eyes(r);
		roi = detect_frame(r);
		return true;
	}

	bool find_roi(cv::Mat& frame, cv::Mat& roi, bool create_face_image) 
	{
		cv::Rect face;
		std::vector<cv::Rect> eyes;

		const auto detection_succeeded = detect(frame, face, eyes);

		if (!detection_succeeded)
			return false;

		if (create_face_image) {
			crop_face_to_only_eyes(face);
			roi = detect_frame(face);
			try
			{
//			cv::resize(roi, roi, cv::Size(FACE_RECT_SIZE, FACE_RECT_SIZE));
			return true;
				
			} catch (std::exception& /* ex */)
			{
				return false;
			}
		}
		// if we're here, we're trying to detect eyes

		for (auto& eye : eyes) {
			const cv::Point tl(eye.x += face.x, eye.y += face.y);
			const cv::Point br(tl.x + eye.width, tl.y + eye.width);
			//			rectangle(output_frame, tl, br, cv::Scalar(255, 0, 0), 1);
					// just care about middle 50% of detected rect
			eye.y += eye.height / 4;
			eye.height /= 2;
		}
		cv::Mat eye1;
		cv::Mat eye2;
		try {
			if (eyes[0].x < eyes[1].x)
			{
				eye1 = detect_frame(eyes[0]);
				eye2 = detect_frame(eyes[1]);
			}
			else
			{
				eye1 = detect_frame(eyes[1]);
				eye2 = detect_frame(eyes[0]);

			}

			cv::resize(eye1, eye1, cv::Size(EYE_RECT_SIZE, EYE_RECT_SIZE));
			cv::resize(eye2, eye2, cv::Size(EYE_RECT_SIZE, EYE_RECT_SIZE));


			cv::hconcat(eye1, eye2, roi);


		} catch (std::exception &ex)
		{
			std::cerr << "find_roi: " << ex.what();
			return false;
		}
		return true;
	}

	/*
	 * Given an input image,  detect face and eyes, and return them, if exactly two eyes were detected
	 * Always check that the return value === true before using the returned face rect and eye rects vector.
	 */
private:
	bool detect(cv::Mat& input_frame, cv::Rect& face, std::vector<cv::Rect>& eyes) 
	{
		cvtColor(input_frame, detect_frame, cv::COLOR_BGR2GRAY);
		std::vector<cv::Rect> faces;
		face_cascade.detectMultiScale(detect_frame, faces, 1.1, 2, cv::CASCADE_FIND_BIGGEST_OBJECT);


		cv::Rect empty_rect = { 0, 0, 0,  0 };
		face = face_rect(faces.empty() ? empty_rect : faces[0]);

		cv::Mat faceROI = detect_frame(face);
		//equalizeHist(faceROI, faceROI);

		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, cv::CASCADE_FIND_BIGGEST_OBJECT);

		return (eyes.size() == 2);
	}

};
