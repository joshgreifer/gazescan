#pragma once
#include <opencv2/dnn/dnn.hpp>

/*
 * Detect the gaze direction given an input image, 32 X 16 of two eyes, which was used to train a DNN
 *
 */

class GazeDetector
{
	// See gazescan.m 
//	Load the CNN we trained and exported in Matlab
	cv::dnn::Net TrainedNet;

	// Aggregated average image
	cv::Mat accumulated_sum_image_;
	size_t num_images_averaged_ = 0;

public:

	static cv::Size ImageSize()
	{
		return { 96, 48 };
	}

	auto& net() const
	{
		return TrainedNet;
	}

	explicit GazeDetector(const std::string& onnx_filename)
	{
		TrainedNet = cv::dnn::readNetFromONNX(onnx_filename);
		std::cout <<"[\n";
		for (auto& layer_name : TrainedNet.getLayerNames())
			std::cout << '\t' << layer_name << '\n';
		std::cout <<"]\n";

	}

	auto predict(cv::Mat& detection_image, float& confidence)
	{

		auto results = detect(detection_image);

		int label = 0;

		confidence = 0.0f;
		const auto num_labels = results.size().width;
		for (auto i = 0; i < num_labels; ++i)
		{
			const auto conf = results.at<float>(i);
			if (conf > confidence) {
				label = i;
				confidence = conf;
			}
		}

		return num_labels - label - 1;
	}

	cv::Mat average() const
	{
		cv::Mat average_image;
		if (num_images_averaged_ > 1)
			average_image = accumulated_sum_image_ / static_cast<double>(num_images_averaged_);

		return average_image;
	}
	cv::Mat average_img() const
	{
		cv::Mat average_image = accumulated_sum_image_ / static_cast<double>(num_images_averaged_);
		average_image.convertTo(average_image, CV_8UC1, 256.0);

		return average_image;
	}

	void accumulate_image_for_average(cv::Mat& new_image)
	{
		cv::Mat im;
		new_image.convertTo(im, CV_64F, 1/256.0);
		cv::resize(im, im, ImageSize());

		if (num_images_averaged_++ == 0) 
			accumulated_sum_image_ = im;
		else 
			accumulated_sum_image_ = accumulated_sum_image_ + im;
	}

	cv::Mat detect(cv::Mat detection_image)
	{

		cv::resize(detection_image, detection_image, ImageSize());
		detection_image.convertTo(detection_image, CV_64F, 1/256.0);
		detection_image = detection_image - average();
		const auto blob = cv::dnn::blobFromImage(detection_image);
//		std::cerr << "detection_image: "  << sel::opencv::utils::GetMatType(detection_image) << std::endl;
		TrainedNet.setInput(blob);

		return TrainedNet.forward("softmax");;

	}
	
	static void unit_test()
	{
		GazeDetector g("D:/projects/gazescan/gazescan.onnx");
		cv::Mat R1 = cv::Mat(3, 3, CV_64F);
		cv::Mat R2 = cv::Mat(3, 3, CV_8UC1);
		cv::Mat R3;
		cv::randu(R1, 0, 1.0);
		cv::randu(R2, 0, 255);
		std::cout << "R2 (CV_8UC1) = " << std::endl <<        R2           << std::endl << std::endl;
		R2.convertTo(R2, CV_64F, 1/256.0);
		std::cout << "R1 = " << std::endl <<        R1           << std::endl << std::endl;
		std::cout << "R2 (CV_64F) = " << std::endl <<        R2           << std::endl << std::endl;
		R3 = R1 + R2;
		std::cout << "R1 + R2 = " << std::endl <<        R3           << std::endl << std::endl;
		R3 = R1 - R2;
		std::cout << "R1 - R2 = " << std::endl <<        R3           << std::endl << std::endl;
		exit(0);

	}

};