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

public:

	auto& net() const
	{
		return TrainedNet;
	}

	explicit GazeDetector(const std::string& onnx_filename)
	{
		TrainedNet = cv::dnn::readNetFromONNX(onnx_filename);
	}

	auto predict(cv::Mat& eyes_combined, float& confidence)
	{
		auto results = detect(eyes_combined);

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

	cv::Mat detect(cv::Mat& eyes_combined)
	{
		const auto blob = cv::dnn::blobFromImage(eyes_combined, 1/256.0, eyes_combined.size(), 128);
//		std::cerr << "eyes_combined: "  << sel::opencv::utils::GetMatType(eyes_combined) << std::endl;
		TrainedNet.setInput(blob);


		return TrainedNet.forward("softmax");;

	}
};