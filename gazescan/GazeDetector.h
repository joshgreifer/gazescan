#pragma once

#pragma warning(push)
#pragma warning(disable : 4244 4267)
#include <tiny_dnn/tiny_dnn.h>
#pragma warning(pop)

#include <opencv2/dnn/dnn.hpp>


class NetImpl
{
private:
	tiny_dnn::network<tiny_dnn::sequential>Net;

	void Mat2Vec(cv::Mat& mat, std::vector<float>& vec, const float scale) {

		const auto& m = (cv::Mat_<uint8_t>&) mat;

		std::transform(m.begin(), m.end(), std::back_inserter(vec),
			[=](uint8_t c) {
				return c * scale;
			});

	}

public:
	NetImpl()
	{

	}

	cv::Mat predict(cv::Mat& image) 
	{
		std::vector<float> vec;
		Mat2Vec(image, vec, 1 / 256.0f);
		auto result_vec = Net.predict(vec);
		std::vector<float> vv(result_vec.begin(), result_vec.end());
		return cv::Mat(vv);

	}

};

/*
 * Detect the gaze direction given an input image, 32 X 16 of two eyes, which was used to train a DNN
 *
 */

class GazeDetector
{
	// See gazescan.m 
//	Load the CNN we trained and exported in Matlab
	cv::dnn::Net TrainedNet;

	tiny_dnn::network<tiny_dnn::sequential>Net;

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

	explicit GazeDetector(const std::string& onnx_filename) : TrainedNet(cv::dnn::readNetFromONNX(onnx_filename))
	{
		//std::cout <<"[\n";
		//for (auto& layer_name : TrainedNet.getLayerNames())
		//	std::cout << '\t' << layer_name << '\n';
		//std::cout <<"]\n";

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

	void average(cv::Mat& average_image ) const
	{
		if (num_images_averaged_ > 1)
			average_image = accumulated_sum_image_ / static_cast<double>(num_images_averaged_);

	}
	void  average_img(cv::Mat& average_image) const
	{
		average_image = accumulated_sum_image_ / static_cast<double>(num_images_averaged_);
		average_image.convertTo(average_image, CV_8UC1, 256.0);

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

	cv::Mat detect(cv::Mat& detection_image)
	{

		cv::resize(detection_image, detection_image, ImageSize());



//		detection_image.convertTo(detection_image, CV_64F, 1/255.0);
//		cv::Mat average_image; average(average_image);
//		detection_image = detection_image - average_image;
//		cv::normalize(detection_image, detection_image, -1.0, 1.0, cv::NORM_MINMAX, CV_32F);
//		detection_image.convertTo(detection_image, CV_32F);
		const auto blob = cv::dnn::blobFromImage(detection_image,1.0, ImageSize(), 0.0);
		std::cerr << "detection_image: "  << sel::opencv::utils::GetMatType(detection_image) << std::endl;
		TrainedNet.setInput(blob);

		return TrainedNet.forward("softmax");

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
	}

};