#pragma once
#include <string>
#include <opencv2/core.hpp>

namespace sel
{
	namespace opencv
	{
		using namespace cv;
		namespace  utils
		{
			std::string GetMatDepth(const cv::Mat& mat)
			{
				const int depth = mat.depth();

				switch (depth)
				{
				case CV_8U:  return "CV_8U";
				case CV_8S:  return "CV_8S";
				case CV_16U: return "CV_16U";
				case CV_16S: return "CV_16S";
				case CV_32S: return "CV_32S";
				case CV_32F: return "CV_32F";
				case CV_64F: return "CV_64F";
				default:
					return "Invalid depth type of matrix!";
				}
			}

			std::string GetMatType(const cv::Mat& mat)
			{
				const int mtype = mat.type();

				switch (mtype)
				{
				case CV_8UC1:  return "CV_8UC1";
				case CV_8UC2:  return "CV_8UC2";
				case CV_8UC3:  return "CV_8UC3";
				case CV_8UC4:  return "CV_8UC4";

				case CV_8SC1:  return "CV_8SC1";
				case CV_8SC2:  return "CV_8SC2";
				case CV_8SC3:  return "CV_8SC3";
				case CV_8SC4:  return "CV_8SC4";

				case CV_16UC1: return "CV_16UC1";
				case CV_16UC2: return "CV_16UC2";
				case CV_16UC3: return "CV_16UC3";
				case CV_16UC4: return "CV_16UC4";

				case CV_16SC1: return "CV_16SC1";
				case CV_16SC2: return "CV_16SC2";
				case CV_16SC3: return "CV_16SC3";
				case CV_16SC4: return "CV_16SC4";

				case CV_32SC1: return "CV_32SC1";
				case CV_32SC2: return "CV_32SC2";
				case CV_32SC3: return "CV_32SC3";
				case CV_32SC4: return "CV_32SC4";

				case CV_32FC1: return "CV_32FC1";
				case CV_32FC2: return "CV_32FC2";
				case CV_32FC3: return "CV_32FC3";
				case CV_32FC4: return "CV_32FC4";

				case CV_64FC1: return "CV_64FC1";
				case CV_64FC2: return "CV_64FC2";
				case CV_64FC3: return "CV_64FC3";
				case CV_64FC4: return "CV_64FC4";

				default:
					return "Invalid type of matrix!";
				}
			}


			constexpr float rect_overlap(cv::Rect& a, cv::Rect& b)
			{
				const auto a_area = a.width * a.height;
				const auto b_area = b.width * b.height;
				const auto a_right = a.x + a.width;
				const auto b_right = b.x + b.width;
				const auto a_bottom = a.y + a.height;
				const auto b_bottom = b.y + b.height;

				const auto intersect_area = std::max(0, std::min(a_right, b_right) - std::max(a.x, b.x)) *
					std::max(0, std::min(a_bottom, b_bottom) - std::max(a.y, b.y));

				const auto union_area = a_area + b_area - intersect_area;

				return static_cast<float>(intersect_area) / union_area;
			}
			
		}
	}
}
