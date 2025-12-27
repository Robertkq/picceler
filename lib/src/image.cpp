#include "image.h"

#include "spdlog/spdlog.h"

namespace picceler {

Image *loadImage(const std::string &filename) {
  Image *img = new Image();
  cv::Mat loaded = cv::imread(filename);
  if (loaded.empty()) {
    spdlog::error("Failed to load image: {}", filename);
    return img;
  }

  cv::Mat rgba;
  cv::cvtColor(loaded, rgba, cv::COLOR_BGR2RGBA);

  img->_width = rgba.cols;
  img->_height = rgba.rows;
  img->_data = new unsigned char[rgba.total() * rgba.elemSize()];
  std::memcpy(img->_data, rgba.data, rgba.total() * rgba.elemSize());

  spdlog::info("Loaded image: {} ({}x{})", filename, img->_width, img->_height);
  spdlog::info(
      "OpenCV image details - cols: {}, rows: {}, channels: {}, total: {}",
      rgba.cols, rgba.rows, rgba.channels(), rgba.total());

  return img;
}

void saveImage(const Image &image, const std::string &filename) {

  cv::Mat rgbaMat(image._height, image._width, CV_8UC4, image._data);

  cv::Mat bgrMat;
  cv::cvtColor(rgbaMat, bgrMat, cv::COLOR_RGBA2BGR);
  if (!cv::imwrite(filename, bgrMat)) {
    spdlog::error("Failed to save image: {}", filename);
    return;
  }
  spdlog::info("Saved image: {}", filename);
}

void showImage(const Image &image) {
  cv::Mat rgbaMat(image._height, image._width, CV_8UC4, image._data);

  cv::Mat bgrMat;
  cv::cvtColor(rgbaMat, bgrMat, cv::COLOR_RGBA2BGR);
  cv::namedWindow("Image", cv::WINDOW_NORMAL);
  cv::imshow("Image", bgrMat);
  while (cv::getWindowProperty("Image", cv::WND_PROP_VISIBLE) >= 1) {
    if (cv::waitKey(100) >= 0)
      break;
  }
  cv::destroyWindow("Image");
  cv::waitKey(1);
}

} // namespace picceler