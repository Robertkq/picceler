#pragma once

#include <opencv2/opencv.hpp>

namespace picceler {

class Image {
public:
  Image() : _width(0), _height(0), _data(nullptr) {}

  uint32_t _width;
  uint32_t _height;
  unsigned char *_data;
};

Image *loadImage(const std::string &filename);
void saveImage(const Image &image, const std::string &filename);
void showImage(const Image &image);

} // namespace picceler