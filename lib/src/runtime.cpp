#include "runtime.h"

#include <iostream>

#include "spdlog/spdlog.h"

#include "image.h"

extern "C" {

picceler::Image *piccelerLoadImage(const char *filename) {
  std::cout << "piccelerLoadImage called with filename: " << filename
            << std::endl;
  picceler::Image *imgPtr = picceler::loadImage(std::string(filename));
  printf("Size: %zu, Width Off: %zu, Height Off: %zu, Data Off: %zu\n",
         sizeof(picceler::Image), offsetof(picceler::Image, _width),
         offsetof(picceler::Image, _height), offsetof(picceler::Image, _data));

  return imgPtr;
}

picceler::Image piccelerBlurImage(picceler::Image image, const char *mode) {
  std::cout << "piccelerBlurImage called with mode: " << mode << std::endl;
  return image;
}

void piccelerShowImage(picceler::Image *image) {
  std::cout << "piccelerShowImage called" << std::endl;
  picceler::showImage(*image);
}

void piccelerSaveImage(picceler::Image *image, const char *filename) {
  std::cout << "piccelerSaveImage called with filename: " << filename
            << std::endl;
  picceler::saveImage(*image, std::string(filename));
}

picceler::Image *piccelerCreateImage(uint32_t width, uint32_t height) {
  std::cout << "piccelerCreateImage called with " << width << " x " << height
            << std::endl;
  picceler::Image *newImage = new picceler::Image();
  newImage->_width = width;
  newImage->_height = height;
  constexpr auto channels = 4;
  newImage->_data = new unsigned char[width * height * channels];

  return newImage;
}
}
