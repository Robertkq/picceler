#include "runtime.h"

#include <iostream>

#include "spdlog/spdlog.h"

#include "image.h"

extern "C" {

void *piccelerLoadImage(const char *filename) {
  std::cout << "piccelerLoadImage called with filename: " << filename
            << std::endl;
  picceler::Image *imgPtr = picceler::loadImage(std::string(filename));

  return imgPtr;
}

void *piccelerBlurImage(void *image, const char *mode) {
  std::cout << "piccelerBlurImage called with mode: " << mode << std::endl;
  return image;
}

void piccelerShowImage(void *image) {
  std::cout << "piccelerShowImage called" << std::endl;
  picceler::Image *imgPtr = static_cast<picceler::Image *>(image);
  picceler::showImage(*imgPtr);
}

void piccelerSaveImage(void *image, const char *filename) {
  std::cout << "piccelerSaveImage called with filename: " << filename
            << std::endl;
  picceler::Image *imgPtr = static_cast<picceler::Image *>(image);
  picceler::saveImage(*imgPtr, std::string(filename));
}
}
