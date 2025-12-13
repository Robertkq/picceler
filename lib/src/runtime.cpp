#include <iostream>

#include "runtime.h"

extern "C" {

void *piccelerLoadImage(const char *filename) {
  std::cout << "piccelerLoadImage called with filename: " << filename
            << std::endl;
  return nullptr;
}

void *piccelerBlurImage(void *image, const char *mode) {
  std::cout << "piccelerBlurImage called with mode: " << mode << std::endl;
  return image;
}

void piccelerShowImage(void *image) {
  std::cout << "piccelerShowImage called" << std::endl;
}

void piccelerSaveImage(void *image, const char *filename) {
  std::cout << "piccelerSaveImage called with filename: " << filename
            << std::endl;
}
}
