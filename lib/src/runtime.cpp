#include "runtime.h"
#include <iostream>

extern "C" {

void *picceler_load_image(const char *filename) {
  std::cout << "picceler_load_image called with filename: " << filename
            << std::endl;
  // Return a dummy pointer
  return reinterpret_cast<void *>(0x1234);
}

void *picceler_blur_image(void *image, const char *mode) {
  std::cout << "picceler_blur_image called with mode: " << mode << std::endl;
  // Return the same dummy pointer
  return image;
}

void picceler_show_image(void *image) {
  std::cout << "picceler_show_image called" << std::endl;
}

void picceler_save_image(void *image, const char *filename) {
  std::cout << "picceler_save_image called with filename: " << filename
            << std::endl;
}
}
