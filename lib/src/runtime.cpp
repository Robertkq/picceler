#include "runtime.h"

#include <cstddef>
#include <iostream>

#include "spdlog/spdlog.h"

#include "image.h"

extern "C" {

void piccelerLoadImage(const char *filename, uint8_t **data, int64_t *height, int64_t *width) {
  picceler::Image *img = picceler::loadImage(std::string(filename));
  if (!img || !img->_data) {
    *data = nullptr;
    *height = 0;
    *width = 0;
    return;
  }

  *data = img->_data;
  *height = static_cast<int64_t>(img->_height);
  *width = static_cast<int64_t>(img->_width);
}

void piccelerShowImage(void *data, uint32_t width, uint32_t height) {
  spdlog::debug("piccelerShowImage called");
  picceler::Image image{width, height, static_cast<unsigned char *>(data)};
  picceler::showImage(image);
}

void piccelerSaveImage(void *data, uint32_t width, uint32_t height, const char *filename) {
  spdlog::debug("piccelerSaveImage called with filename: {}", filename);
  picceler::Image image{width, height, static_cast<unsigned char *>(data)};
  picceler::saveImage(image, std::string(filename));
}

void *piccelerReadString(const char *prompt) {
  spdlog::debug("piccelerReadString called with prompt: {}", prompt);
  std::string *result = new std::string(); // for now just let it leak

  std::cout << prompt;
  std::getline(std::cin, *result);
  return static_cast<void *>(const_cast<char *>(result->c_str()));
}

double piccelerReadNumber(const char *prompt) {
  spdlog::debug("piccelerReadNumber called with prompt: {}", prompt);
  double *result = new double(); // for now just let it leak

  std::cout << prompt;
  std::cin >> *result;
  std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  return *result;
}

} // extern "C"
