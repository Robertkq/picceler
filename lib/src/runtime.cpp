#include "runtime.h"

#include <iostream>

#include "spdlog/spdlog.h"

#include "image.h"

extern "C" {

picceler::Image *piccelerLoadImage(const char *filename) {
  spdlog::debug("piccelerLoadImage called with filename: {}", filename);
  picceler::Image *imgPtr = picceler::loadImage(std::string(filename));
  spdlog::debug("Size: {}, Width Off: {}, Height Off: {}, Data Off: {}", sizeof(picceler::Image),
                offsetof(picceler::Image, _width), offsetof(picceler::Image, _height),
                offsetof(picceler::Image, _data));

  return imgPtr;
}

void piccelerShowImage(picceler::Image *image) {
  spdlog::debug("piccelerShowImage called");
  picceler::showImage(*image);
}

void piccelerSaveImage(picceler::Image *image, const char *filename) {
  spdlog::debug("piccelerSaveImage called with filename: {}", filename);
  picceler::saveImage(*image, std::string(filename));
}

picceler::Image *piccelerCreateImage(uint32_t width, uint32_t height) {
  spdlog::debug("piccelerCreateImage called with width: {}, height: {}", width, height);
  picceler::Image *newImage = new picceler::Image();
  newImage->_width = width;
  newImage->_height = height;
  constexpr auto channels = 4;
  newImage->_data = new unsigned char[width * height * channels];

  return newImage;
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
