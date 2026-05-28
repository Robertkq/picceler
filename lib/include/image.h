#pragma once

#include <opencv2/opencv.hpp>

namespace picceler {

/**
 * @brief A class representing an image in the Picceler framework.
 */
class Image {
public:
  Image() : _width(0), _height(0), _data(nullptr) {}

  uint32_t _width;
  uint32_t _height;
  unsigned char *_data;
};

/**
 * @brief Load an image from a file.
 * @param filename The path to the image file.
 * @return A pointer to the loaded image, or nullptr if loading failed.
 */
Image *loadImage(const std::string &filename);

/**
 * @brief Save an image to a file.
 * @param image The image to save.
 * @param filename The path to the output file.
 */
void saveImage(const Image &image, const std::string &filename);

/**
 * @brief Display an image in a window.
 * @param image The image to display.
 */
void showImage(const Image &image);

} // namespace picceler