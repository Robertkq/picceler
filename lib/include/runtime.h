#pragma once

#include "image.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Wrapper C-style functions for proper picceler runtime implementation in C++
 * \{
 */

picceler::Image *piccelerLoadImage(const char *filename);
picceler::Image piccelerBlurImage(picceler::Image image, const char *mode);
void piccelerShowImage(picceler::Image *image);
void piccelerSaveImage(picceler::Image *image, const char *filename);
picceler::Image *piccelerCreateImage(uint32_t width, uint32_t height);

/**
 * \}
 */

#ifdef __cplusplus
}
#endif
