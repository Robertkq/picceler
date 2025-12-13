#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Wrapper C-style functions for proper picceler runtime implementation in C++
 * \{
 */

void *piccelerLoadImage(const char *filename);
void *piccelerBlurImage(void *image, const char *mode);
void piccelerShowImage(void *image);
void piccelerSaveImage(void *image, const char *filename);

/**
 * \}
 */

#ifdef __cplusplus
}
#endif
