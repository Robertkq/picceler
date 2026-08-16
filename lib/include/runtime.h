#pragma once

#include "image.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Wrapper C-style functions for proper picceler runtime implementation in C++
 * \{
 */

void piccelerLoadImage(const char *filename, uint8_t **data, int64_t *height, int64_t *width);
void piccelerShowImage(void *data, uint32_t width, uint32_t height);
void piccelerSaveImage(void *data, uint32_t width, uint32_t height, const char *filename);

void *piccelerReadString(const char *prompt);
double piccelerReadNumber(const char *prompt);

void piccelerPrintString(const char *str);

void piccelerPrintFloat64(double value);

/**
 * \}
 */

#ifdef __cplusplus
}
#endif
