#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void *picceler_load_image(const char *filename);
void *picceler_blur_image(void *image, const char *mode);
void picceler_show_image(void *image);
void picceler_save_image(void *image, const char *filename);

#ifdef __cplusplus
}
#endif
