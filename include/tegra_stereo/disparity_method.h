#pragma once

#include <stdint.h>
#include <opencv2/opencv.hpp>

void init_disparity_method(const uint8_t _p1, const uint8_t _p2);
cv::Mat compute_disparity_method(cv::Mat left, cv::Mat right, float *elapsed_time_ms);
void finish_disparity_method();
static void free_memory();
