#pragma once

#include <stdint.h>
#include <opencv2/opencv.hpp>

#include "tegra_stereo/util.h"
#include "tegra_stereo/configuration.h"
#include "tegra_stereo/costs.h"
#include "tegra_stereo/hamming_cost.h"
#include "tegra_stereo/median_filter.h"
#include "tegra_stereo/cost_aggregation.h"
#include "tegra_stereo/debug.h"

void init_disparity_method(const uint8_t _p1, const uint8_t _p2);
cv::Mat compute_disparity_method(cv::Mat left, cv::Mat right, float *elapsed_time_ms);
void finish_disparity_method();
static void free_memory();
