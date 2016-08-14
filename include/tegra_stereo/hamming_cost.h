#pragma once

#include "tegra_stereo/configuration.h"
#include "tegra_stereo/util.h"
#include <stdint.h>

__global__ void
HammingDistanceCostKernel (  const cost_t *d_transform0, const cost_t *d_transform1,
		uint8_t *d_cost, const int rows, const int cols );

