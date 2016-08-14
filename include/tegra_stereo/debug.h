#pragma once

#include <iostream>
#include <stdio.h>
#include "tegra_stereo/configuration.h"

template<typename T>
void write_file(const char* fname, const T *data, const int size) {
	FILE* fp = fopen(fname, "wb");
	if (fp == NULL) {
		std::cerr << "Couldn't write transform file" << std::endl;
		exit(-1);
	}
	fwrite (data, sizeof(T), size, fp);
	fclose(fp);
}

void debug_log(const char *str);
