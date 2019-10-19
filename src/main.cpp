#include <cassert>
#include <cstdio>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cfloat>
#include <map>
#include <fstream>

#include <hip/hip_runtime.h>
#include <hiprand_kernel.h>

#include "logger.h"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

template<typename T>
static __host__ __device__
T clamp(T x, T a, T b)
{
	return max(a, min(x, b));
}

constexpr size_t threadsPerBlockX = 16;
constexpr size_t threadsPerBlockY = 16;
constexpr size_t threadsPerBlockZ = 1;

__global__ void setupRender(float3 *framebuffer, const int width, const int height) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= width) || (y >= height)) {
        return;
    }
    
    int pixel_index = y * width + x;
    
    framebuffer[pixel_index].x = 1.0f;
    framebuffer[pixel_index].y = 1.0f;
    framebuffer[pixel_index].z = 1.0f;
}

int main() {
    float3* framebuffer;
    constexpr int width = 1280;
    constexpr int height = 720;
    constexpr int framebuffer_size = width * height * sizeof(float3);
    
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    
    writeLog(General, "System minor %d", devProp.minor);
    writeLog(General, "System major %d", devProp.major);
    writeLog(General, "Device name %s", devProp.name);
    writeLog(General, "hip Device prop succeeded");
    HIP_ASSERT(hipMallocManaged(&framebuffer, framebuffer_size));
        
    hipLaunchKernelGGL(setupRender,
                       dim3(width / threadsPerBlockX + 1, height / threadsPerBlockY + 1),
                       dim3(threadsPerBlockX, threadsPerBlockY),
                       0, 0,
                       framebuffer, width, height);
    hipDeviceSynchronize();
    writeLog(General, "setupRender finished");
    
    std::ofstream fout("result.ppm");;
    fout << "P3\n" << width << " " << height << "\n255\n";
    for (int j = height - 1; j >= 0; j--) {
        for (int i = 0; i < width; i++) {
            size_t pixel_index = j * width + i;
            float r = sqrtf(framebuffer[pixel_index].x);
            float g = sqrtf(framebuffer[pixel_index].y);
            float b = sqrtf(framebuffer[pixel_index].z);
            int ir = static_cast<int>( clamp(254.99f*(r), 0.0f, 255.0f) );
            int ig = static_cast<int>( clamp(254.99f*(g), 0.0f, 255.0f) );
            int ib = static_cast<int>( clamp(254.99f*(b), 0.0f, 255.0f) );
            fout << ir << " " << ig << " " << ib << "\n";
        }
    }
    fout.close();
    return 0;
}
