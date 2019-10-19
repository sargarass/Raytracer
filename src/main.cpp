#include <iostream>
#include <fstream>
#include "common.h"
#include "ray.h"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

__global__ 
void setupRender(float3 *framebuffer, const int width, const int height) {
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

__device__
float hitSphere(float3 const &center, float radious, Ray const &ray) noexcept {
    float3 oc = ray.origin() - center;
    float a = dot(ray.direction(), ray.direction());
    float b = 2.0f * dot(oc, ray.direction());
    float c = dot(oc, oc) - radious * radious;
    float discriminant = b * b - 4.0f * a *c;
    if (discriminant < 0.0f) {
        return -1.0f;
    }
    return (-b - sqrtf(discriminant)) / (2.0f * a);
}

__device__
float3 color(Ray const &ray) {
    constexpr float3 sphereOrigin { 0.0f, 0.0f, -1.0f };
    constexpr float  sphereRadious { 0.5f };
    float t = hitSphere(sphereOrigin, sphereRadious, ray);
    if (t > 0) {
        auto normal = normalize(ray.pointAtParameter(t) - sphereOrigin);
        return 0.5f * float3 { normal.x + 1.0f, normal.y + 1.0f, normal.z + 1.0f };
    }
    
    float3 unitDirection = normalize(ray.direction());
    t = 0.5f * (unitDirection.y + 1.0f);
    return (1.0f - t) * float3 {1.0f, 1.0f, 1.0f} + t * float3{0.5, 0.7, 1.0};
}

__global__
void render(float3 *framebuffer, const int width, const int height) {
    constexpr float3 lower_left_corner { -2.0f, -0.857f, -1.0f };
    constexpr float3 horizontal { 4.0f, 0.0f, 0.0f };
    constexpr float3 vertical { 0.0f, 1.714, 0.0f };
    constexpr float3 origin { 0.0, 0.0, 0.0 };
    
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
    if ((x >= width) || (y >= height)) {
        return;
    }
    
    int pixel_index = y * width + x;    
    float u = static_cast<float>(x) / width;
    float v = static_cast<float>(y) / height;
    Ray ray(origin, lower_left_corner + u * horizontal + v * vertical);
    framebuffer[pixel_index] = color(ray);
}


int main() {
    float3* framebuffer;
    constexpr int width = 1680;
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
    
    hipLaunchKernelGGL(render,
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
