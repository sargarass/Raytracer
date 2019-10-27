#include <iostream>
#include <fstream>
#include "common.h"
#include "kernel.h"

int main() {
    constexpr size_t width = 1680;
    constexpr size_t height = 720;
    constexpr size_t framebuffer_size = width * height * sizeof(float3);
    
    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    
    writeLog(General, "System minor %d", devProp.minor);
    writeLog(General, "System major %d", devProp.major);
    writeLog(General, "Device name %s", devProp.name);
    writeLog(General, "hip Device prop succeeded");
    auto framebuffer = hip_alloc_managed_unique<float3[]>(framebuffer_size);

    // 2 spheres & 1 triangle
    auto meshes = hip_alloc_managed_unique<float3[]>(2 * (2) + 1 * 36);
    constexpr size_t objects_count = 3;

    auto objects = hip_alloc_managed_unique<object[]>(3);
    objects[0] = object {object_type::sphere, meshes.get() + 0, 2};
    objects[1] = object {object_type::sphere, meshes.get() + 2, 2};
    objects[2] = object {object_type::polygonal, meshes.get() + 4, 36};



    float3 sphere_0[2] { float3{0, 0, -1}, float3{ 0.1f } };
    float3 sphere_1[2] { float3{0, -100.5, -1}, float3{ 100 } };
    float3 triangle[36] { float3{-1.0f, -1.0,   1.0f}, float3{ 1.0f, -1.0f,  1.0f}, float3{ 1.0f,  1.0f,  1.0f},
                          float3{-1.0f, -1.0f,  1.0f}, float3{ 1.0f,  1.0f,  1.0f}, float3{-1.0f,  1.0f,  1.0f},
                          float3{ 1.0f, -1.0f,  1.0f}, float3{ 1.0f, -1.0f, -1.0f}, float3{ 1.0f,  1.0f, -1.0f},
                          float3{ 1.0f, -1.0f,  1.0f}, float3{ 1.0f,  1.0f, -1.0f}, float3{ 1.0f,  1.0f,  1.0f},
                          float3{ 1.0f, -1.0f, -1.0f}, float3{-1.0f, -1.0f, -1.0f}, float3{-1.0f,  1.0f, -1.0f},
                          float3{ 1.0f, -1.0f, -1.0f}, float3{-1.0f,  1.0f, -1.0f}, float3{ 1.0f,  1.0f, -1.0f},
                          float3{-1.0f, -1.0f, -1.0f}, float3{-1.0f, -1.0f,  1.0f}, float3{-1.0f,  1.0f,  1.0f},
                          float3{-1.0f, -1.0f, -1.0f}, float3{-1.0f,  1.0f,  1.0f}, float3{-1.0f,  1.0f,  1.0f},
                          float3{-1.0f,  1.0f,  1.0f}, float3{ 1.0f,  1.0f,  1.0f}, float3{ 1.0f,  1.0f, -1.0f},
                          float3{-1.0f,  1.0f,  1.0f}, float3{ 1.0f,  1.0f, -1.0f}, float3{-1.0f,  1.0f, -1.0f},
                          float3{ 1.0f, -1.0f,  1.0f}, float3{-1.0f, -1.0f, -1.0f}, float3{ 1.0f, -1.0f, -1.0f},
                          float3{ 1.0f, -1.0f,  1.0f}, float3{-1.0f, -1.0f,  1.0f}, float3{-1.0f, -1.0f, -1.0f} };

    for (auto &vec : triangle) {
        vec.z -= 3.0f;
        vec.x -= 2.0f;
    }

    memcpy(objects[0].mesh, &sphere_0, sizeof(sphere_0));
    memcpy(objects[1].mesh, &sphere_1, sizeof(sphere_1));
    memcpy(objects[2].mesh, &triangle, sizeof(triangle));

    kernel::global_data host_kg(framebuffer.get(), width, height, objects.get(), objects_count);
    HIP_ASSERT(hipMemcpyToSymbol(&kernel::kg, &host_kg, sizeof(kernel::global_data)));

    hipLaunchKernelGGL(kernel::setupRender,
                       dim3(width / threadsPerBlockX + 1, height / threadsPerBlockY + 1),
                       dim3(threadsPerBlockX, threadsPerBlockY),
                       0, 0);
    
    hipLaunchKernelGGL(kernel::render,
                       dim3(width / threadsPerBlockX + 1, height / threadsPerBlockY + 1),
                       dim3(threadsPerBlockX, threadsPerBlockY),
                       0, 0);
    
    hipDeviceSynchronize();
    writeLog(General, "setupRender finished");
        
    std::ofstream fout("result.ppm");;
    fout << "P3\n" << width << " " << height << "\n255\n";
    for (int j = height - 1; j >= 0; j--) {
        for (size_t i = 0; i < width; i++) {
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
