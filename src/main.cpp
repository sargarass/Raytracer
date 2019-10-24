#include <iostream>
#include <fstream>
#include "common.h"
#include "ray.h"
#include "triangle.h"

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

struct object {
    enum class type_t {
        polygonal,
        sphere
    };

    type_t type;
    float3 *mesh = nullptr;
    size_t mesh_size;
};

__device__
float hitSphere(float3 const &center, float radious, core::ray const &ray) noexcept {
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
float3 color(object const *objects, uint32_t const objects_count, core::ray const &ray) {
    float t = FLT_MAX;
    for (uint32_t i = 0; i < objects_count; ++i) {
        switch (objects[i].type) {
            case object::type_t::sphere: {
                float3 origin = objects[i].mesh[0];
                float radious = objects[i].mesh[1].x;
                float t = hitSphere(origin, radious, ray);
                if (t > 0) {
                    auto normal = normalize(ray.pointAtParameter(t) - origin);
                    return 0.5f * float3 { normal.x + 1.0f, normal.y + 1.0f, normal.z + 1.0f };
                }
                break;
            }
            case object::type_t::polygonal: {

            }
        }
    }

    
    float3 unitDirection = normalize(ray.direction());
    t = 0.5f * (unitDirection.y + 1.0f);
    return (1.0f - t) * float3 {1.0f, 1.0f, 1.0f} + t * float3{0.5, 0.7, 1.0};
}

__global__
void render(float3 *framebuffer, uint32_t const width, uint32_t const height, object const *objects, uint32_t const objects_count) {
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
    core::ray ray(origin, lower_left_corner + u * horizontal + v * vertical);
    framebuffer[pixel_index] = color(objects, objects_count, ray);
}

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
    auto meshes = hip_alloc_managed_unique<float3[]>(2 * (2) + 1 * 3);
    constexpr size_t objects_count = 3;

    auto objects = hip_alloc_managed_unique<object[]>(3);
    objects[0] = object {object::type_t::sphere, meshes.get() + 0, 2};
    objects[1] = object {object::type_t::sphere, meshes.get() + 2, 2};
    objects[2] = object {object::type_t::polygonal, meshes.get() + 4, 3};

    float3 sphere_0[2] { float3{0, 0, -1}, float3{ 0.5 } };
    float3 sphere_1[2] { float3{0, -100.5, -1}, float3{ 100 } };
    float3 triangle[3] { float3{-2, 0, -1}, float3{0, 0, -1}, float3{-1, 1, -1} };

    memcpy(objects[0].mesh, &sphere_0, sizeof(sphere_0));
    memcpy(objects[1].mesh, &sphere_1, sizeof(sphere_1));
    memcpy(objects[2].mesh, &triangle, sizeof(triangle));


    hipLaunchKernelGGL(setupRender,
                       dim3(width / threadsPerBlockX + 1, height / threadsPerBlockY + 1),
                       dim3(threadsPerBlockX, threadsPerBlockY),
                       0, 0,
                       framebuffer.get(), width, height);
    
    hipLaunchKernelGGL(render,
                       dim3(width / threadsPerBlockX + 1, height / threadsPerBlockY + 1),
                       dim3(threadsPerBlockX, threadsPerBlockY),
                       0, 0,
                       framebuffer.get(), width, height, objects.get(), objects_count);
    
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
