#pragma once 
#include <math.h>

class Ray {
public:
    constexpr __host__ __device__ 
    Ray() noexcept {}
    
    constexpr __host__ __device__ 
    Ray(float3 const &orig, float3 const &dir) noexcept
    : orig { orig }, dir { dir }
    {}
    
    constexpr __host__ __device__ inline
    float3 const &origin() const noexcept { return orig; }
    
    constexpr __host__ __device__ inline
    float3 const &direction() const noexcept { return dir; }
    
    float3 pointAtParameter(float const t) const noexcept { return orig + t * dir; }
private:
    float3 orig = {0.0, 0.0, 0.0};
    float3 dir = {0.0, 0.0, 0.0};
};
