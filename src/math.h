#pragma once
#include <algorithm>
#include <cmath>
#include <hip/hip_runtime.h>

template<typename T>
__host__ __device__ static inline constexpr
T clamp(T const &x, float a, float b) noexcept {
    return std::max(a, std::min(x, b));
} 

template<>
__host__ __device__ inline constexpr
float4 clamp(float4 const &v, float a, float b) noexcept {
    return {clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b)};
} 

template<>
__host__ __device__ inline constexpr
float3 clamp(float3 const &v, float a, float b) noexcept {
    return {clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b)};
} 

template<>
__host__ __device__ inline constexpr
float2 clamp(float2 const &v, float a, float b) noexcept {
    return {clamp(v.x, a, b), clamp(v.y, a, b)};
} 

__host__ __device__ static inline constexpr
float2 &operator+=(float2 &a, float2 const &b) noexcept {
    a.x += b.x;
    a.y += b.y;
    return a;
}

__host__ __device__ static inline constexpr
float2 &operator*=(float2 &a, float2 const &b) noexcept {
    a.x *= b.x;
    a.y *= b.y;
    return a;
}

__host__ __device__ static inline constexpr
float2 operator-(float2 const &a, float2 const &b) noexcept {
    return {a.x - b.x, a.y - b.y};
}

__host__ __device__ static inline constexpr
float2 operator+(float2 const &a, float2 const &b) noexcept {
    return {a.x + b.x, a.y + b.y};
}

__host__ __device__ static inline constexpr
float2 operator*(float2 const &a, float c) noexcept {
    return {a.x * c, a.y * c};
}

__host__ __device__ static inline constexpr
float2 operator*(float c, float2 const &a) noexcept {
    return {a.x * c, a.y * c};
}

__host__ __device__ static inline constexpr
float length(float2 const &a) noexcept {
    return sqrtf(a.x * a.x + a.y * a.y);
}

__host__ __device__ static inline constexpr
float3 make_float3(float4 const &f4) noexcept {
    return {f4.x, f4.y, f4.z};
}

__host__ __device__ static inline constexpr
float3 make_float3(float2 const &xy, float z) noexcept {
    return {xy.x, xy.y, z};
}

__host__ __device__ static inline constexpr
float3 make_float3(float x) noexcept {
    return {x, x, x};
}

__host__ __device__ static inline constexpr
float3 &operator+=(float3 &a, float3 const &b) noexcept
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__host__ __device__ static inline constexpr
float3 & operator-=(float3 &a, float3 const &b) noexcept
{
    a.x -= b.x; 
    a.y -= b.y; 
    a.z -= b.z;
    return a;
}

__host__ __device__ static inline constexpr
float3 & operator *= (float3 &a, float3 const &b) noexcept
{
    a.x *= b.x; 
    a.y *= b.y; 
    a.z *= b.z;
    return a;
}

__host__ __device__ static inline constexpr
float3 & operator *= (float3 &a, float c) noexcept
{
    a.x *= c;
    a.y *= c;
    a.z *= c;
    return a;
}

__host__ __device__ static inline constexpr
float3 & operator /= (float3 &a, float3 const &b) noexcept {
    a.x /= b.x; 
    a.y /= b.y; 
    a.z /= b.z;
    return a;
}

__host__ __device__ static inline constexpr
float3 & operator /= (float3 &a, float c) noexcept {
    a.x /= c;
    a.y /= c;
    a.z /= c;
    return a;
}

__host__ __device__ static inline constexpr
float3 operator-(float3 const &a) noexcept {
    return {-a.x, -a.y, -a.z};
}

__host__ __device__ static inline constexpr
float3 operator+(float3 const &a, float3 const &b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__host__ __device__ static inline constexpr
float3 operator+(float3 const &a, float c) noexcept {
    return {a.x + c, a.y + c, a.z + c};
}

__host__ __device__ static inline constexpr
float3 operator-(float3 const &a, float3 const &b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__host__ __device__ static inline constexpr
float3 operator-(float3 const &a, float c) noexcept {
    return {a.x - c, a.y - c, a.z - c};
}

__host__ __device__ static inline constexpr
float3 operator*(float3 const &a, float3 const &b) noexcept {
    return {a.x * b.x, a.y * b.y, a.z * b.z};
}

__host__ __device__ static inline constexpr
float3 operator*(float3 const &a, float c) noexcept {
    return {a.x * c, a.y * c, a.z * c};
}

__host__ __device__ static inline constexpr
float3 operator * (float c, float3 const &a) noexcept {
    return {a.x * c, a.y * c, a.z * c};
}

__host__ __device__ static inline constexpr
float3 operator / (float3 const &a, float3 b) {
    return {a.x / b.x, a.y / b.y, a.z / b.z};
}

__host__ __device__ static inline constexpr
float3 operator/(float3 const &a, float c) noexcept
{
    float rc = 1.0f / c;
    return {a.x * rc, a.y * rc, a.z * rc};
}

__host__ __device__ static inline constexpr
float3 operator/(float c, float3 const &a) noexcept {
    return {c / a.x, c / a.y, c / a.z};
}

__host__ __device__ static inline constexpr
float3 cross(float3 const &a, float3 const &b) noexcept {
    return {a.y * b.z - a.z * b.y, 
            a.z * b.x - a.x * b.z, 
            a.x * b.y - a.y * b.x};
}

__host__ __device__ static inline constexpr
float dot(float3 const &a, float3 const &b) noexcept {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ static inline constexpr
float dot(float2 const &a, float2 const &b) noexcept {
    return a.x * b.x + a.y * b.y;
}

__host__ __device__ static inline constexpr
float length(float3 const &a) noexcept {
    return sqrtf(a.x * a.x + a.y * a.y + a.z * a.z);
}

__host__ __device__ static inline constexpr
float3 normalize(float3 const &a) noexcept {
    float c = 1.0f / (length(a) + 1e-15f);
    return {a.x * c, a.y * c, a.z * c};
}

__host__ __device__ static inline constexpr
float4 operator+(float4 const &a, float4 const &b) noexcept {
    return {a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w};
}

__host__ __device__ static inline constexpr
float4 operator-(float4 const &a, float4 const &b) noexcept {
    return {a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w};
}

__host__ __device__ static inline constexpr
float4 operator*(float4 const &a, float4 const &b) noexcept {
    return {a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w};
}

__host__ __device__ static inline constexpr
float4 operator*(float4 const &a, float c) noexcept {
    return {a.x * c, a.y * c, a.z * c, a.w * c};
}

__host__ __device__ static inline constexpr
float4 operator / (float4 const &a, float c) noexcept {
    return {a.x / c, a.y / c, a.z / c, a.w / c};
}

__host__ __device__ static inline constexpr
float4 operator*(float c, float4 const &a) noexcept {
    return {a.x * c, a.y * c, a.z * c, a.w * c};
}

__host__ __device__ static inline constexpr
float4 &operator += (float4 &a, float4 const &b) noexcept {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
    return a;
}

__host__ __device__ static inline constexpr
float4 & operator *= (float4 & a, float4 b) noexcept {
    a.x *= b.x;
    a.y *= b.y;
    a.z *= b.z;
    a.w *= b.w;
    return a;
}

__host__ __device__ static inline constexpr
float4 &operator *= (float4 &a, float b) {
    a.x *= b;
    a.y *= b;
    a.z *= b;
    a.w *= b;
    return a;
}
