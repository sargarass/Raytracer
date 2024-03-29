#pragma once

#include "common.h"
#include "math.h"

namespace primitives {
    class triangle {
    public:
        constexpr __host__ __device__
        triangle() noexcept
                : a {0.0f, 0.0f, 0.0f},
                  b {0.0f, 0.0f, 0.0f},
                  c {0.0f, 0.0f, 0.0f}
        {}

        constexpr __host__ __device__
        triangle(float3 const &a, float3 const &b, float3 const &c) noexcept
                : a { a }, b { b }, c { c }
        {}

        constexpr __host__ __device__
        float3 &operator[](size_t idx) noexcept {
            Expects(idx < 3);
            return packed[idx];
        }

        constexpr __host__ __device__
        float3 *begin() noexcept {
            return &packed[0];
        }

        constexpr __host__ __device__
        float3 *end() noexcept {
            return &packed[3];
        }

        union {
            float3 packed[3];
            struct {
                float3 a, b, c;
            };
        };
    };
}