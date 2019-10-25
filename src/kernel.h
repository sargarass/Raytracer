#pragma once

#include "common.h"
#include "ray.h"
#include "triangle.h"

enum class object_type {
    polygonal,
    sphere
};

struct object {
    object_type type;
    float3 *mesh = nullptr;
    size_t mesh_size;
};

namespace kernel {
    struct hit_record {
        float t;
        uint32_t object_id;
    };

    struct global_data {
        constexpr global_data() noexcept {}

        constexpr global_data(float3 *framebuffer, uint32_t width, uint32_t height, object *objects,
                              uint32_t objects_count) noexcept
                : framebuffer { framebuffer },
                  width { width },
                  height { height },
                  objects { objects },
                  objects_count { objects_count } {}

        float3 *const framebuffer = nullptr;
        uint32_t const width = 0;
        uint32_t const height = 0;
        object const *const objects = nullptr;
        uint32_t const objects_count = 0;
    };

    extern __constant__ global_data
    kg;

    __global__ void setupRender();

    __global__ void render();

}