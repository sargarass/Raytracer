#include "kernel.h"

namespace kernel {
    __constant__ global_data kg;

    __device__
    bool hit_sphere(float3 const &center, float const radius, core::ray const &ray,
                    float const t_min, float const t_max, float &t) noexcept {
        float3 oc = ray.origin() - center;
        float a = dot(ray.direction(), ray.direction());
        float b = dot(oc, ray.direction());
        float c = dot(oc, oc) - radius * radius;
        float discriminant = b * b - a * c;
        if (discriminant >= 0.0f) {
            float sqrt_discriminant = sqrtf(discriminant);
            float t1 = (-b - sqrt_discriminant) / a;
            if (t1 > t_min && t1 < t_max) {
                t = t1;
                return true;
            }

            float t2 = (-b + sqrt_discriminant) / a;
            if (t2 > t_min && t2 < t_max) {
                t = t2;
                return true;
            }
        }
        return false;
    }

    __device__
    bool hit_triangle(core::ray const &ray, primitives::triangle const &triangle,
                      float const t_min, float const t_max, float &t, float &u, float &v) {
        float3 origin = ray.origin();
        float3 direction = ray.direction();

        const float3 v0 = triangle.a - origin;
        const float3 v1 = triangle.b - origin;
        const float3 v2 = triangle.c - origin;

        const float3 e0 = v2 - v0;
        const float3 e1 = v0 - v1;
        const float3 e2 = v1 - v2;

        const float U = dot(cross(v2 + v0, e0), direction);
        const float V = dot(cross(v0 + v1, e1), direction);
        const float W = dot(cross(v1 + v2, e2), direction);
        const float minUVW = min(U, min(V, W));
        const float maxUVW = max(U, max(V, W));

        if(minUVW < 0.0f && maxUVW > 0.0f) {
            return false;
        }

        /* Calculate geometry normal and denominator. */
        const float3 Ng1 = cross(e1, e0);

        const float3 Ng = Ng1 + Ng1;
        const float den = dot(Ng, direction);

        if (den == 0.0f) {
            return false;
        }

        const float T = dot(v0, Ng);
        const int sign_den = (float_as_int(den) & 0x80000000);
        const float sign_T = xor_signmask(T, sign_den);
        if (sign_T < 0.0f) {
            return false;
        }
        const float inv_den = 1.0f / den;
        t = T * inv_den;
        if (t > t_min && t < t_max) {
            u = U * inv_den;
            v = V * inv_den;
            return true;
        }
        return false;
    }

    __device__
    hit_record scene_intersect(core::ray const &ray) {
        hit_record result { FLT_MAX, 0 };
        constexpr float t_min = 1.0e-7f;
        for (uint32_t i = 0; i < kg.objects_count; ++i) {
            switch (kg.objects[i].type) {
                case object_type::sphere: {
                    float3 origin = kg.objects[i].mesh[0];
                    float radius = kg.objects[i].mesh[1].x;
                    float t;
                    if (hit_sphere(origin, radius, ray, t_min, result.t, t)) {
                        result.t = t;
                        result.object_id = i;
                    }
                    break;
                }
                case object_type::polygonal: {
                    auto mesh_size = kg.objects[i].mesh_size / 3;
                    for (int j = 0; j < mesh_size; ++j) {
                        float3 a = kg.objects[i].mesh[j * 3 + 0];
                        float3 b = kg.objects[i].mesh[j * 3 + 1];
                        float3 c = kg.objects[i].mesh[j * 3 + 2];
                        primitives::triangle tri { a, b, c};
                        float t, u, v;
                        if (hit_triangle(ray, tri, t_min, result.t, t, u, v)) {
                            result.t = t;
                            result.object_id = i;
                        }
                    }
                }
            }
        }
        return result;
    }

    __device__
    float3 color(core::ray const &ray, hit_record const &hit) {
        if (hit.t < FLT_MAX) {
            switch (kg.objects[hit.object_id].type) {
                case object_type::sphere: {
                    float3 origin = kg.objects[hit.object_id].mesh[0];
                    auto normal = normalize(ray.pointAtParameter(hit.t) - origin);
                    return 0.5f * float3 { normal.x + 1.0f, normal.y + 1.0f, normal.z + 1.0f };
                }
                case object_type::polygonal: {
                    return float3{1.0f, 0.0f, 0.0f};
                }
            }
        }
        float3 unitDirection = normalize(ray.direction());
        float t = 0.5f * (unitDirection.y + 1.0f);
        return (1.0f - t) * float3 {1.0f, 1.0f, 1.0f} + t * float3{0.5, 0.7, 1.0};
    }


    __global__
    void setupRender() {
        int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
        if ((x >= kg.width) || (y >= kg.height)) {
            return;
        }

        int pixel_index = y * kg.width + x;
        kg.framebuffer[pixel_index].x = 1.0f;
        kg.framebuffer[pixel_index].y = 1.0f;
        kg.framebuffer[pixel_index].z = 1.0f;
    }

    __global__
    void render() {
        constexpr float3 lower_left_corner { -2.0f, -0.857f, -1.0f };
        constexpr float3 horizontal { 4.0f, 0.0f, 0.0f };
        constexpr float3 vertical { 0.0f, 1.714, 0.0f };
        constexpr float3 origin { 0.0, 0.0, 0.0 };

        int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
        int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;
        if ((x >= kg.width) || (y >= kg.height)) {
            return;
        }

        int pixel_index = y * kg.width + x;
        float u = static_cast<float>(x) / kg.width;
        float v = static_cast<float>(y) / kg.height;
        core::ray ray(origin, lower_left_corner + u * horizontal + v * vertical);

        auto hit = scene_intersect(ray);
        kg.framebuffer[pixel_index] = color(ray, hit);
    }
}