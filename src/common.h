#pragma once
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cfloat>

#include <algorithm>
#include <vector>
#include <memory>
#include <stdexcept>
#include <map>

#include <hip/hip_runtime.h>
#include <hiprand_kernel.h> 
#include <gsl/gsl_assert.h>
#include <nonstd/byte.hpp>
#include <nonstd/expected.hpp>
#include <nonstd/optional.hpp>
#include <nonstd/span.hpp>
#include <nonstd/ring_span.hpp>
#include <nonstd/string_view.hpp>
#include <nonstd/variant.hpp>
#include <nonstd/value_ptr.hpp>

#include "kernel_config.h"
#include "logger.h"
#include "math.h"

using nonstd::variant;
using nonstd::value_ptr;
using nonstd::make_value;
using nonstd::ring_span;
using nonstd::expected;
using nonstd::make_unexpected;
using nonstd::optional;
using nonstd::nullopt;
using nonstd::make_optional;
using nonstd::byte;
using nonstd::to_integer;
using nonstd::to_byte;
using nonstd::to_string;
using nonstd::string_view;
using nonstd::span;
using nonstd::make_span;
using bytearray = std::vector<byte>;

#define HIP_ASSERT(x) (Ensures((x)==hipSuccess))

struct hip_free_deleter {
    void operator()(void *p) const noexcept { hipFree(p); }
};

template<typename T>
using host_unique_ptr = std::unique_ptr<T, hip_free_deleter>;

template<class T> struct _host_unique_if {
    typedef void _single_object;
};

template<class T> struct _host_unique_if<T[]> {
    typedef host_unique_ptr<T[]> _unknown_bound;
};

template<class T, std::size_t N> struct _host_unique_if<T[N]> {
    typedef void _known_bound;
};

template<class T, class... Args>
typename _host_unique_if<T>::_single_object
hip_alloc_managed_unique(Args&&... args) = delete;

template<class T>
typename _host_unique_if<T>::_unknown_bound
hip_alloc_managed_unique(std::size_t n) {
    typedef typename std::remove_extent<T>::type U;
    T *tmp;
    HIP_ASSERT(hipMallocManaged(&tmp, n * sizeof(U)));
    return host_unique_ptr<T>(reinterpret_cast<U*>(tmp));
}

template<class T, class... Args>
typename _host_unique_if<T>::_known_bound
hip_alloc_managed_unique(Args&&...) = delete;