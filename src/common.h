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
