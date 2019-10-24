#pragma once
#include "nonstd/string_view.hpp"
#include "constexpr_string.h"
using nonstd::string_view;

enum class Log {
    General,
    Warning,
    Error,
    Alert,
    Debug,
};

namespace detail {  
constexpr string_view logGetColor(Log type) {
    switch (type) {
        case Log::General: return "\e[m";
        case Log::Warning: return "\e[33m";
        case Log::Error: return "\e[31m";
        case Log::Alert: return "\e[41m";
        case Log::Debug: return "\e[2m";
    }
    return "\e[m";
}
    
    
constexpr string_view trimFilePath(const char *filename) {
        size_t length = 0;
        for (; *filename; ++length, ++filename) {}
        
        for (; length > 0 && *filename != '/'; --length, --filename) {}
        
        if (!length) {
            return filename;
        }
        
        if (*(filename-1) == '.') {
            return filename + 1;
        }
        
        for (--filename, --length; length > 0 && *filename != '/'; --length, --filename) {}
        return (!length)? filename : filename + 1;
    }

    template<typename...Args>
    static inline void writeLog_(string_view fmt, Args &&... args) {
        time_t t;
        tm now;
        char time_str[40];
        std::time(&t);
        localtime_r(&t, &now);
        std::strftime(time_str, sizeof(time_str), "%Y-%m-%dT%H:%M:%S", &now);
        printf(fmt.data(), time_str , std::forward<Args>(args)...);
    }
}

#define DETAIL_STRINGIZE(x) DETAIL_STRINGIZE2(x)
#define DETAIL_STRINGIZE2(x) #x
#define LINE_STRING DETAIL_STRINGIZE(__LINE__)
#ifndef __HIP_DEVICE_COMPILE__
#define DETAIL_WRITE_LOG(_filename, _line, _type, s_type, _fmt, ...) do {\
    static_assert(std::is_convertible<decltype(_filename), const char*>(), "_filename should be convertable to const char*"); \
    static_assert(std::is_convertible<decltype(_line), const char*>(), "_line should be convertable to const char*"); \
    static_assert(std::is_convertible<decltype(_type), Log>(), "_type should be convertable to Log"); \
    static_assert(std::is_convertible<decltype(s_type), const char*>(), "s_type should be convertable to const char*"); \
    static_assert(std::is_convertible<decltype(_fmt), const char*>(), "_fmt should be convertable to const char*"); \
    static constexpr string_view ____LOG_view_filename = detail::trimFilePath(_filename);\
    static constexpr auto ____LOG_color = detail::logGetColor(_type); \
    static constexpr auto ____LOG_fmt_ = make_constexpr_string<____LOG_color.size()>(____LOG_color.data()) + "HOST: %s " + make_constexpr_string(s_type) + ": " + make_constexpr_string<____LOG_view_filename.size()>(____LOG_view_filename.data()) + ":" + _line + ": " + _fmt + "\e[m" "\n"; \
    static constexpr string_view ____LOG_fmt {____LOG_fmt_.data(), ____LOG_fmt_.length()};\
    detail::writeLog_(____LOG_fmt ,##__VA_ARGS__);\
} while(0)
#else
#define DETAIL_WRITE_LOG(_filename, _line, _type, s_type, _fmt, ...) do {\
    static constexpr string_view ____LOG_view_filename = detail::trimFilePath(_filename);\
    static constexpr auto ____LOG_color = detail::logGetColor(_type); \
    static constexpr auto ____LOG_fmt_ = make_constexpr_string<____LOG_color.size()>(____LOG_color.data()) + "DEVIDE: " + make_constexpr_string(s_type) + ": " + make_constexpr_string<____LOG_view_filename.size()>(____LOG_view_filename.data()) + ":" + _line + ": " + _fmt + "\e[m" "\n"; \
    printf(____LOG_fmt_.data() ,##__VA_ARGS__); \
} while(0)
#endif
#define writeLog(type, fmt, ...) DETAIL_WRITE_LOG(__FILE__, LINE_STRING, Log::type, #type, fmt,##__VA_ARGS__)

