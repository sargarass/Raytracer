#pragma once
#include <utility>

template<std::size_t size>
struct constexpr_string {
    template <typename... Characters>
    constexpr constexpr_string( Characters... characters )
        : m_data{ characters..., '\0' }
    {}
    
    template <std::size_t... Indexes>
    explicit constexpr constexpr_string(char const variable[size  + 1], std::index_sequence<Indexes...>) 
        : m_data{ variable[Indexes]..., '\0' }
    {}
    
    constexpr constexpr_string(char const variable[size + 1]) 
        : constexpr_string(variable, std::make_index_sequence<size>{})
    {}
        
    constexpr size_t length() const {
        return size;
    }
    
    constexpr const char *data() const {
        return m_data;
    }
    
private:
    const char m_data[size + 1];
};

template<std::size_t... IndexesLhs, std::size_t... IndexesRhs>
constexpr auto
mergeConstexprString(char const *lhs,
                     std::index_sequence<IndexesLhs...>,
                     char const *rhs,
                     std::index_sequence<IndexesRhs...>) {
    return constexpr_string<sizeof...(IndexesLhs) + sizeof...(IndexesRhs)>{lhs[IndexesLhs]..., rhs[IndexesRhs]... };
}

template<std::size_t lhsSize, std::size_t rhsSize>
constexpr auto operator+(char const (&lhs)[lhsSize], constexpr_string<rhsSize> rhs) {
    return mergeConstexprString(lhs, std::make_index_sequence<lhsSize - 1>{}, rhs.data(), std::make_index_sequence<rhsSize>{});
}

template<std::size_t lhsSize, std::size_t rhsSize>
constexpr auto operator+(constexpr_string<lhsSize> lhs, char const (&rhs)[rhsSize]) {
    return mergeConstexprString(lhs.data(), std::make_index_sequence<lhsSize>{}, rhs, std::make_index_sequence<rhsSize - 1>{});
}

template<std::size_t lhsSize, std::size_t rhsSize>
constexpr auto operator+(constexpr_string<lhsSize> lhs, constexpr_string<rhsSize> rhs) {
    return mergeConstexprString(lhs.data(), std::make_index_sequence<lhsSize>{}, rhs.data(), std::make_index_sequence<rhsSize>{});
}


template <std::size_t N>
constexpr auto make_constexpr_string(const char (&variable)[N]) {
    return constexpr_string<N - 1>{ variable };
}

template <std::size_t N>
constexpr auto make_constexpr_string(const char *variable) {
    return constexpr_string<N>{ variable };
}
