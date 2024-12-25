#pragma once

#if defined(_MSC_VER)
#define CHUNK_EXPORT __declspec(dllexport)
#else
#define CHUNK_EXPORT __attribute__((visibility("default")))
#endif

#include <type_traits>
#include <vector>

namespace chunk_processing {

// Type trait to check if T is a std::vector
template <typename T>
struct is_vector : std::false_type {};

template <typename T, typename A>
struct is_vector<std::vector<T, A>> : std::true_type {};

// Type trait to check if T is a multi-dimensional vector
template <typename T, typename = void>
struct is_multidimensional : std::false_type {};

// Specialization for types that have value_type and are vectors
template <typename T>
struct is_multidimensional<
    T, std::enable_if_t<is_vector<T>::value && is_vector<typename T::value_type>::value>>
    : std::true_type {};

// Helper variable template
template <typename T>
inline constexpr bool is_multidimensional_v = is_multidimensional<T>::value;

} // namespace chunk_processing