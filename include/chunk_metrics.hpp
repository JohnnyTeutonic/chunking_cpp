/**
 * @file chunk_metrics.hpp
 * @brief Quality metrics and analysis tools for chunk evaluation
 * @author Jonathan Reich
 * @date 2024-12-07
 */

#pragma once

#include "chunk_common.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace chunk_metrics {

namespace detail {
template <typename T>
bool is_valid_chunk(const std::vector<T>& chunk) {
    return !chunk.empty() && std::all_of(chunk.begin(), chunk.end(), [](const T& val) {
        return std::isfinite(static_cast<double>(val));
    });
}

template <typename T>
bool is_valid_chunks(const std::vector<std::vector<T>>& chunks) {
    return !chunks.empty() && std::all_of(chunks.begin(), chunks.end(),
                                          [](const auto& chunk) { return is_valid_chunk(chunk); });
}

template <typename T>
double safe_mean(const std::vector<T>& data) {
    if (data.empty())
        return 0.0;
    double sum = 0.0;
    size_t valid_count = 0;

    for (const auto& val : data) {
        double d_val = static_cast<double>(val);
        if (std::isfinite(d_val)) {
            sum += d_val;
            ++valid_count;
        }
    }

    return valid_count > 0 ? sum / valid_count : 0.0;
}

template <typename T>
double safe_distance(const T& a, const T& b) {
    try {
        double d_a = static_cast<double>(a);
        double d_b = static_cast<double>(b);
        if (!std::isfinite(d_a) || !std::isfinite(d_b)) {
            return std::numeric_limits<double>::max();
        }
        return std::abs(d_a - d_b);
    } catch (...) {
        return std::numeric_limits<double>::max();
    }
}
} // namespace detail

/**
 * @brief Class for analyzing and evaluating chunk quality
 * @tparam T The data type of the chunks (must support arithmetic operations)
 */
template <typename T>
class CHUNK_EXPORT ChunkQualityAnalyzer {
private:
    mutable std::mutex mutex_;
    mutable std::recursive_mutex compute_mutex_;
    mutable std::atomic<bool> is_processing_{false};
    mutable std::atomic<bool> is_computing_quality_{false};
    mutable std::atomic<size_t> computation_depth_{0};

    // Add a simple reset flag instead of complex cache
    mutable std::atomic<bool> needs_reset_{false};

    struct ComputationGuard {
        std::atomic<size_t>& depth;
        explicit ComputationGuard(std::atomic<size_t>& d) : depth(d) { ++depth; }
        ~ComputationGuard() { --depth; }
    };

    template<typename Func>
    auto compute_safely(Func&& func) const -> decltype(func()) {
        if (computation_depth_ > 10) {
            throw std::runtime_error("Computation depth exceeded");
        }
        ComputationGuard guard(computation_depth_);
        std::lock_guard<std::recursive_mutex> lock(compute_mutex_);
        return func();
    }

    double compute_chunk_cohesion(const std::vector<T>& chunk, double chunk_mean) const {
        return compute_safely([&]() {
            if (chunk.empty()) return 0.0;

            std::vector<double> distances;
            distances.reserve((chunk.size() * (chunk.size() - 1)) / 2);

            for (size_t i = 0; i < chunk.size(); ++i) {
                for (size_t j = i + 1; j < chunk.size(); ++j) {
                    try {
                        double dist = detail::safe_distance(chunk[i], chunk[j]);
                        if (dist < std::numeric_limits<double>::max()) {
                            distances.push_back(dist);
                        }
                    } catch (...) {
                        continue;
                    }
                }
            }

            if (distances.empty()) return 0.0;

            std::sort(distances.begin(), distances.end());
            if (distances.size() % 2 == 0) {
                return (distances[distances.size()/2 - 1] + distances[distances.size()/2]) / 2.0;
            }
            return distances[distances.size()/2];
        });
    }

public:
    ChunkQualityAnalyzer() = default;

    /**
     * @brief Reset internal state
     */
    void clear_cache() {
        std::lock_guard<std::mutex> lock(mutex_);
        needs_reset_ = true;
        computation_depth_ = 0;
        is_processing_ = false;
        is_computing_quality_ = false;
    }

    double compute_cohesion(const std::vector<std::vector<T>>& chunks) const {
        return compute_safely([&]() {
            try {
                std::unique_lock<std::mutex> lock(mutex_);

                if (chunks.empty()) {
                    throw std::invalid_argument("Empty chunks vector");
                }

                for (const auto& chunk : chunks) {
                    if (!detail::is_valid_chunk(chunk)) {
                        throw std::invalid_argument("Invalid chunk detected");
                    }
                }

                std::vector<double> means;
                means.reserve(chunks.size());
                for (const auto& chunk : chunks) {
                    means.push_back(detail::safe_mean(chunk));
                }

                double total_cohesion = 0.0;
                size_t valid_chunks = 0;

                for (size_t i = 0; i < chunks.size(); ++i) {
                    try {
                        if (!chunks[i].empty()) {
                            double chunk_cohesion = compute_chunk_cohesion(chunks[i], means[i]);
                            if (std::isfinite(chunk_cohesion)) {
                                total_cohesion += chunk_cohesion;
                                ++valid_chunks;
                            }
                        }
                    } catch (...) {
                        continue;
                    }
                }

                if (valid_chunks == 0) {
                    throw std::runtime_error("No valid chunks for cohesion calculation");
                }

                double result = total_cohesion / valid_chunks;
                if (!std::isfinite(result)) {
                    throw std::runtime_error("Invalid cohesion result computed");
                }

                return result;

            } catch (const std::exception& e) {
                throw std::runtime_error(std::string("Cohesion computation failed: ") + e.what());
            } catch (...) {
                throw std::runtime_error("Unknown error in cohesion computation");
            }
        });
    }

    /**
     * @brief Calculate separation (dissimilarity between chunks)
     * @param chunks Vector of chunk data
     * @return Separation score between 0 and 1
     * @throws std::invalid_argument if chunks is empty or contains single chunk
     */
    double compute_separation(const std::vector<std::vector<T>>& chunks) const {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (chunks.size() < 2) {
            throw std::invalid_argument("Need at least two chunks for separation");
        }

        try {
            double total_separation = 0.0;
            size_t comparisons = 0;

            for (size_t i = 0; i < chunks.size(); ++i) {
                for (size_t j = i + 1; j < chunks.size(); ++j) {
                    if (chunks[i].empty() || chunks[j].empty()) {
                        continue;
                    }

                    double mean_i = detail::safe_mean(chunks[i]);
                    double mean_j = detail::safe_mean(chunks[j]);

                    if (std::isfinite(mean_i) && std::isfinite(mean_j)) {
                        double separation = std::abs(mean_i - mean_j);
                        total_separation += separation;
                        ++comparisons;
                    }
                }
            }

            return comparisons > 0 ? total_separation / comparisons : 0.0;

        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Error computing separation: ") + e.what());
        }
    }

    /**
     * @brief Calculate silhouette score for chunk validation
     * @param chunks Vector of chunk data
     * @return Silhouette score between -1 and 1
     * @throws std::invalid_argument if chunks is empty or contains single chunk
     */
    double compute_silhouette_score(const std::vector<std::vector<T>>& chunks) {
        if (chunks.size() < 2) {
            throw std::invalid_argument("Need at least two chunks for silhouette score");
        }

        double total_score = 0.0;
        size_t total_points = 0;

        for (size_t i = 0; i < chunks.size(); ++i) {
            for (const auto& point : chunks[i]) {
                // Calculate a (average distance to points in same chunk)
                double a = 0.0;
                for (const auto& other_point : chunks[i]) {
                    if (&point != &other_point) {
                        a += std::abs(point - other_point);
                    }
                }
                a = chunks[i].size() > 1 ? a / (chunks[i].size() - 1) : 0;

                // Calculate b (minimum average distance to points in other chunks)
                double b = std::numeric_limits<double>::max();
                for (size_t j = 0; j < chunks.size(); ++j) {
                    if (i != j) {
                        double avg_dist = 0.0;
                        for (const auto& other_point : chunks[j]) {
                            avg_dist += std::abs(point - other_point);
                        }
                        avg_dist /= chunks[j].size();
                        b = std::min(b, avg_dist);
                    }
                }

                // Calculate silhouette score for this point
                double max_ab = std::max(a, b);
                if (max_ab > 0) {
                    total_score += (b - a) / max_ab;
                }
                ++total_points;
            }
        }

        return total_score / total_points;
    }

    /**
     * @brief Calculate overall quality score combining multiple metrics
     * @param chunks Vector of chunk data
     * @return Quality score between 0 and 1
     * @throws std::invalid_argument if chunks is empty
     */
    double compute_quality_score(const std::vector<std::vector<T>>& chunks) const {
        // Prevent reentrant calls and ensure thread safety
        bool expected = false;
        if (!is_computing_quality_.compare_exchange_strong(expected, true)) {
            throw std::runtime_error("Quality score computation already in progress");
        }

        struct QualityGuard {
            std::atomic<bool>& flag;
            QualityGuard(std::atomic<bool>& f) : flag(f) {}
            ~QualityGuard() { flag = false; }
        } guard(const_cast<std::atomic<bool>&>(is_computing_quality_));

        try {
            std::unique_lock<std::mutex> lock(mutex_);
            
            if (chunks.empty()) {
                throw std::invalid_argument("Empty chunks vector");
            }

            // Store cohesion result safely
            double cohesion = 0.0;
            {
                // Compute cohesion with its own lock scope
                cohesion = compute_cohesion(chunks);
            }

            // Store separation result safely
            double separation = 0.0;
            if (chunks.size() > 1) {
                // Compute separation with its own lock scope
                try {
                    separation = compute_separation(chunks);
                } catch (const std::exception&) {
                    separation = 1.0;  // Fallback for single chunk
                }
            } else {
                separation = 1.0;  // Default for single chunk
            }

            // Validate results before computation
            if (!std::isfinite(cohesion) || !std::isfinite(separation)) {
                throw std::runtime_error("Invalid metric values computed");
            }

            return (cohesion + separation) / 2.0;

        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Error computing quality score: ") + e.what());
        } catch (...) {
            throw std::runtime_error("Unknown error in quality score computation");
        }
    }

    /**
     * @brief Compute size-based metrics for chunks
     * @param chunks Vector of chunk data
     * @return Map of metric names to values
     */
    std::map<std::string, double>
    compute_size_metrics(const std::vector<std::vector<T>>& chunks) const {
        std::map<std::string, double> metrics;

        if (chunks.empty()) {
            throw std::invalid_argument("Empty chunks vector");
        }

        // Calculate average chunk size
        double avg_size = 0.0;
        double max_size = 0.0;
        double min_size = static_cast<double>(chunks[0].size());

        for (const auto& chunk : chunks) {
            double size = static_cast<double>(chunk.size());
            avg_size += size;
            max_size = std::max(max_size, size);
            min_size = std::min(min_size, size);
        }
        avg_size /= static_cast<double>(chunks.size());

        // Calculate size variance
        double variance = 0.0;
        for (const auto& chunk : chunks) {
            double diff = static_cast<double>(chunk.size()) - avg_size;
            variance += diff * diff;
        }
        variance /= static_cast<double>(chunks.size());

        metrics["average_size"] = avg_size;
        metrics["max_size"] = max_size;
        metrics["min_size"] = min_size;
        metrics["size_variance"] = variance;
        metrics["size_stddev"] = std::sqrt(variance);

        return metrics;
    }

private:
    /**
     * @brief Calculate mean value of a chunk
     * @param chunk Single chunk data
     * @return Mean value of the chunk
     */
    T calculate_mean(const std::vector<T>& chunk) {
        if (chunk.empty()) {
            return T{};
        }
        T sum = T{};
        for (const auto& val : chunk) {
            sum += val;
        }
        return sum / static_cast<T>(chunk.size());
    }

    /**
     * @brief Calculate variance of a chunk
     * @param chunk Single chunk data
     * @param mean Pre-calculated mean value
     * @return Variance of the chunk
     */
    T calculate_variance(const std::vector<T>& chunk, T mean) {
        if (chunk.size() < 2) {
            return T{};
        }
        T sum_sq_diff = T{};
        for (const auto& val : chunk) {
            T diff = val - mean;
            sum_sq_diff += diff * diff;
        }
        return sum_sq_diff / static_cast<T>(chunk.size() - 1);
    }
};

} // namespace chunk_metrics