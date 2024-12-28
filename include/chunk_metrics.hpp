/**
 * @file chunk_metrics.hpp
 * @brief Quality metrics and analysis tools for chunk evaluation
 * @author Jonathan Reich
 * @date 2024-12-07
 */

#pragma once

#include "chunk_common.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace chunk_metrics {

namespace detail {
    template<typename T>
    bool is_valid_chunk(const std::vector<T>& chunk) {
        return !chunk.empty() && std::all_of(chunk.begin(), chunk.end(), 
            [](const T& val) { return std::isfinite(static_cast<double>(val)); });
    }

    template<typename T>
    double safe_mean(const std::vector<T>& data) {
        if (data.empty()) return 0.0;
        double sum = 0.0;
        size_t count = 0;
        
        for (const auto& val : data) {
            double d_val = static_cast<double>(val);
            if (std::isfinite(d_val)) {
                sum += d_val;
                ++count;
            }
        }
        return count > 0 ? sum / count : 0.0;
    }

    template<typename T>
    double safe_distance(const T& a, const T& b) {
        try {
            double d_a = static_cast<double>(a);
            double d_b = static_cast<double>(b);
            return std::isfinite(d_a) && std::isfinite(d_b) ? 
                   std::abs(d_a - d_b) : 
                   std::numeric_limits<double>::max();
        } catch (...) {
            return std::numeric_limits<double>::max();
        }
    }
}

template <typename T>
class CHUNK_EXPORT ChunkQualityAnalyzer {
private:
    double compute_chunk_cohesion(const std::vector<T>& chunk) const {
        if (chunk.size() < 2) return 0.0;

        std::vector<double> distances;
        distances.reserve((chunk.size() * (chunk.size() - 1)) / 2);

        for (size_t i = 0; i < chunk.size(); ++i) {
            for (size_t j = i + 1; j < chunk.size(); ++j) {
                double dist = detail::safe_distance(chunk[i], chunk[j]);
                if (dist < std::numeric_limits<double>::max()) {
                    distances.push_back(dist);
                }
            }
        }

        if (distances.empty()) return 0.0;
        std::sort(distances.begin(), distances.end());
        return distances[distances.size() / 2];  // Return median distance
    }

public:
    ChunkQualityAnalyzer() = default;
    ~ChunkQualityAnalyzer() = default;

    // Prevent copying/moving
    ChunkQualityAnalyzer(const ChunkQualityAnalyzer&) = delete;
    ChunkQualityAnalyzer& operator=(const ChunkQualityAnalyzer&) = delete;
    ChunkQualityAnalyzer(ChunkQualityAnalyzer&&) = delete;
    ChunkQualityAnalyzer& operator=(ChunkQualityAnalyzer&&) = delete;

    double compute_cohesion(const std::vector<std::vector<T>>& chunks) const {
        if (chunks.empty()) {
            throw std::invalid_argument("Empty chunks");
        }

        std::vector<double> cohesion_values;
        cohesion_values.reserve(chunks.size());

        for (const auto& chunk : chunks) {
            if (chunk.empty() || chunk.size() > 1000000) {
                throw std::invalid_argument("Invalid chunk size");
            }
            double chunk_cohesion = compute_chunk_cohesion(chunk);
            if (std::isfinite(chunk_cohesion)) {
                cohesion_values.push_back(chunk_cohesion);
            }
        }

        if (cohesion_values.empty()) {
            throw std::runtime_error("No valid cohesion values computed");
        }

        std::sort(cohesion_values.begin(), cohesion_values.end());
        return cohesion_values[cohesion_values.size() / 2];
    }

    bool compare_cohesion(const std::vector<std::vector<T>>& well_separated,
                         const std::vector<std::vector<T>>& mixed,
                         double& high_result,
                         double& mixed_result) const {
        try {
            if (well_separated.empty() || mixed.empty()) {
                return false;
            }

            high_result = compute_cohesion(well_separated);
            mixed_result = compute_cohesion(mixed);

            return std::isfinite(high_result) && 
                   std::isfinite(mixed_result) && 
                   high_result > mixed_result;
        } catch (...) {
            return false;
        }
    }

    double compute_separation(const std::vector<std::vector<T>>& chunks) const {
        if (chunks.size() < 2) {
            throw std::invalid_argument("Need at least two chunks for separation");
        }

        double total_separation = 0.0;
        size_t valid_pairs = 0;

        for (size_t i = 0; i < chunks.size(); ++i) {
            for (size_t j = i + 1; j < chunks.size(); ++j) {
                if (chunks[i].empty() || chunks[j].empty()) continue;

                double mean_i = detail::safe_mean(chunks[i]);
                double mean_j = detail::safe_mean(chunks[j]);

                if (std::isfinite(mean_i) && std::isfinite(mean_j)) {
                    total_separation += std::abs(mean_i - mean_j);
                    ++valid_pairs;
                }
            }
        }

        if (valid_pairs == 0) {
            throw std::runtime_error("No valid separation values computed");
        }

        return total_separation / valid_pairs;
    }

    double compute_silhouette_score(const std::vector<std::vector<T>>& chunks) const {
        if (chunks.size() < 2) {
            throw std::invalid_argument("Need at least two chunks for silhouette score");
        }

        double total_score = 0.0;
        size_t total_points = 0;

        for (size_t i = 0; i < chunks.size(); ++i) {
            for (const auto& point : chunks[i]) {
                // Calculate a (average distance to points in same chunk)
                double a = 0.0;
                size_t same_chunk_count = 0;
                for (const auto& other_point : chunks[i]) {
                    if (&point != &other_point) {
                        double dist = detail::safe_distance(point, other_point);
                        if (dist < std::numeric_limits<double>::max()) {
                            a += dist;
                            ++same_chunk_count;
                        }
                    }
                }
                a = same_chunk_count > 0 ? a / same_chunk_count : 0.0;

                // Calculate b (minimum average distance to other chunks)
                double b = std::numeric_limits<double>::max();
                for (size_t j = 0; j < chunks.size(); ++j) {
                    if (i != j) {
                        double avg_dist = 0.0;
                        size_t valid_dist = 0;
                        for (const auto& other_point : chunks[j]) {
                            double dist = detail::safe_distance(point, other_point);
                            if (dist < std::numeric_limits<double>::max()) {
                                avg_dist += dist;
                                ++valid_dist;
                            }
                        }
                        if (valid_dist > 0) {
                            b = std::min(b, avg_dist / valid_dist);
                        }
                    }
                }

                if (std::isfinite(a) && std::isfinite(b) && b < std::numeric_limits<double>::max()) {
                    double max_ab = std::max(a, b);
                    if (max_ab > 0) {
                        total_score += (b - a) / max_ab;
                        ++total_points;
                    }
                }
            }
        }

        if (total_points == 0) {
            throw std::runtime_error("No valid silhouette scores computed");
        }

        return total_score / total_points;
    }

    double compute_quality_score(const std::vector<std::vector<T>>& chunks) const {
        if (chunks.empty()) {
            throw std::invalid_argument("Empty chunks vector");
        }

        try {
            double cohesion = compute_cohesion(chunks);
            double separation = chunks.size() > 1 ? compute_separation(chunks) : 1.0;

            if (!std::isfinite(cohesion) || !std::isfinite(separation)) {
                throw std::runtime_error("Invalid metric values computed");
            }

            return (cohesion + separation) / 2.0;
        } catch (const std::exception& e) {
            throw std::runtime_error(std::string("Error computing quality score: ") + e.what());
        }
    }

    std::map<std::string, double> compute_size_metrics(const std::vector<std::vector<T>>& chunks) const {
        if (chunks.empty()) {
            throw std::invalid_argument("Empty chunks vector");
        }

        std::map<std::string, double> metrics;
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

    void clear_cache() const {
        // No-op in single-threaded version
    }
};

} // namespace chunk_metrics