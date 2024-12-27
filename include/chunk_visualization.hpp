/**
 * @file chunk_visualization.hpp
 * @brief Visualization utilities for chunk data
 * @author Jonathan Reich
 * @date 2024-12-07
 */

#pragma once
#include "chunk_common.hpp"
#include "chunk_errors.hpp"
#include <algorithm> // for std::abs
#include <cmath>     // Add this for std::sqrt
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace chunk_viz {

/**
 * @brief Class for visualizing chunk data in various formats
 * @tparam T The data type of the chunks (must be arithmetic)
 */
template <typename T>
class CHUNK_EXPORT ChunkVisualizer {
private:
    std::vector<T> data;
    std::string output_dir;
    static constexpr double threshold = 1.0; // Default threshold for chunk detection

    // Add helper to calculate difference between elements
    double calculate_difference(const T& current, const T& previous) const {
        if constexpr (std::is_arithmetic_v<T>) {
            return std::abs(current - previous);
        } else if constexpr (std::is_same_v<T, std::vector<double>> ||
                             std::is_same_v<T, std::vector<float>> ||
                             std::is_same_v<T, std::vector<int>>) {
            // For vectors, calculate Euclidean distance
            if (current.size() != previous.size()) {
                return std::numeric_limits<double>::max();
            }
            double sum = 0.0;
            for (size_t i = 0; i < current.size(); ++i) {
                double diff = current[i] - previous[i];
                sum += diff * diff;
            }
            return std::sqrt(sum);
        } else {
            throw chunk_processing::VisualizationError(
                "Unsupported type for difference calculation");
        }
    }

    // Add back format_value with improved implementation
    std::string format_value(const T& value) const {
        if constexpr (std::is_arithmetic_v<T>) {
            return std::to_string(value);
        } else if constexpr (std::is_same_v<T, std::string>) {
            return value;
        } else if constexpr (std::is_same_v<T, std::vector<double>> ||
                             std::is_same_v<T, std::vector<float>> ||
                             std::is_same_v<T, std::vector<int>>) {
            std::stringstream ss;
            ss << "[";
            for (size_t i = 0; i < value.size(); ++i) {
                ss << value[i];
                if (i < value.size() - 1)
                    ss << ",";
            }
            ss << "]";
            return ss.str();
        } else {
            throw chunk_processing::VisualizationError("Unsupported type for visualization");
        }
    }

    // Add helper to ensure data is written
    void write_data_file(const std::string& filename,
                         const std::vector<std::pair<size_t, size_t>>& chunk_sizes) {
        std::ofstream file(filename);
        if (!file) {
            throw chunk_processing::ChunkingError("Failed to create data file: " + filename);
        }

        // Write header
        file << "# Chunk_Index Size\n";

        // Write data with explicit flush
        for (size_t i = 0; i < chunk_sizes.size(); ++i) {
            file << i << " " << chunk_sizes[i].second << "\n";
            file.flush(); // Ensure data is written
        }

        file.close();

        // Verify file was written
        std::ifstream check(filename);
        if (!check || check.peek() == std::ifstream::traits_type::eof()) {
            throw chunk_processing::ChunkingError("Failed to write data to file: " + filename);
        }
    }

public:
    // Update constructor to handle both single vector and vector of vectors
    ChunkVisualizer(const std::vector<T>& input_data, const std::string& viz_dir = "./viz")
        : data(input_data), output_dir(viz_dir) {
        std::filesystem::create_directories(output_dir);
    }

    // Add overloaded constructor for vector of vectors
    template <typename U = T>
    ChunkVisualizer(const std::vector<std::vector<U>>& input_data,
                    const std::string& viz_dir = "./viz")
        : data(reinterpret_cast<const std::vector<T>&>(input_data)), output_dir(viz_dir) {
        std::filesystem::create_directories(output_dir);
        static_assert(std::is_same_v<T, std::vector<U>>,
                      "Template parameter must match vector element type");
    }

    void plot_chunk_sizes() {
        // Calculate chunk sizes
        std::vector<std::pair<size_t, size_t>> chunk_sizes;
        size_t current_size = 1; // Initialize to 1 since we start from first element

        // Handle empty data case
        if (data.empty()) {
            throw chunk_processing::ChunkingError("Cannot plot chunk sizes for empty data");
        }

        // Handle single element case
        if (data.size() == 1) {
            chunk_sizes.push_back({0, 1});
        } else {
            // Process chunks
            for (size_t i = 1; i < data.size(); ++i) {
                if (calculate_difference(data[i], data[i - 1]) > threshold) {
                    chunk_sizes.push_back({i - current_size, current_size});
                    current_size = 1; // Reset to 1 since we're starting a new chunk
                } else {
                    current_size++;
                }
            }

            // Add final chunk
            if (current_size > 0) {
                chunk_sizes.push_back({data.size() - current_size, current_size});
            }
        }

        // Write data file
        std::string dat_file = output_dir + "/chunk_sizes.dat";
        write_data_file(dat_file, chunk_sizes);

        // Create gnuplot script
        std::string gnu_file = output_dir + "/plot_chunks.gnu";
        std::ofstream gnuplot(gnu_file);
        if (!gnuplot) {
            throw chunk_processing::ChunkingError("Failed to create gnuplot file");
        }

        gnuplot << "set title 'Chunk Size Distribution'\n"
                << "set xlabel 'Chunk Index'\n"
                << "set ylabel 'Size'\n"
                << "set style fill solid\n"
                << "plot '" << dat_file << "' using 1:2 with boxes title 'Chunk Sizes'\n";

        gnuplot.close();
    }

    void export_to_graphviz(const std::string& filename = "chunks.dot") {
        // Ensure the output directory exists
        std::filesystem::create_directories(output_dir);
        
        // Create full path
        std::string actual_filename;
        if (filename.find('/') != std::string::npos) {
            // If filename contains a path, use it as is
            actual_filename = filename;
        } else {
            // Otherwise, append to output_dir
            actual_filename = output_dir + "/" + filename;
        }
        
        // Create the file
        std::ofstream file(actual_filename);
        if (!file.is_open()) {
            throw chunk_processing::VisualizationError(
                "Failed to create GraphViz file: " + actual_filename);
        }

        try {
            file << "digraph chunks {\n";
            for (size_t i = 0; i < data.size(); ++i) {
                file << "  chunk" << i << " [label=\"Value: " << format_value(data[i]) << "\"];\n";
                if (i > 0) {
                    file << "  chunk" << (i - 1) << " -> chunk" << i << ";\n";
                }
            }
            file << "}\n";
            
            // Ensure everything is written
            file.flush();
            
            if (file.fail()) {
                throw chunk_processing::VisualizationError(
                    "Failed to write to GraphViz file: " + actual_filename);
            }
        } catch (const std::exception& e) {
            throw chunk_processing::VisualizationError(
                std::string("Error writing GraphViz file: ") + e.what());
        }
        
        file.close();
        
        // Verify the file was created and has content
        if (!std::filesystem::exists(actual_filename) || 
            std::filesystem::file_size(actual_filename) == 0) {
            throw chunk_processing::VisualizationError(
                "GraphViz file was not created properly: " + actual_filename);
        }
    }

    void visualize_boundaries() {
        std::ofstream boundary_data(output_dir + "/boundaries.dat");
        if (!boundary_data) {
            throw chunk_processing::VisualizationError("Failed to create boundaries.dat");
        }

        // Create gnuplot script for boundary visualization
        std::ofstream gnuplot_script(output_dir + "/plot_boundaries.gnu");
        if (!gnuplot_script) {
            throw chunk_processing::VisualizationError("Failed to create gnuplot script");
        }

        // Write data points and mark boundaries
        for (size_t i = 0; i < data.size(); ++i) {
            if constexpr (std::is_arithmetic_v<T>) {
                boundary_data << i << " " << data[i] << " "
                              << (i > 0 && std::abs(data[i] - data[i - 1]) > 1.0 ? "1" : "0")
                              << "\n";
            } else {
                // For vector types, use the first element or size as indicator
                double value = data[i].empty() ? 0.0 : data[i][0];
                boundary_data << i << " " << value << " "
                              << (i > 0 && std::abs(value -
                                                    (data[i - 1].empty() ? 0.0 : data[i - 1][0])) >
                                               1.0
                                      ? "1"
                                      : "0")
                              << "\n";
            }
        }

        // Write gnuplot script
        gnuplot_script
            << "set terminal png\n"
            << "set output '" << output_dir << "/boundaries.png'\n"
            << "set title 'Chunk Boundaries'\n"
            << "set xlabel 'Index'\n"
            << "set ylabel 'Value'\n"
            << "plot '" << output_dir << "/boundaries.dat' using 1:2 with lines title 'Data', "
            << "     '" << output_dir
            << "/boundaries.dat' using 1:($3 * $2) with points pt 7 title 'Boundaries'\n";
    }

    const std::vector<T>& get_data() const {
        return data;
    }
    const std::string& get_output_dir() const {
        return output_dir;
    }
};
} // namespace chunk_viz