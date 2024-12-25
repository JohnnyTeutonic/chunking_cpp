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
#include <filesystem>
#include <fstream>
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
    const std::vector<T>& data;
    std::string output_dir;

    std::string format_value(const T& value) {
        if constexpr (std::is_arithmetic_v<T>) {
            return std::to_string(value);
        } else if constexpr (std::is_same_v<T, std::vector<int>> ||
                             std::is_same_v<T, std::vector<double>> ||
                             std::is_same_v<T, std::vector<float>>) {
            std::stringstream ss;
            ss << "[";
            for (size_t i = 0; i < value.size(); ++i) {
                ss << value[i];
                if (i < value.size() - 1) {
                    ss << ",";
                }
            }
            ss << "]";
            return ss.str();
        } else {
            throw chunk_processing::VisualizationError("Unsupported type for visualization");
        }
    }

public:
    explicit ChunkVisualizer(const std::vector<T>& data_, const std::string& output_dir_ = "./viz")
        : data(data_), output_dir(output_dir_) {
        if (data_.empty()) {
            throw chunk_processing::VisualizationError("Cannot visualize empty data");
        }
        std::filesystem::create_directories(output_dir);
    }

    void plot_chunk_sizes() {
        std::ofstream plot_data(output_dir + "/chunk_sizes.dat");
        if (!plot_data) {
            throw chunk_processing::VisualizationError("Failed to create chunk_sizes.dat");
        }

        for (size_t i = 0; i < data.size(); ++i) {
            if constexpr (std::is_arithmetic_v<T>) {
                plot_data << i << " " << std::abs(data[i]) << "\n";
            } else {
                plot_data << i << " " << data[i].size() << "\n";
            }
        }
    }

    void export_to_graphviz(const std::string& filename = "chunks.dot") {
        std::string actual_filename = output_dir + "/" + filename;
        std::ofstream file(actual_filename);
        if (!file) {
            throw chunk_processing::VisualizationError("Failed to create GraphViz file");
        }

        file << "digraph chunks {\n";
        for (size_t i = 0; i < data.size(); ++i) {
            file << "  chunk" << i << " [label=\"Value: " << format_value(data[i]) << "\"];\n";
            if (i > 0) {
                file << "  chunk" << (i - 1) << " -> chunk" << i << ";\n";
            }
        }
        file << "}\n";
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