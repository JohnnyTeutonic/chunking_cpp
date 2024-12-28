#include "chunk_metrics.hpp"
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char* argv[]) {
    try {
        // Create test data
        std::vector<std::vector<double>> well_separated = {
            {1.0, 1.1, 1.2}, {5.0, 5.1, 5.2}, {10.0, 10.1, 10.2}};

        std::vector<std::vector<double>> mixed_chunks = {
            {1.0, 1.1, 5.0}, {2.0, 2.1, 8.0}, {3.0, 3.1, 9.0}};

        // Create analyzer
        chunk_metrics::ChunkQualityAnalyzer<double> analyzer;

        // Check for debug flag
        bool debug = (argc > 1 && std::string(argv[1]) == "--debug");

        // Compute and display metrics
        std::cout << "\nComputing chunk metrics...\n" << std::endl;

        try {
            // Cohesion metrics
            double high_cohesion = analyzer.compute_cohesion(well_separated);
            double mixed_cohesion = analyzer.compute_cohesion(mixed_chunks);

            std::cout << "Cohesion Metrics:" << std::endl;
            std::cout << "  Well-separated chunks: " << high_cohesion << std::endl;
            std::cout << "  Mixed cohesion chunks: " << mixed_cohesion << std::endl;

            // Separation metrics
            double separation = analyzer.compute_separation(well_separated);
            std::cout << "\nSeparation Metric: " << separation << std::endl;

            // Silhouette score
            double silhouette = analyzer.compute_silhouette_score(well_separated);
            std::cout << "Silhouette Score: " << silhouette << std::endl;

            // Quality scores
            double high_quality = analyzer.compute_quality_score(well_separated);
            double mixed_quality = analyzer.compute_quality_score(mixed_chunks);

            std::cout << "\nQuality Scores:" << std::endl;
            std::cout << "  Well-separated chunks: " << high_quality << std::endl;
            std::cout << "  Mixed cohesion chunks: " << mixed_quality << std::endl;

            // Size metrics
            auto size_metrics = analyzer.compute_size_metrics(well_separated);
            std::cout << "\nSize Metrics:" << std::endl;
            for (const auto& [metric, value] : size_metrics) {
                std::cout << "  " << metric << ": " << value << std::endl;
            }

            if (debug) {
                std::cout << "\nDebug Information:" << std::endl;
                std::cout << "  Number of chunks: " << well_separated.size() << std::endl;
                std::cout << "  Chunk sizes: ";
                for (const auto& chunk : well_separated) {
                    std::cout << chunk.size() << " ";
                }
                std::cout << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "Error computing metrics: " << e.what() << std::endl;
            return 1;
        }

        std::cout << "\nMetrics computation completed successfully.\n" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}