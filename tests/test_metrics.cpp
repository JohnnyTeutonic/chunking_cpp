#include "chunk_metrics.hpp"
#include "test_base.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

class ChunkMetricsTest : public ChunkTestBase {
protected:
    std::unique_ptr<chunk_metrics::ChunkQualityAnalyzer<double>> analyzer;
    std::vector<std::vector<double>> well_separated_chunks;
    std::vector<std::vector<double>> mixed_cohesion_chunks;
    std::vector<std::vector<double>> empty_chunks;
    mutable std::mutex test_mutex_;
    std::atomic<bool> test_running_{false};

    void SetUp() override {
        ChunkTestBase::SetUp();

        try {
            // Initialize analyzer with proper error checking
            analyzer = std::make_unique<chunk_metrics::ChunkQualityAnalyzer<double>>();
            if (!analyzer) {
                throw std::runtime_error("Failed to create analyzer");
            }

            // Initialize test data with bounds checking
            well_separated_chunks = {
                std::vector<double>{1.0, 1.1, 1.2},
                std::vector<double>{5.0, 5.1, 5.2},
                std::vector<double>{10.0, 10.1, 10.2}
            };

            mixed_cohesion_chunks = {
                std::vector<double>{1.0, 1.1, 5.0},
                std::vector<double>{2.0, 2.1, 8.0},
                std::vector<double>{3.0, 3.1, 9.0}
            };

            // Validate test data
            for (const auto& chunk : well_separated_chunks) {
                if (chunk.empty() || chunk.size() > 1000000) {
                    throw std::runtime_error("Invalid test data in well_separated_chunks");
                }
            }
            for (const auto& chunk : mixed_cohesion_chunks) {
                if (chunk.empty() || chunk.size() > 1000000) {
                    throw std::runtime_error("Invalid test data in mixed_cohesion_chunks");
                }
            }

        } catch (const std::exception& e) {
            FAIL() << "Setup failed: " << e.what();
        }
    }

    void TearDown() override {
        try {
            std::lock_guard<std::mutex> lock(test_mutex_);
            
            if (analyzer) {
                // Ensure no computations are running
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                analyzer.reset();
            }
            
            well_separated_chunks.clear();
            mixed_cohesion_chunks.clear();
            empty_chunks.clear();
            
        } catch (...) {
            // Ensure base teardown still happens
        }

        ChunkTestBase::TearDown();
    }

    // Helper to safely run computations with explicit return type
    template<typename Func>
    auto run_safely(Func&& func) -> typename std::invoke_result<Func>::type {
        if (test_running_.exchange(true)) {
            throw std::runtime_error("Test already running");
        }
        
        struct TestGuard {
            std::atomic<bool>& flag;
            TestGuard(std::atomic<bool>& f) : flag(f) {}
            ~TestGuard() { flag = false; }
        } guard(test_running_);

        return func();
    }
};

TEST_F(ChunkMetricsTest, CohesionCalculation) {
    std::cout << "Starting CohesionCalculation test" << std::endl;
    
    ASSERT_TRUE(analyzer != nullptr) << "Analyzer is null";
    ASSERT_FALSE(well_separated_chunks.empty()) << "Well separated chunks is empty";
    ASSERT_FALSE(mixed_cohesion_chunks.empty()) << "Mixed cohesion chunks is empty";
    
    try {
        // Run computation safely with explicit lambda return type
        auto result = run_safely([this]() -> std::pair<double, double> {
            double high_cohesion = 0.0;
            double mixed_cohesion = 0.0;
            
            {
                std::unique_lock<std::mutex> lock(test_mutex_);
                bool success = analyzer->compare_cohesion(
                    well_separated_chunks,
                    mixed_cohesion_chunks,
                    high_cohesion,
                    mixed_cohesion
                );
                
                if (!success || !std::isfinite(high_cohesion) || !std::isfinite(mixed_cohesion)) {
                    throw std::runtime_error("Invalid cohesion computation results");
                }
                
                return std::make_pair(high_cohesion, mixed_cohesion);
            }
        });
        
        EXPECT_GT(result.first, result.second) 
            << "High cohesion (" << result.first 
            << ") should be greater than mixed cohesion (" << result.second << ")";
            
    } catch (const std::exception& e) {
        FAIL() << "Unexpected exception: " << e.what();
    }
}

TEST_F(ChunkMetricsTest, SeparationCalculation) {
    double separation = analyzer->compute_separation(well_separated_chunks);
    EXPECT_GT(separation, 0.0);
    EXPECT_LE(separation, 1.0);
}

TEST_F(ChunkMetricsTest, SilhouetteScore) {
    double silhouette = analyzer->compute_silhouette_score(well_separated_chunks);
    EXPECT_GE(silhouette, -1.0);
    EXPECT_LE(silhouette, 1.0);
}

TEST_F(ChunkMetricsTest, QualityScore) {
    ASSERT_TRUE(analyzer != nullptr);
    
    try {
        auto result = run_safely([this]() -> std::pair<double, double> {
            std::unique_lock<std::mutex> lock(test_mutex_);
            
            double high_quality = analyzer->compute_quality_score(well_separated_chunks);
            analyzer->clear_cache();
            double mixed_quality = analyzer->compute_quality_score(mixed_cohesion_chunks);
            
            if (!std::isfinite(high_quality) || !std::isfinite(mixed_quality)) {
                throw std::runtime_error("Invalid quality score results");
            }
            
            return std::make_pair(high_quality, mixed_quality);
        });

        EXPECT_GT(result.first, result.second) << "High quality should be greater than mixed quality";
        EXPECT_GE(result.first, 0.0) << "Quality score should be non-negative";
        EXPECT_LE(result.first, 1.0) << "Quality score should not exceed 1.0";
        
    } catch (const std::exception& e) {
        FAIL() << "Unexpected exception: " << e.what();
    }
}

TEST_F(ChunkMetricsTest, SizeMetrics) {
    auto metrics = analyzer->compute_size_metrics(well_separated_chunks);

    EXPECT_EQ(metrics["average_size"], 3.0);
    EXPECT_EQ(metrics["max_size"], 3.0);
    EXPECT_EQ(metrics["min_size"], 3.0);
    EXPECT_NEAR(metrics["size_variance"], 0.0, 1e-10);
}

TEST_F(ChunkMetricsTest, EmptyChunks) {
    std::vector<std::vector<double>> empty_chunks;
    EXPECT_THROW(analyzer->compute_quality_score(empty_chunks), std::invalid_argument);
    EXPECT_THROW(analyzer->compute_cohesion(empty_chunks), std::invalid_argument);
    EXPECT_THROW(analyzer->compute_separation(empty_chunks), std::invalid_argument);
    EXPECT_THROW(analyzer->compute_silhouette_score(empty_chunks), std::invalid_argument);
    EXPECT_THROW(analyzer->compute_size_metrics(empty_chunks), std::invalid_argument);
}

TEST_F(ChunkMetricsTest, SingleChunk) {
    std::vector<std::vector<double>> single_chunk = {{1.0, 2.0, 3.0}};
    EXPECT_NO_THROW(analyzer->compute_cohesion(single_chunk));
    EXPECT_THROW(analyzer->compute_separation(single_chunk), std::invalid_argument);
    EXPECT_THROW(analyzer->compute_silhouette_score(single_chunk), std::invalid_argument);
    EXPECT_NO_THROW(analyzer->compute_quality_score(single_chunk));
}

TEST_F(ChunkMetricsTest, CacheClear) {
    analyzer->compute_cohesion(well_separated_chunks);
    analyzer->compute_separation(well_separated_chunks);
    analyzer->clear_cache();
    // Verify the function runs without errors
    EXPECT_NO_THROW(analyzer->compute_cohesion(well_separated_chunks));
}