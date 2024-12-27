#include "chunk_metrics.hpp"
#include "test_base.hpp"
#include <memory>

class ChunkMetricsTest : public ChunkTestBase {
protected:
    std::unique_ptr<chunk_metrics::ChunkQualityAnalyzer<double>> analyzer;
    std::vector<std::vector<double>> well_separated_chunks;
    std::vector<std::vector<double>> mixed_cohesion_chunks;
    std::vector<std::vector<double>> empty_chunks;

    void SetUp() override {
        ChunkTestBase::SetUp(); // Call base setup first

        try {
            analyzer = std::make_unique<chunk_metrics::ChunkQualityAnalyzer<double>>();

            // Initialize test data
            well_separated_chunks = {{1.0, 1.1, 1.2}, {5.0, 5.1, 5.2}, {10.0, 10.1, 10.2}};

            mixed_cohesion_chunks = {{1.0, 1.1, 5.0}, {2.0, 2.1, 8.0}, {3.0, 3.1, 9.0}};
        } catch (const std::exception& e) {
            FAIL() << "Setup failed: " << e.what();
        }
    }

    void TearDown() override {
        try {
            if (analyzer) {
                analyzer->clear_cache();
                safe_cleanup(analyzer);
            }
            well_separated_chunks.clear();
            mixed_cohesion_chunks.clear();
            empty_chunks.clear();
        } catch (...) {
            // Ensure base teardown still happens
        }

        ChunkTestBase::TearDown(); // Call base teardown last
    }
};

TEST_F(ChunkMetricsTest, CohesionCalculation) {
    ASSERT_TRUE(is_valid_resource(analyzer));

    std::unique_lock<std::mutex> lock(global_test_mutex_);
    double high_cohesion = analyzer->compute_cohesion(well_separated_chunks);
    double mixed_cohesion = analyzer->compute_cohesion(mixed_cohesion_chunks);
    lock.unlock();

    EXPECT_GT(high_cohesion, mixed_cohesion);
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
    double high_quality = analyzer->compute_quality_score(well_separated_chunks);
    double mixed_quality = analyzer->compute_quality_score(mixed_cohesion_chunks);

    EXPECT_GT(high_quality, mixed_quality);
    EXPECT_GE(high_quality, 0.0);
    EXPECT_LE(high_quality, 1.0);
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