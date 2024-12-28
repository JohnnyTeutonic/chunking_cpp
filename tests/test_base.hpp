#pragma once

#include <chrono>
#include <condition_variable>
#include <gtest/gtest.h>
#include <mutex>
#include <thread>

class ChunkTestBase : public ::testing::Test {
protected:
    static std::mutex global_test_mutex_;
    static std::condition_variable test_cv_;
    static bool test_running_;
    static constexpr auto TEST_COOLDOWN = std::chrono::milliseconds(100);

    void SetUp() override;
    void TearDown() override;

    // Helper method to safely clean up resources
    template <typename T>
    void safe_cleanup(T& resource) {
        try {
            if (resource) {
                resource.reset();
            }
        } catch (...) {
            // Log or handle cleanup errors
        }
    }

    // Helper method to verify resource validity
    template <typename T>
    bool is_valid_resource(const T& resource) {
        return resource != nullptr;
    }
};