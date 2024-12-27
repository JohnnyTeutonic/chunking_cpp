#include "test_base.hpp"

// Define static members
std::mutex ChunkTestBase::global_test_mutex_;
std::condition_variable ChunkTestBase::test_cv_;
bool ChunkTestBase::test_running_ = false;

void ChunkTestBase::SetUp() {
    std::unique_lock<std::mutex> lock(global_test_mutex_);
    // Wait until no other test is running
    test_cv_.wait(lock, [] { return !test_running_; });
    test_running_ = true;
}

void ChunkTestBase::TearDown() {
    {
        std::lock_guard<std::mutex> lock(global_test_mutex_);
        test_running_ = false;
    }
    // Notify next test can run
    test_cv_.notify_one();
    // Add cooldown period between tests
    std::this_thread::sleep_for(TEST_COOLDOWN);
}