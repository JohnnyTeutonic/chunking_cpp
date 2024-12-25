#pragma once

#include "chunk_common.hpp"
#include "chunk_errors.hpp"
#include <algorithm> // for std::min_element, std::max_element, std::sort, std::find_if
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <functional> // for std::greater
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unistd.h> // for getpagesize()
#include <unordered_map>
#include <vector>

namespace chunk_resilience {

/**
 * @brief Checkpoint data structure
 */
template <typename T>
struct CHUNK_EXPORT Checkpoint {
    size_t sequence_number;
    std::vector<std::vector<T>> chunks;
    std::chrono::system_clock::time_point timestamp;
    size_t memory_usage;
    bool is_corrupted;

    // Serialization
    void serialize(const std::string& filename) const {
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            throw chunk_processing::ChunkingError("Failed to create checkpoint file: " + filename);
        }

        // Write metadata
        file.write(reinterpret_cast<const char*>(&sequence_number), sizeof(sequence_number));
        file.write(reinterpret_cast<const char*>(&memory_usage), sizeof(memory_usage));

        // Write chunks
        size_t num_chunks = chunks.size();
        file.write(reinterpret_cast<const char*>(&num_chunks), sizeof(num_chunks));

        for (const auto& chunk : chunks) {
            size_t chunk_size = chunk.size();
            file.write(reinterpret_cast<const char*>(&chunk_size), sizeof(chunk_size));
            file.write(reinterpret_cast<const char*>(chunk.data()), chunk_size * sizeof(T));
        }
    }

    // Deserialization
    static Checkpoint<T> deserialize(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw chunk_processing::ChunkingError("Failed to open checkpoint file: " + filename);
        }

        Checkpoint<T> checkpoint;

        try {
            // Read metadata
            file.read(reinterpret_cast<char*>(&checkpoint.sequence_number),
                      sizeof(checkpoint.sequence_number));
            file.read(reinterpret_cast<char*>(&checkpoint.memory_usage),
                      sizeof(checkpoint.memory_usage));

            // Read chunks
            size_t num_chunks;
            file.read(reinterpret_cast<char*>(&num_chunks), sizeof(num_chunks));

            checkpoint.chunks.resize(num_chunks);
            for (auto& chunk : checkpoint.chunks) {
                size_t chunk_size;
                file.read(reinterpret_cast<char*>(&chunk_size), sizeof(chunk_size));

                chunk.resize(chunk_size);
                file.read(reinterpret_cast<char*>(chunk.data()), chunk_size * sizeof(T));
            }

            checkpoint.timestamp = std::chrono::system_clock::now();
            checkpoint.is_corrupted = false;

        } catch (const std::exception& e) {
            checkpoint.is_corrupted = true;
            throw chunk_processing::ChunkingError("Checkpoint corruption detected: " +
                                                  std::string(e.what()));
        }

        return checkpoint;
    }
};

/**
 * @brief Resilient chunking system with error recovery
 */
template <typename T>
class CHUNK_EXPORT ResilientChunker {
private:
    std::string checkpoint_dir;
    size_t max_memory_usage;
    size_t checkpoint_interval;
    std::atomic<size_t> current_sequence;

    std::mutex mutex;
    std::condition_variable cv;

    std::unordered_map<size_t, Checkpoint<T>> checkpoint_history;
    size_t max_history_size;

    /**
     * @brief Create a checkpoint of current state
     */
    void create_checkpoint(const std::vector<std::vector<T>>& chunks) {
        std::lock_guard<std::mutex> lock(mutex);

        Checkpoint<T> checkpoint{current_sequence++, chunks, std::chrono::system_clock::now(),
                                 get_memory_usage(), false};

        // Save checkpoint to file
        std::string filename =
            checkpoint_dir + "/checkpoint_" + std::to_string(checkpoint.sequence_number) + ".bin";
        checkpoint.serialize(filename);

        // Add to history
        checkpoint_history[checkpoint.sequence_number] = checkpoint;

        // Maintain history size
        if (checkpoint_history.size() > max_history_size) {
            auto oldest =
                std::min_element(checkpoint_history.begin(), checkpoint_history.end(),
                                 [](const auto& a, const auto& b) {
                                     return a.second.sequence_number < b.second.sequence_number;
                                 });

            std::filesystem::remove(checkpoint_dir + "/checkpoint_" +
                                    std::to_string(oldest->first) + ".bin");
            checkpoint_history.erase(oldest);
        }
    }

    /**
     * @brief Get current memory usage
     */
    size_t get_memory_usage() const {
#ifdef __linux__
        std::ifstream stat("/proc/self/statm");
        size_t resident;
        stat >> resident; // Skip first value
        stat >> resident; // Read resident set size
#ifdef _SC_PAGESIZE
        return resident * sysconf(_SC_PAGESIZE);
#else
        return resident * 4096; // Default page size
#endif
#else
        return 0; // Placeholder for non-Linux systems
#endif
    }

    /**
     * @brief Verify checkpoint integrity
     */
    bool verify_checkpoint(const Checkpoint<T>& checkpoint) {
        return !checkpoint.is_corrupted && checkpoint.memory_usage <= max_memory_usage;
    }

    void initialize_checkpoint_dir() {
        try {
            std::filesystem::create_directories(checkpoint_dir);
        } catch (const std::exception& e) {
            throw chunk_processing::ResilienceError("Failed to create checkpoint directory: " +
                                                    std::string(e.what()));
        }
    }

public:
    ResilientChunker(const std::string& checkpoint_directory = "./checkpoints",
                     size_t max_mem_usage = 1024 * 1024 * 1024, // 1GB
                     size_t checkpoint_freq = 1000,             // chunks
                     size_t history_size = 5)                   // checkpoints
        : checkpoint_dir(checkpoint_directory), max_memory_usage(max_mem_usage),
          checkpoint_interval(checkpoint_freq), current_sequence(0),
          max_history_size(history_size) {
        // Create checkpoint directory
        std::filesystem::create_directories(checkpoint_dir);
    }

    /**
     * @brief Process data with checkpointing and error recovery
     */
    std::vector<std::vector<T>> process(const std::vector<T>& data) {
        if (data.empty()) {
            throw chunk_processing::ResilienceError("Cannot process empty data");
        }

        try {
            initialize_checkpoint_dir();
            std::vector<std::vector<T>> result;
            size_t processed = 0;

            // Create initial checkpoint
            create_checkpoint(result);

            while (processed < data.size()) {
                // Check memory usage
                if (get_memory_usage() > max_memory_usage) {
                    handle_memory_exhaustion();
                }

                // Process next chunk
                size_t chunk_size = std::min(checkpoint_interval, data.size() - processed);
                std::vector<T> current_chunk(data.begin() + processed,
                                             data.begin() + processed + chunk_size);
                result.push_back(current_chunk);
                processed += chunk_size;

                // Create checkpoint if needed
                if (processed % checkpoint_interval == 0) {
                    create_checkpoint(result);
                }
            }

            return result;
        } catch (const std::exception& e) {
            throw chunk_processing::ResilienceError("Processing failed: " + std::string(e.what()));
        }
    }

    /**
     * @brief Save current state
     */
    void save_checkpoint() {
        if (!checkpoint_history.empty()) {
            auto latest =
                std::max_element(checkpoint_history.begin(), checkpoint_history.end(),
                                 [](const auto& a, const auto& b) {
                                     return a.second.sequence_number < b.second.sequence_number;
                                 });
            latest->second.serialize(checkpoint_dir + "/latest_checkpoint.bin");
        }
    }

    /**
     * @brief Restore from latest valid checkpoint
     */
    std::vector<std::vector<T>> restore_from_checkpoint() {
        std::lock_guard<std::mutex> lock(mutex);

        // Find latest valid checkpoint
        std::vector<size_t> keys;
        for (const auto& pair : checkpoint_history) {
            keys.push_back(pair.first);
        }
        std::sort(keys.begin(), keys.end(), std::greater<size_t>());

        auto latest_valid = std::find_if(keys.begin(), keys.end(), [this](const auto& checkpoint) {
            return verify_checkpoint(checkpoint_history[checkpoint]);
        });

        if (latest_valid == keys.end()) {
            throw chunk_processing::ChunkingError("No valid checkpoint found for recovery");
        }

        return checkpoint_history[*latest_valid].chunks;
    }

    /**
     * @brief Handle memory exhaustion
     */
    void handle_memory_exhaustion() {
        // Free up memory by clearing old checkpoints
        checkpoint_history.clear();

        // Force garbage collection
        std::vector<std::vector<T>>().swap(checkpoint_history[current_sequence].chunks);

        // Wait for memory to be freed
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (get_memory_usage() > max_memory_usage) {
            throw chunk_processing::ChunkingError("Memory exhaustion: Unable to recover");
        }
    }

    /**
     * @brief Handle data corruption
     */
    void handle_corruption() {
        std::cerr << "Data corruption detected at sequence " << current_sequence << std::endl;

        if (checkpoint_history.count(current_sequence)) {
            checkpoint_history[current_sequence].is_corrupted = true;
        }

        // Attempt to restore from last valid checkpoint
        auto valid_checkpoint = restore_from_checkpoint();

        // Find highest sequence number
        size_t max_seq = 0;
        for (const auto& [seq, _] : checkpoint_history) {
            max_seq = std::max(max_seq, seq);
        }
        current_sequence.store(max_seq);
    }

    // Add getters
    size_t get_max_memory_usage() const {
        return max_memory_usage;
    }
    size_t get_checkpoint_interval() const {
        return checkpoint_interval;
    }
    size_t get_max_history_size() const {
        return max_history_size;
    }
    size_t get_current_sequence() const {
        return current_sequence.load();
    }
    const std::string& get_checkpoint_dir() const {
        return checkpoint_dir;
    }

    // Add setters
    void set_max_memory_usage(size_t usage) {
        max_memory_usage = usage;
    }
    void set_checkpoint_interval(size_t interval) {
        checkpoint_interval = interval;
    }
    void set_max_history_size(size_t size) {
        max_history_size = size;
    }
};

} // namespace chunk_resilience