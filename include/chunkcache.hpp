/**
 * @file chunkcache.hpp
 * @brief LRU cache implementation for chunk data
 * @author Assistant
 * @date 2024-03-19
 */

#pragma once

#include <chrono>
#include <list>
#include <stdexcept>
#include <unordered_map>

namespace chunk_processing {

/**
 * @brief A cache implementation for chunk data with LRU eviction policy
 *
 * @tparam T The type of elements stored in the chunks
 */
template <typename T>
class ChunkCache {
private:
    struct CacheEntry {
        std::vector<T> data;                               ///< Chunk data
        size_t access_count{0};                            ///< Number of times accessed
        std::chrono::steady_clock::time_point last_access; ///< Last access timestamp
        typename std::list<size_t>::iterator list_it;      ///< Iterator in LRU list
    };

    std::unordered_map<size_t, CacheEntry> cache_; ///< Cache storage
    std::list<size_t> lru_list_;                   ///< LRU tracking list
    size_t max_entries_;                           ///< Maximum cache size
    size_t hits_{0};                               ///< Cache hit counter
    size_t misses_{0};                             ///< Cache miss counter

public:
    /**
     * @brief Construct a new Chunk Cache
     *
     * @param max_size Maximum number of entries in cache
     * @throw std::invalid_argument if max_size is 0
     */
    explicit ChunkCache(size_t max_size) : max_entries_(max_size) {
        if (max_size == 0)
            throw std::invalid_argument("Cache size must be positive");
    }

    /**
     * @brief Insert a chunk into the cache
     *
     * @param key Unique identifier for the chunk
     * @param chunk Vector of values representing the chunk
     */
    void insert(size_t key, const std::vector<T>& chunk) {
        if (cache_.size() >= max_entries_ && cache_.find(key) == cache_.end()) {
            evict_lru();
        }

        auto now = std::chrono::steady_clock::now();

        auto [it, inserted] = cache_.try_emplace(key);
        if (inserted) {
            lru_list_.push_front(key);
            it->second.list_it = lru_list_.begin();
        } else {
            lru_list_.erase(it->second.list_it);
            lru_list_.push_front(key);
            it->second.list_it = lru_list_.begin();
        }

        it->second.data = chunk;
        it->second.access_count++;
        it->second.last_access = now;
    }

    /**
     * @brief Retrieve a chunk from the cache
     *
     * @param key Unique identifier for the chunk
     * @param out_chunk Vector to store the retrieved chunk
     * @return bool True if chunk was found, false otherwise
     */
    bool get(size_t key, std::vector<T>& out_chunk) {
        auto it = cache_.find(key);
        if (it == cache_.end()) {
            misses_++;
            return false;
        }

        hits_++;
        out_chunk = it->second.data;
        it->second.access_count++;
        it->second.last_access = std::chrono::steady_clock::now();

        lru_list_.erase(it->second.list_it);
        lru_list_.push_front(key);
        it->second.list_it = lru_list_.begin();

        return true;
    }

    /**
     * @brief Get the cache hit rate
     *
     * @return double Hit rate between 0.0 and 1.0
     */
    double hit_rate() const {
        size_t total = hits_ + misses_;
        return total > 0 ? static_cast<double>(hits_) / total : 0.0;
    }

    /**
     * @brief Clear the cache
     */
    void clear() {
        cache_.clear();
        lru_list_.clear();
        hits_ = 0;
        misses_ = 0;
    }

private:
    void evict_lru() {
        if (!lru_list_.empty()) {
            cache_.erase(lru_list_.back());
            lru_list_.pop_back();
        }
    }
};

} // namespace chunk_processing