#pragma once

#include <algorithm>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <cmath>
#include <deque>
#include <memory>
#include <random>
#include <stack>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace advanced_structures {

// Default NLP model implementation
struct DefaultNLPModel {
    template <typename ContentType>
    double calculateSimilarity(const ContentType& content1, const ContentType& content2) {
        // Simple similarity calculation (e.g., based on length)
        return 1.0 - static_cast<double>(std::abs(static_cast<int>(content1.size()) -
                                                  static_cast<int>(content2.size()))) /
                         std::max(content1.size(), content2.size());
    }
};

/**
 * @brief Template class for semantic-based content chunking
 *
 * SemanticChunker splits content based on semantic boundaries using
 * configurable NLP models and similarity metrics.
 *
 * @tparam ContentType Type of content to be chunked
 * @tparam ModelType Type of NLP model to use for similarity calculations
 */
template <typename ContentType, typename ModelType = DefaultNLPModel>
class SemanticChunker {
private:
    ModelType model;             ///< NLP model instance
    double similarity_threshold; ///< Threshold for determining chunk boundaries

public:
    /**
     * @brief Construct a new Semantic Chunker
     *
     * @param threshold Similarity threshold for chunk boundaries (default: 0.7)
     * @param custom_model Custom NLP model instance (optional)
     */
    explicit SemanticChunker(double threshold = 0.7, ModelType custom_model = ModelType())
        : model(custom_model), similarity_threshold(threshold) {}

    /**
     * @brief Chunk content based on semantic boundaries
     *
     * @param content Input content to be chunked
     * @return std::vector<ContentType> Vector of content chunks
     */
    std::vector<ContentType> chunk(const ContentType& content);

    /**
     * @brief Set a new NLP model
     *
     * @param new_model New model instance to use
     */
    void setModel(ModelType new_model) {
        model = new_model;
    }

    /**
     * @brief Set new similarity threshold
     *
     * @param threshold New threshold value between 0.0 and 1.0
     */
    void setSimilarityThreshold(double threshold) {
        similarity_threshold = threshold;
    }
};

/**
 * @brief Specialization of SemanticChunker for string content
 *
 * Provides optimized implementation for string-based content chunking.
 *
 * @tparam ModelType Type of NLP model to use
 */
template <typename ModelType>
class SemanticChunker<std::string, ModelType> {
private:
    ModelType model;             ///< NLP model instance
    double similarity_threshold; ///< Similarity threshold for chunking

public:
    /**
     * @brief Construct a new string-specialized Semantic Chunker
     *
     * @param threshold Similarity threshold (default: 0.7)
     * @param custom_model Custom NLP model instance
     */
    explicit SemanticChunker(double threshold = 0.7, ModelType custom_model = ModelType())
        : model(custom_model), similarity_threshold(threshold) {}

    /**
     * @brief Chunk string content into semantic segments
     *
     * @param content Input string to be chunked
     * @return std::vector<std::string> Vector of string chunks
     */
    std::vector<std::string> chunk(const std::string& content) {
        std::vector<std::string> chunks;
        if (content.empty()) {
            return chunks;
        }

        // Split by sentences (simple implementation)
        size_t start = 0;
        size_t pos = 0;
        std::string current_chunk;

        while ((pos = content.find_first_of(".!?", start)) != std::string::npos) {
            // Include the punctuation mark and any following whitespace
            size_t end = pos + 1;
            while (end < content.length() && std::isspace(content[end])) {
                end++;
            }

            // Extract the sentence
            std::string sentence = content.substr(start, end - start);
            if (!sentence.empty()) {
                chunks.push_back(sentence);
            }

            start = end;
        }

        // Add remaining content if any
        if (start < content.length()) {
            chunks.push_back(content.substr(start));
        }

        return chunks;
    }
};

/**
 * @brief A skip list implementation for efficient chunk searching
 * @tparam T The type of elements stored in the skip list
 */
template <typename T>
class ChunkSkipList {
private:
    struct Node {
        T value;
        std::vector<std::shared_ptr<Node>> forward;
        explicit Node(T val, int level) : value(val), forward(level) {}
    };

    std::shared_ptr<Node> head;
    int max_level;
    float p;
    int current_level;

    /**
     * @brief Generates a random level for node insertion
     * @return The random level
     */
    int random_level() {
        int lvl = 1;
        while ((static_cast<float>(rand()) / RAND_MAX) < p && lvl < max_level) {
            lvl++;
        }
        return lvl;
    }

public:
    /**
     * @brief Constructs a new ChunkSkipList object
     * @param max_lvl Maximum level of the skip list
     * @param prob Probability factor for level generation
     */
    ChunkSkipList(int max_lvl = 16, float prob = 0.5)
        : max_level(max_lvl), p(prob), current_level(1) {
        head = std::make_shared<Node>(T(), max_level);
    }

    /**
     * @brief Inserts a value into the skip list
     * @param value The value to insert
     */
    void insert(const T& value) {
        std::vector<std::shared_ptr<Node>> update(max_level);
        auto current = head;

        for (int i = current_level - 1; i >= 0; i--) {
            while (current->forward[i] && current->forward[i]->value < value) {
                current = current->forward[i];
            }
            update[i] = current;
        }

        int new_level = random_level();
        if (new_level > current_level) {
            for (int i = current_level; i < new_level; i++) {
                update[i] = head;
            }
            current_level = new_level;
        }

        auto new_node = std::make_shared<Node>(value, new_level);
        for (int i = 0; i < new_level; i++) {
            new_node->forward[i] = update[i]->forward[i];
            update[i]->forward[i] = new_node;
        }
    }

    /**
     * @brief Searches for a value in the skip list
     * @param value The value to search for
     * @return True if the value is found, false otherwise
     */
    bool search(const T& value) const {
        auto current = head;
        for (int i = current_level - 1; i >= 0; i--) {
            while (current->forward[i] && current->forward[i]->value < value) {
                current = current->forward[i];
            }
        }
        current = current->forward[0];
        return current && current->value == value;
    }
};

/**
 * @brief A B+ tree implementation for chunk indexing
 * @tparam T The type of elements stored in the tree
 */
template <typename T>
class ChunkBPlusTree {
    static constexpr int ORDER = 4;

    struct Node {
        bool is_leaf;
        std::vector<T> keys;
        std::vector<std::shared_ptr<Node>> children;
        std::shared_ptr<Node> next;

        Node(bool leaf = true) : is_leaf(leaf) {}
    };

    std::shared_ptr<Node> root;

public:
    /**
     * @brief Constructs a new ChunkBPlusTree object
     */
    ChunkBPlusTree() : root(std::make_shared<Node>()) {}

    /**
     * @brief Inserts a key into the B+ tree
     * @param key The key to insert
     */
    void insert(const T& key) {
        if (root->keys.empty()) {
            root->keys.push_back(key);
            return;
        }

        if (root->keys.size() == ORDER - 1) {
            auto new_root = std::make_shared<Node>(false);
            new_root->children.push_back(root);
            split_child(new_root, 0);
            root = new_root;
        }
        insert_non_full(root, key);
    }

    /**
     * @brief Searches for a key in the B+ tree
     * @param key The key to search for
     * @return True if the key is found, false otherwise
     */
    bool search(const T& key) const {
        if (root == nullptr)
            return false;

        return search_node(root, key);
    }

private:
    void split_child(std::shared_ptr<Node> parent, int index) {
        auto child = parent->children[index];
        auto new_child = std::make_shared<Node>(child->is_leaf);

        parent->keys.insert(parent->keys.begin() + index, child->keys[ORDER / 2 - 1]);

        parent->children.insert(parent->children.begin() + index + 1, new_child);

        new_child->keys.assign(child->keys.begin() + ORDER / 2, child->keys.end());

        child->keys.resize(ORDER / 2 - 1);

        if (!child->is_leaf) {
            new_child->children.assign(child->children.begin() + ORDER / 2, child->children.end());
            child->children.resize(ORDER / 2);
        }
    }

    void insert_non_full(std::shared_ptr<Node> node, const T& key) {
        int i = node->keys.size() - 1;

        if (node->is_leaf) {
            node->keys.push_back(T());
            while (i >= 0 && key < node->keys[i]) {
                node->keys[i + 1] = node->keys[i];
                i--;
            }
            node->keys[i + 1] = key;
        } else {
            while (i >= 0 && key < node->keys[i]) {
                i--;
            }
            i++;

            if (node->children[i]->keys.size() == ORDER - 1) {
                split_child(node, i);
                if (key > node->keys[i]) {
                    i++;
                }
            }
            insert_non_full(node->children[i], key);
        }
    }

    bool search_node(const std::shared_ptr<Node>& node, const T& key) const {
        int i = 0;
        while (i < node->keys.size() && key > node->keys[i]) {
            i++;
        }

        if (i < node->keys.size() && key == node->keys[i]) {
            return true;
        }

        if (node->is_leaf) {
            return false;
        }

        return search_node(node->children[i], key);
    }
};

/**
 * @brief A deque-based chunk structure for double-ended operations
 * @tparam T The type of elements stored in the chunk deque
 */
template <typename T>
class ChunkDeque {
private:
    std::deque<T> data_;

public:
    void push_back(const T& value) {
        data_.push_back(value);
    }

    void push_front(const T& value) {
        data_.push_front(value);
    }

    T pop_back() {
        T value = data_.back();
        data_.pop_back();
        return value;
    }

    T pop_front() {
        T value = data_.front();
        data_.pop_front();
        return value;
    }

    size_t size() const {
        return data_.size();
    }

    bool empty() const {
        return data_.empty();
    }
};

/**
 * @brief A stack-based chunk structure for LIFO operations
 * @tparam T The type of elements stored in the chunk stack
 */
template <typename T>
class ChunkStack {
private:
    std::stack<T> data_;

public:
    void push(const T& value) {
        data_.push(value);
    }

    T pop() {
        T value = data_.top();
        data_.pop();
        return value;
    }

    size_t size() const {
        return data_.size();
    }

    bool empty() const {
        return data_.empty();
    }
};

/**
 * @brief A treap implementation for efficient chunk searching and manipulation
 * @tparam T The type of elements stored in the treap
 */
template <typename T>
class ChunkTreap {
private:
    struct Node {
        T value;
        int priority;
        std::shared_ptr<Node> left;
        std::shared_ptr<Node> right;

        Node(T val, int prio) : value(val), priority(prio), left(nullptr), right(nullptr) {}
    };

    std::shared_ptr<Node> root;
    std::mt19937 gen;

    std::shared_ptr<Node> rotate_right(std::shared_ptr<Node> node) {
        auto new_root = node->left;
        node->left = new_root->right;
        new_root->right = node;
        return new_root;
    }

    std::shared_ptr<Node> rotate_left(std::shared_ptr<Node> node) {
        auto new_root = node->right;
        node->right = new_root->left;
        new_root->left = node;
        return new_root;
    }

    std::shared_ptr<Node> insert(std::shared_ptr<Node> node, T value, int priority) {
        if (!node) {
            return std::make_shared<Node>(value, priority);
        }

        if (value < node->value) {
            node->left = insert(node->left, value, priority);
            if (node->left->priority > node->priority) {
                node = rotate_right(node);
            }
        } else {
            node->right = insert(node->right, value, priority);
            if (node->right->priority > node->priority) {
                node = rotate_left(node);
            }
        }
        return node;
    }

    bool search(const std::shared_ptr<Node>& node, T value) const {
        if (!node) {
            return false;
        }
        if (node->value == value) {
            return true;
        }
        if (value < node->value) {
            return search(node->left, value);
        } else {
            return search(node->right, value);
        }
    }

public:
    ChunkTreap() : root(nullptr), gen(std::random_device{}()) {}

    void insert(T value) {
        int priority = std::uniform_int_distribution<>(1, 100)(gen);
        root = insert(root, value, priority);
    }

    bool search(T value) const {
        return search(root, value);
    }
};

/**
 * @brief Semantic boundaries-based chunking implementation
 * @tparam T The type of elements to be chunked
 */
template <typename T>
class SemanticBoundariesChunk {
private:
    double boundary_threshold;

public:
    explicit SemanticBoundariesChunk(double threshold = 0.5) : boundary_threshold(threshold) {}

    std::vector<std::vector<T>> chunk(const std::vector<T>& data) {
        std::vector<std::vector<T>> result;
        if (data.empty())
            return result;

        std::vector<T> current_chunk;
        for (const auto& item : data) {
            current_chunk.push_back(item);
            if (isBoundary(current_chunk)) {
                result.push_back(current_chunk);
                current_chunk.clear();
            }
        }

        if (!current_chunk.empty()) {
            result.push_back(current_chunk);
        }

        return result;
    }

protected:
    virtual bool isBoundary(const std::vector<T>& chunk) {
        return chunk.size() >= 3; // Default implementation
    }
};

/**
 * @brief Fractal pattern-based chunking implementation
 * @tparam T The type of elements to be chunked
 */
template <typename T>
class FractalPatternsChunk {
private:
    size_t pattern_size;
    double similarity_threshold;

public:
    FractalPatternsChunk(size_t size = 3, double threshold = 0.8)
        : pattern_size(size), similarity_threshold(threshold) {}

    std::vector<std::vector<T>> chunk(const std::vector<T>& data) {
        std::vector<std::vector<T>> result;
        if (data.empty())
            return result;

        std::vector<T> current_chunk;
        for (const auto& item : data) {
            current_chunk.push_back(item);
            if (hasPattern(current_chunk)) {
                result.push_back(current_chunk);
                current_chunk.clear();
            }
        }

        if (!current_chunk.empty()) {
            result.push_back(current_chunk);
        }

        return result;
    }

protected:
    virtual bool hasPattern(const std::vector<T>& chunk) {
        return chunk.size() >= pattern_size;
    }
};

/**
 * @brief Bloom filter-based chunking implementation
 * @tparam T The type of elements to be chunked
 */
template <typename T>
class BloomFilterChunk {
private:
    size_t filter_size;
    size_t num_hash_functions;
    std::vector<bool> filter;

public:
    BloomFilterChunk(size_t size = 1024, size_t num_funcs = 3)
        : filter_size(size), num_hash_functions(num_funcs), filter(size, false) {}

    std::vector<std::vector<T>> chunk(const std::vector<T>& data) {
        std::vector<std::vector<T>> result;
        if (data.empty())
            return result;

        std::vector<T> current_chunk;
        for (const auto& item : data) {
            current_chunk.push_back(item);
            if (shouldSplit(current_chunk)) {
                result.push_back(current_chunk);
                current_chunk.clear();
            }
        }

        if (!current_chunk.empty()) {
            result.push_back(current_chunk);
        }

        return result;
    }

protected:
    virtual bool shouldSplit(const std::vector<T>& chunk) {
        return chunk.size() >= 4; // Default implementation
    }
};

/**
 * @brief Graph-based chunking implementation
 * @tparam T The type of elements to be chunked
 */
template <typename T>
class GraphBasedChunk {
private:
    using Graph = boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS>;
    double edge_threshold;

public:
    explicit GraphBasedChunk(double threshold = 0.5) : edge_threshold(threshold) {}

    std::vector<std::vector<T>> chunk(const std::vector<T>& data) {
        std::vector<std::vector<T>> result;
        if (data.empty())
            return result;

        Graph g(data.size());
        buildGraph(data, g);

        std::vector<int> components(data.size());
        int num_components = boost::connected_components(g, &components[0]);

        result.resize(num_components);
        for (size_t i = 0; i < data.size(); ++i) {
            result[components[i]].push_back(data[i]);
        }

        return result;
    }

protected:
    virtual void buildGraph(const std::vector<T>& data, Graph& g) {
        // Default implementation: connect adjacent elements
        for (size_t i = 1; i < data.size(); ++i) {
            boost::add_edge(i - 1, i, g);
        }
    }
};

template <typename T>
class ChunkLSMTree {
private:
    struct Level {
        std::vector<T> data;
        size_t size_limit;

        Level(size_t limit) : size_limit(limit) {}

        bool is_full() const {
            return data.size() >= size_limit;
        }
    };

    std::vector<std::shared_ptr<Level>> levels;
    std::vector<T> memtable; // In-memory buffer
    size_t memtable_size_limit;
    size_t size_ratio; // Size ratio between levels

    void compact_level(size_t level_idx) {
        if (level_idx >= levels.size() - 1) {
            // Create new level if we're at the last one
            levels.push_back(std::make_shared<Level>(levels[level_idx]->size_limit * size_ratio));
        }

        auto& current_level = levels[level_idx]->data;
        auto& next_level = levels[level_idx + 1]->data;

        // Merge current level into next level
        std::vector<T> merged;
        std::merge(current_level.begin(), current_level.end(), next_level.begin(), next_level.end(),
                   std::back_inserter(merged));

        // Update levels
        current_level.clear();
        next_level = std::move(merged);

        // If next level is full, compact it too
        if (levels[level_idx + 1]->is_full()) {
            compact_level(level_idx + 1);
        }
    }

    void flush_memtable() {
        if (memtable.empty())
            return;

        // Sort memtable before flushing
        std::sort(memtable.begin(), memtable.end());

        if (levels.empty()) {
            levels.push_back(std::make_shared<Level>(memtable_size_limit * size_ratio));
        }

        // Merge memtable with first level
        std::vector<T> merged;
        std::merge(memtable.begin(), memtable.end(), levels[0]->data.begin(), levels[0]->data.end(),
                   std::back_inserter(merged));

        levels[0]->data = std::move(merged);
        memtable.clear();

        // If level 0 is full, trigger compaction
        if (levels[0]->is_full()) {
            compact_level(0);
        }
    }

public:
    ChunkLSMTree(size_t memtable_limit = 1024, size_t ratio = 4)
        : memtable_size_limit(memtable_limit), size_ratio(ratio) {}

    void insert(const T& value) {
        memtable.push_back(value);

        if (memtable.size() >= memtable_size_limit) {
            flush_memtable();
        }
    }

    bool search(const T& value) const {
        // Search memtable first
        if (std::binary_search(memtable.begin(), memtable.end(), value)) {
            return true;
        }

        // Search through all levels
        for (const auto& level : levels) {
            if (std::binary_search(level->data.begin(), level->data.end(), value)) {
                return true;
            }
        }
        return false;
    }

    void force_flush() {
        flush_memtable();
    }
};

} // namespace advanced_structures
