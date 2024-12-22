#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "chunk.hpp"
#include "chunk_strategies.hpp"
#include "chunk_compression.hpp"
#include "chunk_metrics.hpp"
#include "chunk_visualization.hpp"
#include "chunk_serialization.hpp"
#include "chunk_resilience.hpp"
#include "chunk_integrations.hpp"
#include "chunk_benchmark.hpp"
#include "neural_chunking.hpp"
#ifdef HAVE_CUDA
#include "gpu_chunking.hpp"
#endif
#include "sophisticated_chunking.hpp"

namespace py = pybind11;

PYBIND11_MODULE(chunking_cpp, m) {
    m.doc() = "Python bindings for the C++ chunking library";

    // Basic Chunking
    py::class_<Chunk<double>>(m, "Chunk")
        .def(py::init<size_t>())
        .def("add", &Chunk<double>::add)
        .def("chunk_by_size", &Chunk<double>::chunk_by_size)
        .def("chunk_by_threshold", &Chunk<double>::chunk_by_threshold)
        .def("chunk_by_similarity", &Chunk<double>::chunk_by_similarity);

    // Neural Chunking
    py::class_<neural_chunking::NeuralChunking<double>>(m, "NeuralChunking")
        .def(py::init<size_t, double>())
        .def("chunk", &neural_chunking::NeuralChunking<double>::chunk)
        .def("set_threshold", &neural_chunking::NeuralChunking<double>::set_threshold)
        .def("get_window_size", &neural_chunking::NeuralChunking<double>::get_window_size);

    // GPU Chunking
#ifdef HAVE_CUDA
    py::class_<gpu_chunking::GPUChunkProcessor<double>>(m, "GPUChunkProcessor")
        .def(py::init<>())
        .def("process_on_gpu", &gpu_chunking::GPUChunkProcessor<double>::process_on_gpu);
#endif

    // Sophisticated Chunking
    py::class_<sophisticated_chunking::WaveletChunking<double>>(m, "WaveletChunking")
        .def(py::init<size_t, double>())
        .def("chunk", &sophisticated_chunking::WaveletChunking<double>::chunk);

    py::class_<sophisticated_chunking::MutualInformationChunking<double>>(m, "MutualInformationChunking")
        .def(py::init<size_t, double>())
        .def("chunk", &sophisticated_chunking::MutualInformationChunking<double>::chunk);

    py::class_<sophisticated_chunking::DTWChunking<double>>(m, "DTWChunking")
        .def(py::init<size_t, double>())
        .def("chunk", &sophisticated_chunking::DTWChunking<double>::chunk);

    // Chunk Metrics
    py::class_<chunk_metrics::ChunkQualityAnalyzer<double>>(m, "ChunkQualityAnalyzer")
        .def(py::init<>())
        .def("compute_cohesion", &chunk_metrics::ChunkQualityAnalyzer<double>::compute_cohesion)
        .def("compute_separation", &chunk_metrics::ChunkQualityAnalyzer<double>::compute_separation)
        .def("compute_silhouette_score", &chunk_metrics::ChunkQualityAnalyzer<double>::compute_silhouette_score)
        .def("compute_quality_score", &chunk_metrics::ChunkQualityAnalyzer<double>::compute_quality_score)
        .def("compute_size_metrics", &chunk_metrics::ChunkQualityAnalyzer<double>::compute_size_metrics)
        .def("clear_cache", &chunk_metrics::ChunkQualityAnalyzer<double>::clear_cache);

    // Chunk Visualization
    py::class_<chunk_viz::ChunkVisualizer<double>>(m, "ChunkVisualizer")
        .def(py::init<std::vector<double>&, const std::string&>())
        .def("plot_chunk_sizes", &chunk_viz::ChunkVisualizer<double>::plot_chunk_sizes)
        .def("visualize_boundaries", &chunk_viz::ChunkVisualizer<double>::visualize_boundaries)
        .def("export_to_graphviz", &chunk_viz::ChunkVisualizer<double>::export_to_graphviz);

    // Chunk Serialization
    py::class_<chunk_serialization::ChunkSerializer<double>>(m, "ChunkSerializer")
        .def(py::init<>())
        .def("to_json", &chunk_serialization::ChunkSerializer<double>::to_json)
        .def("to_protobuf", &chunk_serialization::ChunkSerializer<double>::to_protobuf)
        .def("to_msgpack", &chunk_serialization::ChunkSerializer<double>::to_msgpack);

    // Chunk Resilience
    py::class_<chunk_resilience::ResilientChunker<double>>(m, "ResilientChunker")
        .def(py::init<const std::string&, size_t, size_t, size_t>())
        .def("process", &chunk_resilience::ResilientChunker<double>::process)
        .def("save_checkpoint", &chunk_resilience::ResilientChunker<double>::save_checkpoint)
        .def("restore_from_checkpoint", &chunk_resilience::ResilientChunker<double>::restore_from_checkpoint);

    // Database Integration
    #ifdef HAVE_POSTGRESQL
    py::class_<chunk_integrations::DatabaseChunkStore>(m, "DatabaseChunkStore")
        .def(py::init<std::unique_ptr<chunk_integrations::DatabaseConnection>, const std::string&>())
        .def("store_chunks_postgres", &chunk_integrations::DatabaseChunkStore::store_chunks_postgres<double>)
        #ifdef HAVE_MONGODB
        .def("store_chunks_mongodb", &chunk_integrations::DatabaseChunkStore::store_chunks_mongodb<double>)
        #endif
        ;
    #endif

    // Message Queue Integration
    #if defined(HAVE_KAFKA) || defined(HAVE_RABBITMQ)
    py::class_<chunk_integrations::ChunkMessageQueue>(m, "ChunkMessageQueue")
        .def(py::init<std::unique_ptr<chunk_integrations::MessageQueueConnection>, const std::string&>())
        #ifdef HAVE_KAFKA
        .def("publish_chunks_kafka", &chunk_integrations::ChunkMessageQueue::publish_chunks_kafka<double>)
        #endif
        #ifdef HAVE_RABBITMQ
        .def("publish_chunks_rabbitmq", &chunk_integrations::ChunkMessageQueue::publish_chunks_rabbitmq<double>)
        #endif
        ;
    #endif

    // Chunk Benchmark
    py::class_<chunk_benchmark::ChunkBenchmark<double>>(m, "ChunkBenchmark")
        .def(py::init<std::vector<double>&, const std::string&>())
        .def("run_benchmark", &chunk_benchmark::ChunkBenchmark<double>::run_benchmark)
        .def("save_results", &chunk_benchmark::ChunkBenchmark<double>::save_results)
        .def("measure_throughput", &chunk_benchmark::ChunkBenchmark<double>::measure_throughput)
        .def("measure_memory_usage", &chunk_benchmark::ChunkBenchmark<double>::measure_memory_usage)
        .def("compare_strategies", &chunk_benchmark::ChunkBenchmark<double>::compare_strategies);

    // Add exception translations
    py::register_exception<chunk_resilience::ChunkingError>(m, "ChunkingError");
} 