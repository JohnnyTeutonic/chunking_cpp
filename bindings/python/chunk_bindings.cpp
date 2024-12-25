#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "chunk.hpp"
#include "chunk_benchmark.hpp"
#include "chunk_compression.hpp"
#include "chunk_integrations.hpp"
#include "chunk_metrics.hpp"
#include "chunk_resilience.hpp"
#include "chunk_serialization.hpp"
#include "chunk_strategies.hpp"
#include "chunk_strategy_implementations.hpp"
#include "chunk_visualization.hpp"
#include "neural_chunking.hpp"
#ifdef HAVE_CUDA
#include "gpu_chunking.hpp"
#endif
#include "sophisticated_chunking.hpp"

namespace py = pybind11;

PYBIND11_MODULE(chunking_cpp, m) {
    m.doc() = "Python bindings for the C++ chunking library";

    // Register exception translators
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p)
                std::rethrow_exception(p);
        } catch (const std::invalid_argument& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
        } catch (const std::runtime_error& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });

    // Basic Chunking
    py::class_<chunk_processing::Chunk<double>>(m, "Chunk")
        .def(py::init<size_t>())
        .def("add",
             static_cast<void (chunk_processing::Chunk<double>::*)(const double&)>(
                 &chunk_processing::Chunk<double>::add),
             "Add a single element")
        .def(
            "add",
            [](chunk_processing::Chunk<double>& self, const std::vector<double>& data) {
                if (data.empty()) {
                    throw std::invalid_argument("Cannot add empty vector");
                }
                self.add(data);
            },
            "Add multiple elements")
        .def("chunk_by_size", &chunk_processing::Chunk<double>::chunk_by_size)
        .def("chunk_by_threshold", &chunk_processing::Chunk<double>::chunk_by_threshold)
        .def("get_chunks", &chunk_processing::Chunk<double>::get_chunks);

    py::class_<chunk_processing::Chunk<std::vector<double>>>(m, "Chunk2D")
        .def(py::init<size_t>())
        .def("add", [](chunk_processing::Chunk<std::vector<double>>& self,
                       const py::array_t<double, py::array::c_style>& data) {
            auto buf = data.request();
            if (buf.ndim != 2)
                throw std::invalid_argument("Expected 2D array");

            std::vector<std::vector<double>> nested_data;
            auto ptr = static_cast<double*>(buf.ptr);

            for (py::ssize_t i = 0; i < buf.shape[0]; i++) {
                std::vector<double> row;
                for (py::ssize_t j = 0; j < buf.shape[1]; j++) {
                    row.push_back(ptr[i * buf.shape[1] + j]);
                }
                nested_data.push_back(row);
            }
            self.add(nested_data);
        });

    py::class_<chunk_processing::Chunk<std::vector<std::vector<double>>>>(m, "Chunk3D")
        .def(py::init<size_t>())
        .def("add", [](chunk_processing::Chunk<std::vector<std::vector<double>>>& self,
                       const py::array_t<double, py::array::c_style>& data) {
            auto buf = data.request();
            if (buf.ndim != 3)
                throw std::invalid_argument("Expected 3D array");

            // Convert 3D numpy array to nested vectors
            // Implementation here
        });

    // Neural Chunking
    py::class_<neural_chunking::NeuralChunking<double>>(m, "NeuralChunking")
        .def(py::init<size_t, double>())
        .def("chunk",
             [](neural_chunking::NeuralChunking<double>& self, const std::vector<double>& data) {
                 auto chunks = self.chunk(data);
                 py::list result;
                 for (const auto& chunk : chunks) {
                     result.append(py::array_t<double>(chunk.size(), chunk.data()));
                 }
                 return result;
             })
        .def("get_window_size", &neural_chunking::NeuralChunking<double>::get_window_size)
        .def("get_threshold", &neural_chunking::NeuralChunking<double>::get_threshold)
        .def("set_window_size", &neural_chunking::NeuralChunking<double>::set_window_size)
        .def("set_threshold", &neural_chunking::NeuralChunking<double>::set_threshold);

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

    py::class_<sophisticated_chunking::MutualInformationChunking<double>>(
        m, "MutualInformationChunking")
        .def(py::init<size_t, double>())
        .def("chunk", &sophisticated_chunking::MutualInformationChunking<double>::chunk);

    py::class_<sophisticated_chunking::DTWChunking<double>>(m, "DTWChunking")
        .def(py::init<size_t, double>())
        .def("chunk", &sophisticated_chunking::DTWChunking<double>::chunk)
        .def("set_window_size", &sophisticated_chunking::DTWChunking<double>::set_window_size)
        .def("get_window_size", &sophisticated_chunking::DTWChunking<double>::get_window_size)
        .def("set_dtw_threshold", &sophisticated_chunking::DTWChunking<double>::set_dtw_threshold)
        .def("get_dtw_threshold", &sophisticated_chunking::DTWChunking<double>::get_dtw_threshold);

    // Chunk Metrics
    py::class_<chunk_metrics::ChunkQualityAnalyzer<double>>(m, "ChunkQualityAnalyzer")
        .def(py::init<>())
        .def("compute_cohesion", &chunk_metrics::ChunkQualityAnalyzer<double>::compute_cohesion)
        .def("compute_separation", &chunk_metrics::ChunkQualityAnalyzer<double>::compute_separation)
        .def("compute_silhouette_score",
             &chunk_metrics::ChunkQualityAnalyzer<double>::compute_silhouette_score)
        .def("compute_quality_score",
             &chunk_metrics::ChunkQualityAnalyzer<double>::compute_quality_score)
        .def("compute_size_metrics",
             &chunk_metrics::ChunkQualityAnalyzer<double>::compute_size_metrics)
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
        .def("restore_from_checkpoint",
             &chunk_resilience::ResilientChunker<double>::restore_from_checkpoint);

// Database Integration
#ifdef HAVE_POSTGRESQL
    py::class_<chunk_integrations::DatabaseChunkStore>(m, "DatabaseChunkStore")
        .def(
            py::init<std::unique_ptr<chunk_integrations::DatabaseConnection>, const std::string&>())
        .def("store_chunks_postgres",
             &chunk_integrations::DatabaseChunkStore::store_chunks_postgres<double>)
#ifdef HAVE_MONGODB
        .def("store_chunks_mongodb",
             &chunk_integrations::DatabaseChunkStore::store_chunks_mongodb<double>)
#endif
        ;
#endif

// Message Queue Integration
#if defined(HAVE_KAFKA) || defined(HAVE_RABBITMQ)
    py::class_<chunk_integrations::ChunkMessageQueue>(m, "ChunkMessageQueue")
        .def(py::init<std::unique_ptr<chunk_integrations::MessageQueueConnection>,
                      const std::string&>())
#ifdef HAVE_KAFKA
        .def("publish_chunks_kafka",
             &chunk_integrations::ChunkMessageQueue::publish_chunks_kafka<double>)
#endif
#ifdef HAVE_RABBITMQ
        .def("publish_chunks_rabbitmq",
             &chunk_integrations::ChunkMessageQueue::publish_chunks_rabbitmq<double>)
#endif
        ;
#endif

    // Benchmark bindings
    py::class_<chunk_benchmark::BenchmarkResult>(m, "BenchmarkResult")
        .def_readwrite("execution_time_ms", &chunk_benchmark::BenchmarkResult::execution_time_ms)
        .def_readwrite("memory_usage_bytes", &chunk_benchmark::BenchmarkResult::memory_usage_bytes)
        .def_readwrite("num_chunks", &chunk_benchmark::BenchmarkResult::num_chunks)
        .def_readwrite("strategy_name", &chunk_benchmark::BenchmarkResult::strategy_name);

    py::class_<chunk_benchmark::ChunkBenchmark<double>>(m, "ChunkBenchmark")
        .def(py::init<const std::vector<double>&, const std::string&>())
        .def("add_strategy", &chunk_benchmark::ChunkBenchmark<double>::add_strategy)
        .def("run_benchmark", &chunk_benchmark::ChunkBenchmark<double>::run_benchmark)
        .def("benchmark_chunking", &chunk_benchmark::ChunkBenchmark<double>::benchmark_chunking)
        .def("save_results", &chunk_benchmark::ChunkBenchmark<double>::save_results);

    // Add exception translations
    py::register_exception<chunk_processing::ChunkingError>(m, "ChunkingError");

    // Strategy bindings
    py::class_<chunk_strategies::ChunkStrategy<double>,
               std::shared_ptr<chunk_strategies::ChunkStrategy<double>>>(m, "ChunkStrategy")
        .def("apply", &chunk_strategies::ChunkStrategy<double>::apply);

    py::class_<NeuralChunkingStrategy<double>, chunk_strategies::ChunkStrategy<double>,
               std::shared_ptr<NeuralChunkingStrategy<double>>>(m, "NeuralChunkingStrategy")
        .def(py::init<>())
        .def("apply", &NeuralChunkingStrategy<double>::apply);

    py::class_<SimilarityChunkingStrategy<double>, chunk_strategies::ChunkStrategy<double>,
               std::shared_ptr<SimilarityChunkingStrategy<double>>>(m, "SimilarityChunkingStrategy")
        .def(py::init<double>())
        .def("apply", &SimilarityChunkingStrategy<double>::apply);
}