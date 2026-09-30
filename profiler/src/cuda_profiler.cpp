// CUDA kernel profiler (CUDA Events) + standalone demo + optional pybind11 module.
//
// Demo:   nvcc -O2 -std=c++17 -x cu -Iheaders src/cuda_profiler.cpp -o cuda_profiler_demo
// Python: add -shared -Xcompiler -fPIC -DBUILD_PYBIND $(python3 -m pybind11 --includes)
#include "cuda_profiler.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

void cuda_check(cudaError_t err, const char* what, const char* file, int line) {
    if (err != cudaSuccess) {
        std::ostringstream o;
        o << what << " failed at " << file << ":" << line << ": " << cudaGetErrorString(err);
        throw std::runtime_error(o.str());
    }
}

static double now_us() {
    return std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static std::string escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        o += c;
    }
    return o;
}

CudaProfiler::CudaProfiler(cudaStream_t stream) : stream_(stream) {}

CudaProfiler::~CudaProfiler() {
    for (auto& kv : timers_) {
        if (kv.second.start) cudaEventDestroy(kv.second.start);
        if (kv.second.stop) cudaEventDestroy(kv.second.stop);
    }
}

void CudaProfiler::start(const std::string& name) {
    Timer& t = timers_[name];
    if (!t.start) {
        CUDA_CHECK(cudaEventCreate(&t.start));
        CUDA_CHECK(cudaEventCreate(&t.stop));
    }
    if (t.active) throw std::runtime_error("CudaProfiler::start called twice for '" + name + "'");
    t.active = true;
    t.cpu_start_us = now_us();
    CUDA_CHECK(cudaEventRecord(t.start, stream_));
}

float CudaProfiler::stop(const std::string& name, double flops, double bytes) {
    auto it = timers_.find(name);
    if (it == timers_.end() || !it->second.active)
        throw std::runtime_error("CudaProfiler::stop without start for '" + name + "'");
    Timer& t = it->second;

    CUDA_CHECK(cudaEventRecord(t.stop, stream_));
    CUDA_CHECK(cudaEventSynchronize(t.stop));
    double cpu_us = now_us() - t.cpu_start_us;
    t.active = false;

    float gpu_ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, t.start, t.stop));

    Record& r = records_[name];
    r.gpu_ms.push_back(gpu_ms);
    r.overhead_us.push_back(std::max(0.0, cpu_us - gpu_ms * 1000.0));
    if (flops > 0) r.flops = flops;
    if (bytes > 0) r.bytes = bytes;
    return gpu_ms;
}

float CudaProfiler::time(const std::string& name, const std::function<void()>& launch,
                         double flops, double bytes) {
    start(name);
    launch();
    CUDA_CHECK(cudaGetLastError());  // catch bad launch configs
    return stop(name, flops, bytes);
}

static double percentile(std::vector<float> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    double idx = p / 100.0 * (v.size() - 1);
    size_t lo = (size_t)std::floor(idx), hi = (size_t)std::ceil(idx);
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

KernelStats CudaProfiler::stats(const std::string& name) const {
    KernelStats s;
    s.name = name;
    auto it = records_.find(name);
    if (it == records_.end() || it->second.gpu_ms.empty()) return s;
    const Record& r = it->second;

    s.count = r.gpu_ms.size();
    s.min_ms = *std::min_element(r.gpu_ms.begin(), r.gpu_ms.end());
    s.max_ms = *std::max_element(r.gpu_ms.begin(), r.gpu_ms.end());
    for (float x : r.gpu_ms) s.total_ms += x;
    s.mean_ms = s.total_ms / s.count;
    double var = 0;
    for (float x : r.gpu_ms) var += (x - s.mean_ms) * (x - s.mean_ms);
    s.std_ms = std::sqrt(var / s.count);
    s.p50_ms = percentile(r.gpu_ms, 50);
    s.p95_ms = percentile(r.gpu_ms, 95);
    s.p99_ms = percentile(r.gpu_ms, 99);

    double mean_s = s.mean_ms / 1000.0;
    if (mean_s > 0) {
        s.gflops = r.flops / mean_s / 1e9;
        s.bandwidth_gbs = r.bytes / mean_s / 1e9;
    }
    double oh = 0;
    for (double x : r.overhead_us) oh += x;
    s.mean_launch_overhead_us = oh / r.overhead_us.size();
    return s;
}

std::vector<KernelStats> CudaProfiler::all_stats() const {
    std::vector<KernelStats> out;
    for (const auto& kv : records_) out.push_back(stats(kv.first));
    std::sort(out.begin(), out.end(),
              [](const KernelStats& a, const KernelStats& b) { return a.total_ms > b.total_ms; });
    return out;
}

double CudaProfiler::peak_bandwidth_gbs(int device) {
    int mem_clock_khz = 0, bus_width_bits = 0;
    if (cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device) != cudaSuccess ||
        cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, device) != cudaSuccess)
        return 0;
    // x2 for double data rate
    return 2.0 * mem_clock_khz * 1e3 * (bus_width_bits / 8.0) / 1e9;
}

std::string CudaProfiler::to_json() const {
    std::ostringstream o;
    o << std::fixed << std::setprecision(4);
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, dev);

    o << "{\"device\":\"" << escape(prop.name) << "\""
      << ",\"peak_bandwidth_gbs\":" << peak_bandwidth_gbs(dev)
      << ",\"kernels\":[";
    bool first = true;
    for (const KernelStats& s : all_stats()) {
        if (!first) o << ",";
        first = false;
        o << "{\"name\":\"" << escape(s.name) << "\""
          << ",\"count\":" << s.count
          << ",\"total_ms\":" << s.total_ms
          << ",\"mean_ms\":" << s.mean_ms
          << ",\"min_ms\":" << s.min_ms
          << ",\"max_ms\":" << s.max_ms
          << ",\"std_ms\":" << s.std_ms
          << ",\"p50_ms\":" << s.p50_ms
          << ",\"p95_ms\":" << s.p95_ms
          << ",\"p99_ms\":" << s.p99_ms
          << ",\"gflops\":" << s.gflops
          << ",\"bandwidth_gbs\":" << s.bandwidth_gbs
          << ",\"mean_launch_overhead_us\":" << s.mean_launch_overhead_us
          << "}";
    }
    o << "]}";
    return o.str();
}

bool CudaProfiler::export_json(const std::string& path) const {
    std::ofstream f(path);
    if (!f) return false;
    f << to_json() << "\n";
    return true;
}

void CudaProfiler::reset() { records_.clear(); }

// ===========================================================================
// Demo: a memory-bound kernel and two compute-bound kernels of different quality.
// ===========================================================================
#if !defined(BUILD_PYBIND) && !defined(CUDA_PROFILER_NO_MAIN)

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void matmul_naive(const float* A, const float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n) {
        float acc = 0;
        for (int k = 0; k < n; k++) acc += A[row * n + k] * B[k * n + col];
        C[row * n + col] = acc;
    }
}

constexpr int TILE = 16;
__global__ void matmul_tiled(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0;
    for (int t = 0; t < (n + TILE - 1) / TILE; t++) {
        int ac = t * TILE + threadIdx.x, br = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < n && ac < n) ? A[row * n + ac] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (br < n && col < n) ? B[br * n + col] : 0.f;
        __syncthreads();
        for (int k = 0; k < TILE; k++) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < n && col < n) C[row * n + col] = acc;
}

int main(int argc, char** argv) {
    const char* out_path = argc > 1 ? argv[1] : "cuda_kernels.json";
    const int iters = 50;

    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        std::cerr << "no CUDA device available" << std::endl;
        return 1;
    }

    try {
        CudaProfiler prof;

        // --- vector add: 1 FLOP and 12 bytes per element -> memory bound
        const int n = 1 << 24;
        float *a, *b, *c;
        CUDA_CHECK(cudaMalloc(&a, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b, n * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&c, n * sizeof(float)));
        CUDA_CHECK(cudaMemset(a, 0, n * sizeof(float)));
        CUDA_CHECK(cudaMemset(b, 0, n * sizeof(float)));
        int threads = 256, blocks = (n + threads - 1) / threads;
        vector_add<<<blocks, threads>>>(a, b, c, n);  // warm-up
        for (int i = 0; i < iters; i++) {
            prof.time("vector_add", [&] { vector_add<<<blocks, threads>>>(a, b, c, n); },
                      (double)n, 3.0 * n * sizeof(float));
        }

        // --- matmul: 2*N^3 FLOPs
        const int m = 1024;
        float *A, *B, *C;
        CUDA_CHECK(cudaMalloc(&A, m * m * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&B, m * m * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&C, m * m * sizeof(float)));
        CUDA_CHECK(cudaMemset(A, 0, m * m * sizeof(float)));
        CUDA_CHECK(cudaMemset(B, 0, m * m * sizeof(float)));
        dim3 blk(TILE, TILE), grd((m + TILE - 1) / TILE, (m + TILE - 1) / TILE);
        double mm_flops = 2.0 * m * m * m, mm_bytes = 3.0 * m * m * sizeof(float);
        matmul_naive<<<grd, blk>>>(A, B, C, m);
        matmul_tiled<<<grd, blk>>>(A, B, C, m);
        for (int i = 0; i < iters; i++) {
            prof.time("matmul_naive", [&] { matmul_naive<<<grd, blk>>>(A, B, C, m); }, mm_flops, mm_bytes);
            prof.time("matmul_tiled", [&] { matmul_tiled<<<grd, blk>>>(A, B, C, m); }, mm_flops, mm_bytes);
        }

        double peak = CudaProfiler::peak_bandwidth_gbs();
        std::cout << std::fixed << std::setprecision(3)
                  << "peak DRAM bandwidth: " << peak << " GB/s\n\n"
                  << std::left << std::setw(14) << "kernel" << std::right
                  << std::setw(10) << "mean ms" << std::setw(10) << "p95 ms"
                  << std::setw(11) << "GFLOP/s" << std::setw(10) << "GB/s"
                  << std::setw(10) << "%peakBW" << std::setw(13) << "launch us" << "\n";
        for (const KernelStats& s : prof.all_stats()) {
            std::cout << std::left << std::setw(14) << s.name << std::right
                      << std::setw(10) << s.mean_ms << std::setw(10) << s.p95_ms
                      << std::setw(11) << s.gflops << std::setw(10) << s.bandwidth_gbs
                      << std::setw(9) << (peak > 0 ? 100.0 * s.bandwidth_gbs / peak : 0) << "%"
                      << std::setw(13) << s.mean_launch_overhead_us << "\n";
        }

        if (prof.export_json(out_path)) std::cout << "\nwrote " << out_path << std::endl;

        cudaFree(a); cudaFree(b); cudaFree(c);
        cudaFree(A); cudaFree(B); cudaFree(C);
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    return 0;
}
#endif

// ===========================================================================
// Python bindings: bracket arbitrary GPU work (e.g. PyTorch ops on the
// default stream) with start()/stop().
// ===========================================================================
#ifdef BUILD_PYBIND
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

PYBIND11_MODULE(cuda_profiler, m) {
    m.doc() = "CUDA Event based kernel timer";
    py::class_<KernelStats>(m, "KernelStats")
        .def_readonly("name", &KernelStats::name)
        .def_readonly("count", &KernelStats::count)
        .def_readonly("mean_ms", &KernelStats::mean_ms)
        .def_readonly("min_ms", &KernelStats::min_ms)
        .def_readonly("max_ms", &KernelStats::max_ms)
        .def_readonly("std_ms", &KernelStats::std_ms)
        .def_readonly("p50_ms", &KernelStats::p50_ms)
        .def_readonly("p95_ms", &KernelStats::p95_ms)
        .def_readonly("p99_ms", &KernelStats::p99_ms)
        .def_readonly("total_ms", &KernelStats::total_ms)
        .def_readonly("gflops", &KernelStats::gflops)
        .def_readonly("bandwidth_gbs", &KernelStats::bandwidth_gbs)
        .def_readonly("mean_launch_overhead_us", &KernelStats::mean_launch_overhead_us);

    py::class_<CudaProfiler>(m, "CudaProfiler")
        .def(py::init<>())
        .def("start", &CudaProfiler::start, py::arg("name"))
        .def("stop", &CudaProfiler::stop, py::arg("name"), py::arg("flops") = 0, py::arg("bytes") = 0)
        .def("stats", &CudaProfiler::stats)
        .def("all_stats", &CudaProfiler::all_stats)
        .def("to_json", &CudaProfiler::to_json)
        .def("export_json", &CudaProfiler::export_json)
        .def("reset", &CudaProfiler::reset)
        .def_static("peak_bandwidth_gbs", &CudaProfiler::peak_bandwidth_gbs, py::arg("device") = 0);
}
#endif
