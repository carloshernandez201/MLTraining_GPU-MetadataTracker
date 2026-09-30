#pragma once
// CUDA kernel timer built on CUDA Events.
//
//   CudaProfiler prof;
//   prof.time("matmul", [&]{ matmul<<<grid, block>>>(...); }, flops, bytes);
//   prof.export_json("kernels.json");
//
// For every named kernel it tracks count / mean / min / max / std / p50 / p95 / p99
// of GPU execution time, achieved GFLOP/s and GB/s, and launch overhead
// (CPU wall time of the whole call minus GPU execution time).
#include <cuda_runtime.h>

#include <functional>
#include <map>
#include <string>
#include <vector>

struct KernelStats {
    std::string name;
    size_t count = 0;
    double mean_ms = 0, min_ms = 0, max_ms = 0, std_ms = 0;
    double p50_ms = 0, p95_ms = 0, p99_ms = 0;
    double total_ms = 0;
    double gflops = 0;               // achieved, based on mean time (0 if flops unknown)
    double bandwidth_gbs = 0;        // achieved, based on mean time (0 if bytes unknown)
    double mean_launch_overhead_us = 0;
};

class CudaProfiler {
public:
    explicit CudaProfiler(cudaStream_t stream = 0);
    ~CudaProfiler();
    CudaProfiler(const CudaProfiler&) = delete;
    CudaProfiler& operator=(const CudaProfiler&) = delete;

    // Time one launch. `flops` / `bytes` are the work done by a single call.
    // Returns GPU time in milliseconds.
    float time(const std::string& name, const std::function<void()>& launch,
               double flops = 0, double bytes = 0);

    // Manual bracket API, for code you can't wrap in a lambda (e.g. from Python).
    void start(const std::string& name);
    float stop(const std::string& name, double flops = 0, double bytes = 0);

    KernelStats stats(const std::string& name) const;
    std::vector<KernelStats> all_stats() const;
    std::string to_json() const;
    bool export_json(const std::string& path) const;
    void reset();

    // Theoretical peak DRAM bandwidth of a device, for roofline context.
    static double peak_bandwidth_gbs(int device = 0);

private:
    struct Record {
        std::vector<float> gpu_ms;
        std::vector<double> overhead_us;
        double flops = 0;
        double bytes = 0;
    };
    struct Timer {
        cudaEvent_t start = nullptr, stop = nullptr;
        double cpu_start_us = 0;
        bool active = false;
    };

    cudaStream_t stream_;
    std::map<std::string, Record> records_;
    std::map<std::string, Timer> timers_;   // events are created once per name and reused
};

// Throws std::runtime_error with file/line on a CUDA error.
void cuda_check(cudaError_t err, const char* what, const char* file, int line);
#define CUDA_CHECK(x) cuda_check((x), #x, __FILE__, __LINE__)
