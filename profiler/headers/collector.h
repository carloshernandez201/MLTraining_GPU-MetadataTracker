#pragma once
#include <string>
#include <vector>

// One snapshot of a single GPU. Units are noted per field.
struct GpuSample {
    unsigned int index = 0;
    std::string name;
    unsigned int gpu_util = 0;          // %
    unsigned int mem_util = 0;          // % (memory controller)
    unsigned long long mem_used = 0;    // bytes
    unsigned long long mem_total = 0;   // bytes
    unsigned long long mem_free = 0;    // bytes
    unsigned int temp_c = 0;
    double power_w = 0;
    double power_limit_w = 0;
    unsigned int clock_sm_mhz = 0;
    unsigned int clock_sm_max_mhz = 0;
    unsigned int clock_mem_mhz = 0;
    unsigned int pcie_tx_kbps = 0;
    unsigned int pcie_rx_kbps = 0;
    unsigned int process_count = 0;
};

// Initialise NVML once. Returns false (and prints why) on failure.
bool collector_init();
void collector_shutdown();

// Read every visible GPU into `out`. Returns false on a fatal NVML error.
bool collector_sample(std::vector<GpuSample>& out);

// Build the "data" object for one sample: per-device array + cross-GPU averages.
std::string collector_to_json(const std::vector<GpuSample>& samples);

// Sample once and append a "gpu" record to the metrics log (or stdout).
// Returns 0 on success, non-zero on failure.
int collectorQuery(bool to_stdout = false);
