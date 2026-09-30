#include "collector.h"
#include "metric_writer.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <cmath>
#include <chrono>

#ifdef PROFILER_MOCK_NVML
// ---------------------------------------------------------------------------
// Mock backend: synthetic but plausible numbers so the whole pipeline can be
// built and exercised on machines without an NVIDIA driver (laptops, CI).
// ---------------------------------------------------------------------------
bool collector_init() { return true; }
void collector_shutdown() {}

bool collector_sample(std::vector<GpuSample>& out) {
    out.clear();
    double t = std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    for (unsigned int i = 0; i < 2; i++) {
        GpuSample s;
        s.index = i;
        s.name = "Mock GPU";
        s.gpu_util = (unsigned int)(70 + 25 * std::sin(t / 5.0 + i));
        s.mem_util = s.gpu_util / 2;
        s.mem_total = 80ULL << 30;
        s.mem_used = (unsigned long long)(s.mem_total * 0.6);
        s.mem_free = s.mem_total - s.mem_used;
        s.temp_c = 55 + s.gpu_util / 5;
        s.power_limit_w = 400;
        s.power_w = 100 + 2.8 * s.gpu_util;
        s.clock_sm_max_mhz = 1410;
        s.clock_sm_mhz = 1200 + s.gpu_util;
        s.clock_mem_mhz = 1593;
        s.pcie_tx_kbps = 1000 * s.gpu_util;
        s.pcie_rx_kbps = 2000 * s.gpu_util;
        s.process_count = 1;
        out.push_back(s);
    }
    return true;
}

#else
// ---------------------------------------------------------------------------
// Real backend: NVML (the library nvidia-smi is built on).
// ---------------------------------------------------------------------------
#include <nvml.h>

bool collector_init() {
    nvmlReturn_t r = nvmlInit_v2();
    if (r != NVML_SUCCESS) {
        std::cerr << "[collector] nvmlInit failed: " << nvmlErrorString(r) << std::endl;
        return false;
    }
    return true;
}

void collector_shutdown() { nvmlShutdown(); }

bool collector_sample(std::vector<GpuSample>& out) {
    out.clear();
    unsigned int deviceCount = 0;
    nvmlReturn_t r = nvmlDeviceGetCount_v2(&deviceCount);
    if (r != NVML_SUCCESS) {
        std::cerr << "[collector] nvmlDeviceGetCount failed: " << nvmlErrorString(r) << std::endl;
        return false;
    }

    for (unsigned int i = 0; i < deviceCount; i++) {
        nvmlDevice_t device;
        if (nvmlDeviceGetHandleByIndex_v2(i, &device) != NVML_SUCCESS) continue;

        GpuSample s;
        s.index = i;

        char name[NVML_DEVICE_NAME_V2_BUFFER_SIZE] = {0};
        if (nvmlDeviceGetName(device, name, sizeof(name)) == NVML_SUCCESS) s.name = name;

        // Individual queries can fail (e.g. unsupported on a given SKU);
        // in that case the field keeps its zero default.
        nvmlUtilization_t util;
        if (nvmlDeviceGetUtilizationRates(device, &util) == NVML_SUCCESS) {
            s.gpu_util = util.gpu;
            s.mem_util = util.memory;
        }

        nvmlMemory_t mem;
        if (nvmlDeviceGetMemoryInfo(device, &mem) == NVML_SUCCESS) {
            s.mem_used = mem.used;
            s.mem_total = mem.total;
            s.mem_free = mem.free;
        }

        nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &s.temp_c);

        unsigned int mw = 0;
        if (nvmlDeviceGetPowerUsage(device, &mw) == NVML_SUCCESS) s.power_w = mw / 1000.0;
        if (nvmlDeviceGetPowerManagementLimit(device, &mw) == NVML_SUCCESS) s.power_limit_w = mw / 1000.0;

        nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &s.clock_sm_mhz);
        nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &s.clock_sm_max_mhz);
        nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &s.clock_mem_mhz);

        nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_TX_BYTES, &s.pcie_tx_kbps);
        nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_RX_BYTES, &s.pcie_rx_kbps);

        // Passing a zero-sized buffer returns the count via INSUFFICIENT_SIZE.
        unsigned int procCount = 0;
        r = nvmlDeviceGetComputeRunningProcesses(device, &procCount, nullptr);
        if (r == NVML_SUCCESS || r == NVML_ERROR_INSUFFICIENT_SIZE) s.process_count = procCount;

        out.push_back(s);
    }
    return true;
}
#endif

std::string collector_to_json(const std::vector<GpuSample>& samples) {
    std::ostringstream data;
    data << std::fixed << std::setprecision(4);

    double avg_util = 0, avg_temp = 0, avg_mem = 0, avg_clock = 0, total_power = 0;
    size_t n = samples.size();

    data << "{\"gpu_count\":" << n << ",\"devices\":[";
    for (size_t i = 0; i < n; i++) {
        const GpuSample& s = samples[i];
        double mem_ratio = s.mem_total ? (double)s.mem_used / s.mem_total : 0.0;
        double clock_ratio = s.clock_sm_max_mhz ? (double)s.clock_sm_mhz / s.clock_sm_max_mhz : 1.0;

        if (i) data << ",";
        data << "{\"index\":" << s.index
             << ",\"name\":\"" << json_escape(s.name) << "\""
             << ",\"gpu_util\":" << s.gpu_util
             << ",\"mem_util\":" << s.mem_util
             << ",\"mem_used\":" << s.mem_used
             << ",\"mem_total\":" << s.mem_total
             << ",\"mem_free\":" << s.mem_free
             << ",\"temp_c\":" << s.temp_c
             << ",\"power_w\":" << s.power_w
             << ",\"power_limit_w\":" << s.power_limit_w
             << ",\"clock_sm_mhz\":" << s.clock_sm_mhz
             << ",\"clock_sm_max_mhz\":" << s.clock_sm_max_mhz
             << ",\"clock_mem_mhz\":" << s.clock_mem_mhz
             << ",\"pcie_tx_kbps\":" << s.pcie_tx_kbps
             << ",\"pcie_rx_kbps\":" << s.pcie_rx_kbps
             << ",\"process_count\":" << s.process_count
             << "}";

        avg_util += s.gpu_util;
        avg_temp += s.temp_c;
        avg_mem += mem_ratio;
        avg_clock += clock_ratio;
        total_power += s.power_w;
    }
    if (n) {
        avg_util /= n; avg_temp /= n; avg_mem /= n; avg_clock /= n;
    }

    data << "],\"avg_gpu_util\":" << avg_util
         << ",\"avg_temp_c\":" << avg_temp
         << ",\"avg_mem_usage_ratio\":" << avg_mem
         << ",\"avg_clock_throttle_ratio\":" << avg_clock
         << ",\"total_power_w\":" << total_power
         << "}";
    return data.str();
}

int collectorQuery(bool to_stdout) {
    std::vector<GpuSample> samples;
    if (!collector_sample(samples)) return 1;

    std::string line = envelope("gpu") + collector_to_json(samples) + "}";
    if (to_stdout) std::cout << line << std::endl;
    else write_metric(line);
    return 0;
}
