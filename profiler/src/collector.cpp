#include "collector.h"
#include "metric_writer.h"
// You don't need to redefine these — they're already in nvml.h
// Removed the #defines

//NEED TO ADD GRACEFUL SHUTFDOFWN MAXXING W SIGNALS
using namespace std;

int collectorQuery() {
    auto now = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration<double>(
        now.time_since_epoch()
    ).count();

    // Hostname
    char hostname[256];
    gethostname(hostname, sizeof(hostname));

        // SLURM Job ID
    const char* job_id_env = getenv("SLURM_JOB_ID");
    std::string job_id = job_id_env ? job_id_env : "unknown";

    // PID
    pid_t pid = getpid();


    unsigned int flags = NVML_INIT_FLAG_FORCE_INIT | NVML_INIT_FLAG_NO_GPUS | NVML_INIT_FLAG_NO_ATTACH;
    nvmlReturn_t result = nvmlInitWithFlags(flags);
    unsigned int deviceCount;
    nvmlDeviceGetCount(&deviceCount);


    ofstream outFile("metrics.jsonl", ios::app); // append mode

    double metrics[4];
    // Removed the computerId line — flagging below
    for (unsigned int i = 0; i < deviceCount; i++) {
        nvmlDevice_t device;
        nvmlDeviceGetHandleByIndex(i, &device);

        // OPERATIONS
        char name[256];
        nvmlDeviceGetName(device, name, sizeof(name));


        nvmlUtilization_t gpuUtilization;
        nvmlDeviceGetUtilizationRates(device, &gpuUtilization);
        // gpuUtilization.gpu  → GPU utilization %
        // gpuUtilization.memory → memory controller utilization %

        nvmlMemory_t memInfo;
        nvmlDeviceGetMemoryInfo(device, &memInfo);
        // memInfo.used, memInfo.total, memInfo.free (bytes)

        unsigned int temp;
        nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &temp);

        unsigned int powerMw;
        nvmlDeviceGetPowerUsage(device, &powerMw);
        // returns milliwatts, divide by 1000 for watts

        unsigned int powerLimitMw;
        nvmlDeviceGetPowerManagementLimit(device, &powerLimitMw);
        //COMPUTE SPEED
        unsigned int clockSm;
        nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &clockSm);
        //VRAM
        unsigned int clockMem;
        nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &clockMem);
        // write2 buefr
        
        // get max SM clock for throttle ratio
        unsigned int maxClockSm;
        nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &maxClockSm);

        metrics[0] += (double)gpuUtilization.gpu / deviceCount;
        metrics[1] += (double)temp / deviceCount;
        metrics[2] += ((double)memInfo.used / memInfo.total) / deviceCount;
        metrics[3] += (maxClockSm > 0 ? (double)clockSm / maxClockSm : 1.0) / deviceCount;
    }

    
    std::ostringstream data;
    data << std::fixed << std::setprecision(4);
    data << "{\"avg_gpu_util\":" << metrics[0]
         << ",\"avg_temp_c\":" << metrics[1]
         << ",\"avg_mem_usage_ratio\":" << metrics[2]
         << ",\"avg_clock_throttle_ratio\":" << metrics[3]
         << "}";

    write_metric(envelope("gpu") + data.str() + "}");
    nvmlShutdown();

    return 0;
}