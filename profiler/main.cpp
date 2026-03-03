#include "collector.h"
#include "slurm_monitor.h"
#include <chrono>


int main(int argc, char* argv){
    int interval = argv[1]
    int duration = argc > 2 ? argv[2] : 9999

    
    int slurmREs = slurmQuery();
    const std::chrono::seconds DURATION_SECONDS(duration);

    auto start_time = std::chrono::steady_clock::now();
    
    // Calculate the time point at which the loop should finish
    auto finish_time = start_time + DURATION_SECONDS;
    while (finish_time > start_time){
        int collectorRes = collectorQuery();
        if (collectorRes != 0){
            break;
        }

        std::this_thread::sleep_for(std::chrono::seconds(interval_sec));
    }
    return 0;
}