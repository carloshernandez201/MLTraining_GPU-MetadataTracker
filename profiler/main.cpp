// profiler_daemon — polls GPU metrics on an interval, records Slurm context,
// and applies system-side control commands. Stops on SIGINT/SIGTERM or after
// --duration seconds.
#include "collector.h"
#include "slurm_monitor.h"
#include "system_executor.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>

static std::atomic<bool> g_running{true};

static void handle_signal(int) { g_running = false; }

static void usage(const char* prog) {
    std::cout << "usage: " << prog << " [-i SECONDS] [-d SECONDS] [--slurm-every N] [--stdout]\n"
                 "  -i, --interval     seconds between GPU samples (default 1)\n"
                 "  -d, --duration     stop after this many seconds (default: run until signalled)\n"
                 "  --slurm-every N    refresh Slurm metadata every N samples (default 60, 0 = start only)\n"
                 "  --stdout           print JSONL to stdout instead of logs/<job>/metrics.jsonl\n";
}

int main(int argc, char** argv) {
    double interval = 1.0;
    double duration = -1;  // < 0 => unbounded
    int slurm_every = 60;
    bool to_stdout = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) { usage(argv[0]); std::exit(2); }
            return argv[++i];
        };
        try {
            if (a == "-i" || a == "--interval") interval = std::stod(next());
            else if (a == "-d" || a == "--duration") duration = std::stod(next());
            else if (a == "--slurm-every") slurm_every = std::stoi(next());
            else if (a == "--stdout") to_stdout = true;
            else if (a == "-h" || a == "--help") { usage(argv[0]); return 0; }
            else { usage(argv[0]); return 2; }
        } catch (const std::exception&) {
            std::cerr << "invalid value for " << a << std::endl;
            return 2;
        }
    }
    if (interval <= 0) {
        std::cerr << "interval must be > 0" << std::endl;
        return 2;
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    if (!collector_init()) return 1;

    // Slurm context is best-effort: the collector still runs outside a job.
    slurm_query(to_stdout);

    auto start = std::chrono::steady_clock::now();
    auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>(interval));
    auto next_tick = start;
    long samples = 0;

    while (g_running) {
        if (duration >= 0 &&
            std::chrono::steady_clock::now() - start >= std::chrono::duration<double>(duration)) {
            break;
        }

        if (collectorQuery(to_stdout) != 0) {
            std::cerr << "[daemon] collector failed, exiting" << std::endl;
            break;
        }
        samples++;

        check_system_commands();

        if (slurm_every > 0 && samples % slurm_every == 0) slurm_query(to_stdout);

        // Sleep in short slices so a signal stops us promptly.
        next_tick += period;
        while (g_running && std::chrono::steady_clock::now() < next_tick) {
            auto remaining = next_tick - std::chrono::steady_clock::now();
            std::this_thread::sleep_for(std::min<std::chrono::steady_clock::duration>(
                remaining, std::chrono::milliseconds(100)));
        }
    }

    collector_shutdown();
    std::cerr << "[daemon] stopped after " << samples << " samples" << std::endl;
    return 0;
}
