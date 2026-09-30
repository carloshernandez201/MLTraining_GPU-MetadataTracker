#pragma once
#include <string>
#include <fstream>
#include <iterator>
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <iostream>
#include <unistd.h>
#ifndef PROFILER_MOCK_NVML
#include <nvml.h>
#endif
#include "metric_writer.h"

// Reads logs/<job>/control.json and executes *system-side* actions
// (reduce_power_limit, drain_node). Training-side actions (set_lr, ...) are
// left in place for the Python ControlHandler to consume.
// Call this in the main polling loop between collector + slurm queries.
inline void check_system_commands() {
    std::string path = get_log_dir() + "/control.json";

    std::ifstream f(path);
    if (!f.is_open()) return;  // no command waiting — fast path

    std::string raw((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    f.close();

    // Minimal JSON field lookup — control.json is a small flat object.
    auto find_string = [&](const std::string& key) -> std::string {
        std::string search = "\"" + key + "\"";
        size_t pos = raw.find(search);
        if (pos == std::string::npos) return "";
        pos = raw.find(':', pos + search.size());
        if (pos == std::string::npos) return "";
        pos = raw.find('"', pos);
        if (pos == std::string::npos) return "";
        pos++;
        size_t end = raw.find('"', pos);
        if (end == std::string::npos) return "";
        return raw.substr(pos, end - pos);
    };

    auto find_int = [&](const std::string& key) -> int {
        std::string search = "\"" + key + "\"";
        size_t pos = raw.find(search);
        if (pos == std::string::npos) return -1;
        pos = raw.find(':', pos + search.size());
        if (pos == std::string::npos) return -1;
        pos++;
        while (pos < raw.size() && (raw[pos] == ' ' || raw[pos] == '\t')) pos++;
        return std::atoi(raw.c_str() + pos);
    };

    std::string action = find_string("action");
    if (action != "reduce_power_limit" && action != "drain_node") return;  // not ours

    std::remove(path.c_str());  // consume it
    std::cout << "[executor] action: " << action << std::endl;

    if (action == "reduce_power_limit") {
        int gpu_idx = find_int("gpu_index");
        int watts = find_int("watts");
        if (gpu_idx < 0 || watts <= 0) {
            std::cerr << "[executor] reduce_power_limit needs gpu_index and watts" << std::endl;
            return;
        }
#ifdef PROFILER_MOCK_NVML
        std::cout << "[executor] (mock) would set GPU " << gpu_idx
                  << " power limit to " << watts << "W" << std::endl;
#else
        // Requires root / admin privileges on the node.
        nvmlDevice_t device;
        nvmlReturn_t r = nvmlDeviceGetHandleByIndex_v2(gpu_idx, &device);
        if (r == NVML_SUCCESS) r = nvmlDeviceSetPowerManagementLimit(device, watts * 1000);
        std::cout << "[executor] set GPU " << gpu_idx << " power limit to " << watts << "W: "
                  << (r == NVML_SUCCESS ? "ok" : nvmlErrorString(r)) << std::endl;
#endif
    } else if (action == "drain_node") {
        // Reason goes into a shell command: keep only safe characters.
        std::string reason;
        for (char c : find_string("reason")) {
            if (std::isalnum((unsigned char)c) || c == ' ' || c == '-' || c == '_' || c == '.') reason += c;
        }
        if (reason.empty()) reason = "profiler";
        char hostname[256] = {0};
        gethostname(hostname, sizeof(hostname) - 1);
        std::string cmd = "scontrol update NodeName=" + std::string(hostname)
                        + " State=DRAIN Reason=\"" + reason + "\"";
        std::cout << "[executor] running: " << cmd << std::endl;
        int rc = std::system(cmd.c_str());
        if (rc != 0) std::cerr << "[executor] scontrol exited with " << rc << std::endl;
    }
}
