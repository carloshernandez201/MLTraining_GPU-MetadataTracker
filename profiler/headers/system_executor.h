#pragma once
#include <string>
#include <fstream>
#include <cstdlib>
#include <nvml.h>
#include <iostream>

// reads control.json, executes system-side actions, deletes the file.
// call this in your main polling loop between collector + slurm queries.
inline void check_system_commands(const std::string& job_id) {
    std::string path = "logs/" + job_id + "/control.json";

    std::ifstream f(path);
    if (!f.is_open()) return;  // no command waiting — fast path

    // slurp the whole file
    std::string raw((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    f.close();
    std::remove(path.c_str());  // consume it

    // ghetto JSON parsing — find the action field
    // (for a real project you'd use nlohmann/json, but this works)
    auto find_string = [&](const std::string& key) -> std::string {
        std::string search = "\"" + key + "\"";
        size_t pos = raw.find(search);
        if (pos == std::string::npos) return "";

        // skip past key, colon, optional whitespace, opening quote
        pos = raw.find(':', pos);
        if (pos == std::string::npos) return "";
        pos = raw.find('"', pos);
        if (pos == std::string::npos) return "";
        pos++;  // skip opening quote

        size_t end = raw.find('"', pos);
        if (end == std::string::npos) return "";
        return raw.substr(pos, end - pos);
    };

    auto find_int = [&](const std::string& key) -> int {
        std::string search = "\"" + key + "\"";
        size_t pos = raw.find(search);
        if (pos == std::string::npos) return -1;
        pos = raw.find(':', pos);
        if (pos == std::string::npos) return -1;
        pos++;
        while (pos < raw.size() && (raw[pos] == ' ' || raw[pos] == '\t')) pos++;
        return std::atoi(raw.c_str() + pos);
    };

    std::string action = find_string("action");
    if (action.empty()) return;

    std::cout << "[executor] action: " << action << std::endl;

    if (action == "reduce_power_limit") {
        int gpu_idx = find_int("gpu_index");
        int watts = find_int("watts");
        if (gpu_idx >= 0 && watts > 0) {
            nvmlDevice_t device;
            nvmlDeviceGetHandleByIndex(gpu_idx, &device);
            nvmlReturn_t r = nvmlDeviceSetPowerManagementLimit(device, watts * 1000);
            std::cout << "[executor] set GPU " << gpu_idx
                      << " power limit to " << watts << "W: "
                      << (r == NVML_SUCCESS ? "ok" : "failed") << std::endl;
        }
    }
    else if (action == "drain_node") {
        std::string reason = find_string("reason");
        char hostname[256];
        gethostname(hostname, sizeof(hostname));
        std::string cmd = "scontrol update NodeName=" + std::string(hostname)
                        + " State=DRAIN Reason=\"" + reason + "\"";
        std::cout << "[executor] running: " << cmd << std::endl;
        std::system(cmd.c_str());
    }
}