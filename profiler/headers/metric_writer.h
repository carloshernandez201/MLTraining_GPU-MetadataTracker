#pragma once
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>

// "logs/<SLURM_JOB_ID>/metrics.jsonl" — creates dirs if needed
inline std::string get_log_path() {
    const char* env = std::getenv("SLURM_JOB_ID");
    std::string job_id = env ? env : "unknown";
    std::string dir = "logs/" + job_id;
    mkdir("logs", 0755);
    mkdir(dir.c_str(), 0755);
    return dir + "/metrics.jsonl";
}

// Returns: {"timestamp":..., "job_id":"...", "host":"...", "pid":..., "source":"X", "data":
// You append your data object then close with "}"
inline std::string envelope(const std::string& source) {
    auto now = std::chrono::system_clock::now();
    double epoch = std::chrono::duration<double>(now.time_since_epoch()).count();

    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    const char* env = std::getenv("SLURM_JOB_ID");
    std::string job_id = env ? env : "unknown";

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "{\"timestamp\":" << epoch
        << ",\"job_id\":\"" << job_id << "\""
        << ",\"host\":\"" << hostname << "\""
        << ",\"pid\":" << getpid()
        << ",\"source\":\"" << source << "\""
        << ",\"data\":";
    return oss.str();
}

// flock'd append to the shared JSONL
inline void write_metric(const std::string& json_line) {
    std::string path = get_log_path();
    int fd = open(path.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644);
    if (fd < 0) return;

    flock(fd, LOCK_EX);
    std::string out = json_line + "\n";
    ::write(fd, out.c_str(), out.size());
    flock(fd, LOCK_UN);
    close(fd);
}