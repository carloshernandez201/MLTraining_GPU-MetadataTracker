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

// Escape a value for embedding inside a JSON string literal.
inline std::string json_escape(const std::string& s) {
    std::ostringstream o;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o << "\\\""; break;
            case '\\': o << "\\\\"; break;
            case '\n': o << "\\n"; break;
            case '\r': o << "\\r"; break;
            case '\t': o << "\\t"; break;
            default:
                if (c < 0x20) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << (int)c;
                else o << c;
        }
    }
    return o.str();
}

// "logs/<SLURM_JOB_ID>/metrics.jsonl" — creates dirs if needed
// (base dir can be overridden with PROFILER_LOG_DIR)
inline std::string get_job_id() {
    const char* env = std::getenv("SLURM_JOB_ID");
    return env ? env : "unknown";
}

inline std::string get_log_dir() {
    const char* base_env = std::getenv("PROFILER_LOG_DIR");
    std::string base = base_env ? base_env : "logs";
    std::string dir = base + "/" + get_job_id();
    mkdir(base.c_str(), 0755);
    mkdir(dir.c_str(), 0755);
    return dir;
}

inline std::string get_log_path() {
    return get_log_dir() + "/metrics.jsonl";
}

// Returns: {"timestamp":..., "job_id":"...", "host":"...", "pid":..., "source":"X", "data":
// You append your data object then close with "}"
inline std::string envelope(const std::string& source) {
    auto now = std::chrono::system_clock::now();
    double epoch = std::chrono::duration<double>(now.time_since_epoch()).count();

    char hostname[256] = {0};
    gethostname(hostname, sizeof(hostname) - 1);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(3);
    oss << "{\"timestamp\":" << epoch
        << ",\"job_id\":\"" << json_escape(get_job_id()) << "\""
        << ",\"host\":\"" << json_escape(hostname) << "\""
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
    const char* p = out.c_str();
    size_t left = out.size();
    while (left > 0) {
        ssize_t n = ::write(fd, p, left);
        if (n <= 0) break;
        p += n;
        left -= (size_t)n;
    }
    flock(fd, LOCK_UN);
    close(fd);
}