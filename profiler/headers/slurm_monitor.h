#pragma once
#include <string>
#include <vector>
#include <map>

// Metadata for one Slurm job, merged from scontrol / squeue.
struct SlurmJobInfo {
    std::string job_id;
    std::string name;
    std::string user;
    std::string state;
    std::string partition;
    std::string node_list;
    int num_nodes = 0;
    int num_cpus = 0;
    double mem_mb = 0;          // requested memory, MB
    int num_gpus = 0;
    std::string gpu_type;       // e.g. "a100", empty if untyped
    std::string time_limit;     // Slurm format, e.g. "1-00:00:00"
    std::string elapsed;        // Slurm format, e.g. "04:12:33"
    std::string tres;           // raw AllocTRES / ReqTRES string
    std::string work_dir;
};

// Resource efficiency for a job (meaningful once it has run for a while / finished).
struct SlurmEfficiency {
    std::string job_id;
    std::string state;
    long elapsed_sec = 0;
    long time_limit_sec = 0;
    double max_rss_mb = 0;
    double req_mem_mb = 0;
    int num_gpus = 0;
    double walltime_efficiency = 0;  // elapsed / time limit
    double memory_efficiency = 0;    // peak RSS / requested memory
    double gpu_seconds = 0;          // GPUs * elapsed seconds (what you are billed for)
};

// --- helpers (pure, unit-testable) ---
std::string run(const std::string& cmd);
std::vector<std::string> split(const std::string& s, char delim);
std::string trim(const std::string& s);
bool is_valid_job_id(const std::string& id);
double parse_mem_mb(const std::string& s);            // "64G", "4000M", "16Gn", "512K"
long parse_slurm_duration(const std::string& s);      // "1-02:03:04", "03:04", "UNLIMITED" -> -1
std::map<std::string, std::string> parse_scontrol(const std::string& text);
void parse_gpus_from_tres(const std::string& tres, int& count, std::string& type);

SlurmJobInfo job_info_from_scontrol(const std::string& scontrol_output);
SlurmEfficiency efficiency_from_sacct(const std::string& job_id, const std::string& sacct_output);

std::string job_info_to_json(const SlurmJobInfo& info);
std::string efficiency_to_json(const SlurmEfficiency& eff);

// --- live queries (shell out to Slurm) ---
bool get_job_info(const std::string& job_id, SlurmJobInfo& out);
bool get_job_efficiency(const std::string& job_id, SlurmEfficiency& out);

// Query the current job ($SLURM_JOB_ID), or all of the user's jobs outside a
// job, and append "slurm" records to the metrics log (or stdout).
int slurm_query(bool to_stdout = false);
