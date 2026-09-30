// Slurm job metadata + efficiency.
//
// Standalone:   g++ -O2 -std=c++17 -Iheaders src/slurm_monitor.cpp -o slurm_monitor
// As a library: compile with -DSLURM_MONITOR_HEADER_ONLY to drop main().
#include "slurm_monitor.h"
#include "metric_writer.h"

#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <algorithm>

using namespace std;

string run(const string& cmd) {
    string result;
    char buffer[512];
    FILE* pipe = popen((cmd + " 2>/dev/null").c_str(), "r");
    if (!pipe) return "";
    while (fgets(buffer, sizeof(buffer), pipe)) {
        result += buffer;
    }
    pclose(pipe);
    return result;
}

vector<string> split(const string& s, char delim) {
    vector<string> parts;
    istringstream iss(s);
    string token;
    while (getline(iss, token, delim)) {
        parts.push_back(token);
    }
    return parts;
}

string trim(const string& s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == string::npos) return "";
    size_t e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

// Job IDs go into shell commands, so only allow digits plus array/het separators.
bool is_valid_job_id(const string& id) {
    if (id.empty()) return false;
    return all_of(id.begin(), id.end(), [](char c) {
        return isdigit((unsigned char)c) || c == '_' || c == '+' || c == '.';
    });
}

double parse_mem_mb(const string& raw) {
    string s = trim(raw);
    if (s.empty()) return 0;
    // Older sacct appends n (per node) / c (per cpu) to ReqMem.
    if (s.back() == 'n' || s.back() == 'c') s.pop_back();
    if (s.empty()) return 0;

    char unit = 'M';
    if (isalpha((unsigned char)s.back())) {
        unit = (char)toupper((unsigned char)s.back());
        s.pop_back();
    }
    double v = atof(s.c_str());
    switch (unit) {
        case 'K': return v / 1024.0;
        case 'M': return v;
        case 'G': return v * 1024.0;
        case 'T': return v * 1024.0 * 1024.0;
        default:  return v;
    }
}

long parse_slurm_duration(const string& raw) {
    string s = trim(raw);
    if (s.empty()) return 0;
    if (s == "UNLIMITED" || s == "Partition_Limit" || s == "INVALID") return -1;

    long days = 0;
    size_t dash = s.find('-');
    if (dash != string::npos) {
        days = atol(s.substr(0, dash).c_str());
        s = s.substr(dash + 1);
    }
    vector<string> p = split(s, ':');
    long h = 0, m = 0, sec = 0;
    // Formats: MM:SS, HH:MM:SS, D-HH, D-HH:MM, D-HH:MM:SS
    if (dash != string::npos) {
        if (p.size() >= 1) h = atol(p[0].c_str());
        if (p.size() >= 2) m = atol(p[1].c_str());
        if (p.size() >= 3) sec = atol(p[2].c_str());
    } else if (p.size() == 3) {
        h = atol(p[0].c_str()); m = atol(p[1].c_str()); sec = atol(p[2].c_str());
    } else if (p.size() == 2) {
        m = atol(p[0].c_str()); sec = atol(p[1].c_str());
    } else if (p.size() == 1) {
        m = atol(p[0].c_str());  // bare number = minutes in Slurm
    }
    return ((days * 24 + h) * 60 + m) * 60 + sec;
}

map<string, string> parse_scontrol(const string& text) {
    // scontrol prints whitespace-separated Key=Value tokens across several lines.
    map<string, string> kv;
    istringstream iss(text);
    string token;
    while (iss >> token) {
        size_t eq = token.find('=');
        if (eq == string::npos || eq == 0) continue;
        kv[token.substr(0, eq)] = token.substr(eq + 1);
    }
    return kv;
}

void parse_gpus_from_tres(const string& tres, int& count, string& type) {
    // e.g. "cpu=32,mem=64G,node=1,billing=32,gres/gpu=2"
    //      "cpu=8,mem=32G,gres/gpu:a100=2,gres/gpu=2"
    count = 0;
    type.clear();
    for (const string& item : split(tres, ',')) {
        size_t eq = item.find('=');
        if (eq == string::npos) continue;
        string key = item.substr(0, eq);
        int val = atoi(item.substr(eq + 1).c_str());
        if (key == "gres/gpu") {
            count = max(count, val);
        } else if (key.rfind("gres/gpu:", 0) == 0) {
            type = key.substr(9);
            count = max(count, val);
        }
    }
}

SlurmJobInfo job_info_from_scontrol(const string& out) {
    map<string, string> kv = parse_scontrol(out);
    auto get = [&](const string& k) { auto it = kv.find(k); return it == kv.end() ? string() : it->second; };

    SlurmJobInfo j;
    j.job_id = get("JobId");
    j.name = get("JobName");
    string user = get("UserId");               // "alice(12345)"
    j.user = user.substr(0, user.find('('));
    j.state = get("JobState");
    j.partition = get("Partition");
    j.node_list = get("NodeList");
    j.num_nodes = atoi(get("NumNodes").c_str());
    j.num_cpus = atoi(get("NumCPUs").c_str());
    j.time_limit = get("TimeLimit");
    j.elapsed = get("RunTime");
    j.work_dir = get("WorkDir");

    j.tres = get("AllocTRES");
    if (j.tres.empty()) j.tres = get("ReqTRES");
    for (const string& item : split(j.tres, ',')) {
        if (item.rfind("mem=", 0) == 0) j.mem_mb = parse_mem_mb(item.substr(4));
    }
    if (j.mem_mb == 0) j.mem_mb = parse_mem_mb(get("MinMemoryNode"));
    parse_gpus_from_tres(j.tres, j.num_gpus, j.gpu_type);

    // Older Slurm reports GPUs only through Gres / TresPerNode, e.g. "gpu:a100:2".
    if (j.num_gpus == 0) {
        string gres = get("TresPerNode");
        if (gres.empty()) gres = get("Gres");
        for (const string& g : split(gres, ',')) {
            vector<string> p = split(g, ':');
            string head = p.empty() ? "" : p[0];
            if (head.find("gpu") == string::npos) continue;
            if (p.size() == 3) { j.gpu_type = p[1]; j.num_gpus = atoi(p[2].c_str()); }
            else if (p.size() == 2) { j.num_gpus = atoi(p[1].c_str()); }
            if (j.num_nodes > 1) j.num_gpus *= j.num_nodes;
        }
    }
    return j;
}

SlurmEfficiency efficiency_from_sacct(const string& job_id, const string& out) {
    // Expected columns (--parsable2 --noheader):
    // JobID|State|ElapsedRaw|TimelimitRaw|MaxRSS|ReqMem|AllocTRES
    SlurmEfficiency e;
    e.job_id = job_id;
    for (const string& line : split(out, '\n')) {
        vector<string> f = split(line, '|');
        if (f.size() < 7) continue;

        // The allocation line (no ".step" suffix) carries the limits;
        // step lines (".batch", ".0", ...) carry MaxRSS.
        bool is_alloc = f[0].find('.') == string::npos;
        if (is_alloc) {
            e.state = f[1];
            e.elapsed_sec = atol(f[2].c_str());
            e.time_limit_sec = atol(f[3].c_str()) * 60;  // TimelimitRaw is minutes
            e.req_mem_mb = parse_mem_mb(f[5]);
            string type;
            parse_gpus_from_tres(f[6], e.num_gpus, type);
        }
        e.max_rss_mb = max(e.max_rss_mb, parse_mem_mb(f[4]));
    }
    if (e.time_limit_sec > 0) e.walltime_efficiency = (double)e.elapsed_sec / e.time_limit_sec;
    if (e.req_mem_mb > 0) e.memory_efficiency = e.max_rss_mb / e.req_mem_mb;
    e.gpu_seconds = (double)e.num_gpus * e.elapsed_sec;
    return e;
}

string job_info_to_json(const SlurmJobInfo& j) {
    ostringstream d;
    d << fixed << setprecision(1);
    d << "{\"type\":\"job_info\""
      << ",\"slurm_job_id\":\"" << json_escape(j.job_id) << "\""
      << ",\"name\":\"" << json_escape(j.name) << "\""
      << ",\"user\":\"" << json_escape(j.user) << "\""
      << ",\"state\":\"" << json_escape(j.state) << "\""
      << ",\"partition\":\"" << json_escape(j.partition) << "\""
      << ",\"node_list\":\"" << json_escape(j.node_list) << "\""
      << ",\"num_nodes\":" << j.num_nodes
      << ",\"num_cpus\":" << j.num_cpus
      << ",\"mem_mb\":" << j.mem_mb
      << ",\"num_gpus\":" << j.num_gpus
      << ",\"gpu_type\":\"" << json_escape(j.gpu_type) << "\""
      << ",\"time_limit\":\"" << json_escape(j.time_limit) << "\""
      << ",\"elapsed\":\"" << json_escape(j.elapsed) << "\""
      << ",\"tres\":\"" << json_escape(j.tres) << "\""
      << ",\"work_dir\":\"" << json_escape(j.work_dir) << "\""
      << "}";
    return d.str();
}

string efficiency_to_json(const SlurmEfficiency& e) {
    ostringstream d;
    d << fixed << setprecision(4);
    d << "{\"type\":\"efficiency\""
      << ",\"slurm_job_id\":\"" << json_escape(e.job_id) << "\""
      << ",\"state\":\"" << json_escape(e.state) << "\""
      << ",\"elapsed_sec\":" << e.elapsed_sec
      << ",\"time_limit_sec\":" << e.time_limit_sec
      << ",\"max_rss_mb\":" << e.max_rss_mb
      << ",\"req_mem_mb\":" << e.req_mem_mb
      << ",\"num_gpus\":" << e.num_gpus
      << ",\"walltime_efficiency\":" << e.walltime_efficiency
      << ",\"memory_efficiency\":" << e.memory_efficiency
      << ",\"gpu_seconds\":" << e.gpu_seconds
      << "}";
    return d.str();
}

bool get_job_info(const string& job_id, SlurmJobInfo& out) {
    if (!is_valid_job_id(job_id)) return false;
    string text = run("scontrol show job " + job_id);
    if (text.find("JobId=") == string::npos) return false;
    out = job_info_from_scontrol(text);
    return true;
}

bool get_job_efficiency(const string& job_id, SlurmEfficiency& out) {
    if (!is_valid_job_id(job_id)) return false;
    string text = run("sacct -j " + job_id +
                      " --format=JobID,State,ElapsedRaw,TimelimitRaw,MaxRSS,ReqMem,AllocTRES"
                      " --parsable2 --noheader");
    if (trim(text).empty()) return false;
    out = efficiency_from_sacct(job_id, text);
    return true;
}

static void emit(const string& data, bool to_stdout) {
    string line = envelope("slurm") + data + "}";
    if (to_stdout) cout << line << endl;
    else write_metric(line);
}

int slurm_query(bool to_stdout) {
    vector<string> job_ids;
    const char* env = getenv("SLURM_JOB_ID");
    if (env && is_valid_job_id(env)) {
        job_ids.push_back(env);
    } else {
        // Outside a job: report every job belonging to the current user.
        for (const string& line : split(run("squeue --me --noheader --format=%i"), '\n')) {
            string id = trim(line);
            if (is_valid_job_id(id)) job_ids.push_back(id);
        }
    }

    if (job_ids.empty()) {
        cerr << "[slurm] no Slurm jobs found (not inside a job, or Slurm not available)" << endl;
        return 1;
    }

    int found = 0;
    for (const string& id : job_ids) {
        SlurmJobInfo info;
        if (get_job_info(id, info)) {
            emit(job_info_to_json(info), to_stdout);
            found++;
        }
        SlurmEfficiency eff;
        if (get_job_efficiency(id, eff)) emit(efficiency_to_json(eff), to_stdout);
    }
    return found > 0 ? 0 : 1;
}

#ifndef SLURM_MONITOR_HEADER_ONLY
int main(int argc, char** argv) {
    bool to_stdout = true;
    string job_id;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--log") to_stdout = false;
        else if (a == "-h" || a == "--help") {
            cout << "usage: slurm_monitor [JOB_ID] [--log]\n"
                    "  JOB_ID  job to inspect (default: $SLURM_JOB_ID, else all of your jobs)\n"
                    "  --log   append to logs/<job>/metrics.jsonl instead of stdout\n";
            return 0;
        } else job_id = a;
    }

    if (job_id.empty()) return slurm_query(to_stdout);

    if (!is_valid_job_id(job_id)) {
        cerr << "invalid job id: " << job_id << endl;
        return 2;
    }
    SlurmJobInfo info;
    SlurmEfficiency eff;
    bool ok = get_job_info(job_id, info);
    if (ok) emit(job_info_to_json(info), to_stdout);
    if (get_job_efficiency(job_id, eff)) { emit(efficiency_to_json(eff), to_stdout); ok = true; }
    if (!ok) cerr << "no data for job " << job_id << endl;
    return ok ? 0 : 1;
}
#endif
