// Unit tests for the Slurm output parsers — runs anywhere, no Slurm needed.
#include "slurm_monitor.h"

#include <cmath>
#include <iostream>
#include <string>

static int failures = 0;

#define CHECK(cond) do { if (!(cond)) { \
    std::cerr << "FAIL " << __FILE__ << ":" << __LINE__ << "  " #cond << std::endl; failures++; } } while (0)

static bool near(double a, double b) { return std::fabs(a - b) < 1e-6; }

static const char* SCONTROL = R"(JobId=12345 JobName=train_resnet
   UserId=alice(51234) GroupId=lab(9001) MCS_label=N/A
   Priority=1 Nice=0 Account=lab QOS=lab
   JobState=RUNNING Reason=None Dependency=(null)
   RunTime=04:12:33 TimeLimit=1-00:00:00 TimeMin=N/A
   Partition=gpu AllocNode:Sid=login1:4242
   NodeList=c1000a-s17
   NumNodes=1 NumCPUs=32 NumTasks=1 CPUs/Task=32 ReqB:S:C:T=0:0:*:*
   ReqTRES=cpu=32,mem=64G,node=1,billing=32,gres/gpu=2
   AllocTRES=cpu=32,mem=64G,node=1,billing=32,gres/gpu:a100=2,gres/gpu=2
   MinMemoryNode=64G
   WorkDir=/blue/lab/alice/project
)";

static const char* SACCT =
    "12345|COMPLETED|14400|1440|0|64G|billing=32,cpu=32,gres/gpu=2,mem=64G,node=1\n"
    "12345.batch|COMPLETED|14400||20480M|||\n"
    "12345.0|COMPLETED|14000||32G|||\n";

int main() {
    // durations
    CHECK(parse_slurm_duration("1-00:00:00") == 86400);
    CHECK(parse_slurm_duration("04:12:33") == 4 * 3600 + 12 * 60 + 33);
    CHECK(parse_slurm_duration("05:30") == 330);
    CHECK(parse_slurm_duration("2-12") == (2 * 24 + 12) * 3600);
    CHECK(parse_slurm_duration("90") == 5400);
    CHECK(parse_slurm_duration("UNLIMITED") == -1);

    // memory
    CHECK(near(parse_mem_mb("64G"), 65536));
    CHECK(near(parse_mem_mb("4000M"), 4000));
    CHECK(near(parse_mem_mb("16Gn"), 16384));
    CHECK(near(parse_mem_mb("2048K"), 2));
    CHECK(near(parse_mem_mb(""), 0));

    // job id validation (these end up in shell commands)
    CHECK(is_valid_job_id("12345"));
    CHECK(is_valid_job_id("12345_7"));
    CHECK(!is_valid_job_id("123; rm -rf /"));
    CHECK(!is_valid_job_id(""));

    // TRES
    int n = 0; std::string type;
    parse_gpus_from_tres("cpu=8,mem=32G,gres/gpu:a100=4,gres/gpu=4", n, type);
    CHECK(n == 4); CHECK(type == "a100");
    parse_gpus_from_tres("cpu=8,mem=32G", n, type);
    CHECK(n == 0); CHECK(type.empty());

    // scontrol
    SlurmJobInfo j = job_info_from_scontrol(SCONTROL);
    CHECK(j.job_id == "12345");
    CHECK(j.name == "train_resnet");
    CHECK(j.user == "alice");
    CHECK(j.state == "RUNNING");
    CHECK(j.partition == "gpu");
    CHECK(j.node_list == "c1000a-s17");
    CHECK(j.num_nodes == 1);
    CHECK(j.num_cpus == 32);
    CHECK(near(j.mem_mb, 65536));
    CHECK(j.num_gpus == 2);
    CHECK(j.gpu_type == "a100");
    CHECK(j.time_limit == "1-00:00:00");
    CHECK(j.elapsed == "04:12:33");
    CHECK(j.work_dir == "/blue/lab/alice/project");

    // legacy Gres field
    SlurmJobInfo old = job_info_from_scontrol("JobId=7 NumNodes=2 Gres=gpu:v100:2");
    CHECK(old.num_gpus == 4); CHECK(old.gpu_type == "v100");

    // sacct efficiency
    SlurmEfficiency e = efficiency_from_sacct("12345", SACCT);
    CHECK(e.state == "COMPLETED");
    CHECK(e.elapsed_sec == 14400);
    CHECK(e.time_limit_sec == 86400);
    CHECK(near(e.max_rss_mb, 32768));
    CHECK(near(e.req_mem_mb, 65536));
    CHECK(e.num_gpus == 2);
    CHECK(near(e.walltime_efficiency, 14400.0 / 86400.0));
    CHECK(near(e.memory_efficiency, 0.5));
    CHECK(near(e.gpu_seconds, 28800));

    // JSON output is well-formed enough to contain escaped fields
    std::string js = job_info_to_json(j);
    CHECK(js.find("\"gpu_type\":\"a100\"") != std::string::npos);

    if (failures) {
        std::cerr << failures << " check(s) failed" << std::endl;
        return 1;
    }
    std::cout << "all slurm parser tests passed" << std::endl;
    return 0;
}
