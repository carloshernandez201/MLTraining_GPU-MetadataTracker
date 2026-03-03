#include "slurm_monitor.h"
#include "metric_writer.h"
#include <socket>
using namespace std;

string run(const string& cmd) {
    string result;                              // "" is fine but unnecessary
    char buffer[512];                           // lowercase — C++ is case sensitive
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return "";                       // added null check

    while (fgets(buffer, sizeof(buffer), pipe)) {  // sizeof(buffer) safer than hardcoding 512
        result += buffer;                       // was BUFFER vs buffer mismatch
    }
    pclose(pipe);                               // missing semicolon
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

int slurm_query() {
    ofstream outfile("slurm_metadata.jsonl", ios::app); 
       
    auto now = std::chrono::system_clock::now();
    auto epoch = std::chrono::duration<double>(
        now.time_since_epoch()
    ).count();

    // Hostname
    char hostname[256];
    
    socket.gethostname();

    // SLURM Job ID
    const char* job_id_env = getenv("SLURM_JOB_ID");
    std::string job_id = job_id_env ? job_id_env : "unknown";

    // PID
    pid_t pid = getpid();


    string squeueResult = run("squeue --format='%i|%j|%u|%T|%b' --noheader");
                                                // was missing | delimiters in format

    for (const string& line : split(squeueResult, '\n')) {
        if (line.empty()) continue;
        vector<string> fields = split(line, '|');  // '|' not "|" — split takes a char
        if (fields.size() < 5) continue;

        string jobId = fields[0];
        string name = fields[1];
        string user = fields[2];
        string state = fields[3];               // just use 3, not 4-1
        string gres = fields[4];

        string scontrolResult = run("scontrol show job " + jobId);

        string partition, nodeList, timeLimit, workDir;
        int numCpus = 0;

        for (const string& token : split(scontrolResult, ' ')) {  // was "details" — undefined
            size_t eq = token.find('=');
            if (eq == string::npos) continue;
            string key = token.substr(0, eq);
            string val = token.substr(eq + 1);

            if (key == "Partition") partition = val;
            else if (key == "NodeList") nodeList = val;
            else if (key == "TimeLimit") timeLimit = val;
            else if (key == "NumCPUs") numCpus = stoi(val);
            else if (key == "WorkDir") workDir = val;
        }

    }
std::ostringstream data;
data << "{\"name\":\"" << name << "\""
     << ",\"state\":\"" << state << "\""
     << ",\"partition\":\"" << partition << "\""
     << ",\"node_list\":\"" << nodeList << "\""
     << ",\"num_cpus\":" << numCpus
     << ",\"time_limit\":\"" << timeLimit << "\""
     << "}";

write_metric(envelope("slurm") + data.str() + "}");

    outfile.close();
    return 0;
}