#include <iostream>
#include <string>
#include <cstdio>
#include <sstream>
#include <vector>
#include <fstream>
#include <chrono>
#include <unistd.h>     // gethostname, getpid
#include <cstdlib>      // getenv

string run(const string& cmd);
vector<string> split(const string& s, char delim);
int slurm_query();