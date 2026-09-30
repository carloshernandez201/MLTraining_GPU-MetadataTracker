# GPU training profiler — C++ / CUDA components.
#
#   make                 profiler_daemon + slurm_monitor (needs NVML: CUDA toolkit or driver dev pkg)
#   make MOCK=1          same, with a synthetic GPU backend (no NVIDIA driver required)
#   make cuda            cuda_profiler_demo (needs nvcc)
#   make pybind          cuda_profiler Python extension (needs nvcc + pybind11)
#   make test            build + run C++ unit tests (no GPU / Slurm needed)
#
# On HiPerGator:  module load cuda gcc && make
.RECIPEPREFIX = >

CXX       ?= g++
NVCC      ?= nvcc
CUDA_HOME ?= $(or $(CUDA_PATH),/usr/local/cuda)
BIN       := bin

CXXFLAGS  := -O2 -std=c++17 -Wall -Wextra -Iheaders
NVCCFLAGS := -O2 -std=c++17 -Iheaders

ifeq ($(MOCK),1)
  GPU_FLAGS :=
  GPU_LIBS  :=
  CXXFLAGS  += -DPROFILER_MOCK_NVML
else
  GPU_FLAGS := -I$(CUDA_HOME)/include
  GPU_LIBS  := -L$(CUDA_HOME)/lib64 -L$(CUDA_HOME)/lib64/stubs -lnvidia-ml
endif

HEADERS := $(wildcard headers/*.h headers/*.hpp)

.PHONY: all cuda pybind test clean

all: $(BIN)/profiler_daemon $(BIN)/slurm_monitor

$(BIN):
> mkdir -p $(BIN)

$(BIN)/profiler_daemon: main.cpp src/collector.cpp src/slurm_monitor.cpp $(HEADERS) | $(BIN)
> $(CXX) $(CXXFLAGS) $(GPU_FLAGS) -DSLURM_MONITOR_HEADER_ONLY \
>   main.cpp src/collector.cpp src/slurm_monitor.cpp -o $@ $(GPU_LIBS) -lpthread

$(BIN)/slurm_monitor: src/slurm_monitor.cpp $(HEADERS) | $(BIN)
> $(CXX) $(CXXFLAGS) src/slurm_monitor.cpp -o $@

cuda: $(BIN)/cuda_profiler_demo

$(BIN)/cuda_profiler_demo: src/cuda_profiler.cpp headers/cuda_profiler.hpp | $(BIN)
> $(NVCC) $(NVCCFLAGS) -x cu src/cuda_profiler.cpp -o $@

pybind: src/cuda_profiler.cpp headers/cuda_profiler.hpp
> $(NVCC) $(NVCCFLAGS) -x cu -shared -Xcompiler -fPIC -DBUILD_PYBIND \
>   $$(python3 -m pybind11 --includes) src/cuda_profiler.cpp \
>   -o cuda_profiler$$(python3-config --extension-suffix)

$(BIN)/test_slurm_parse: tests/test_slurm_parse.cpp src/slurm_monitor.cpp $(HEADERS) | $(BIN)
> $(CXX) $(CXXFLAGS) -DSLURM_MONITOR_HEADER_ONLY tests/test_slurm_parse.cpp src/slurm_monitor.cpp -o $@

test: $(BIN)/test_slurm_parse
> ./$(BIN)/test_slurm_parse

clean:
> rm -rf $(BIN) cuda_profiler*.so
