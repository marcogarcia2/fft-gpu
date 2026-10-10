.PHONY: all compile run clean cuda-compile cuda-run cuda-profile cuda-clean roofline

# LOCAL Variables (CPU version)
CXX = g++-14
CXX_FLAGS =
CXX_LIBS = -fopenmp

# REMOTE variables (GPU version)
RCXX = nvcc
RCXX_FLAGS = 
RCXX_LIBS = -lcufft -lnvidia-ml
# NCU_FLAGS = --set full 
PROF_DIR = /tmp/$(USER)_prof
REP = $(TIMESTAMP)_$(KERNEL)_$(NVALUE)_$(PRECISION)
GID = $(shell id -g)

NCU_FLAGS = --section SpeedOfLight --section MemoryWorkloadAnalysis_Tables \
    		--section Occupancy --section LaunchStats --section WarpStateStats


# LOCAL TARGETS:
all: compile run clean

compile:
	$(CXX) $(CXX_FLAGS) cpu_version/main.cpp -o main $(CXX_LIBS)

run: 
	./main

clean:
	rm -f main


# REMOTE TARGETS:
cuda-compile:
	@ $(RCXX) $(RCXX_FLAGS) main.cu -o main $(RCXX_LIBS)

cuda-run:
	@ ./main

cuda-profile:
	@ mkdir -p $(PROF_DIR)
	@ cp ./main $(PROF_DIR)/main
	@ cd $(PROF_DIR) && sudo -S sh -c '/usr/bin/ncu $(NCU_FLAGS) -f -o $(REP) ./main "$(KERNEL)" $(NVALUE) $(PRECISION) && chown $(USER):$(GID) "$(REP).ncu-rep"'
	@ cp $(PROF_DIR)/$(REP).ncu-rep ./

cuda-profile-csv:
	@ ncu -i $(REP).ncu-rep --csv --page raw > $(REP).csv

cuda-clean:
	@ rm -f main


# ROOFLINE MODEL
roofline:
	$(RCXX) roofline.cu -o roofline
	./roofline
	rm -f roofline
