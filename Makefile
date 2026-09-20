# LOCAL Variables (CPU version)
CXX = g++-14
CXX_FLAGS =
CXX_LIBS = -fopenmp

# REMOTE variables (GPU version)
RCXX = nvcc
RCXX_FLAGS = 
RCXX_LIBS = -lcufft -lnvidia-ml

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

cuda-clean:
	@ rm -f main


# ROOFLINE MODEL
roofline:
	$(RCXX) roofline.cu -o roofline
	./roofline
	rm -f roofline
