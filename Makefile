# Local Variables
CXX = g++-14
CXX_FLAGS =

# Host Variables
CXX_HOST = g++
CXX_HOST_FLAGS = 

all:
	$(CXX) $(CXX_FLAGS) main.cpp -o main

run: 
	./main

clean:
	rm -f main

compile-on-host:
	$(CXX_HOST) $(CXX_HOST_FLAGS) main.cpp -o main

run-on-host:
	./main

clean-on-host:
	rm -f main
