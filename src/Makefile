# CXX = g++
# CXXFLAGS = -Wall -std=c++11
# TARGET = neatTest

# # Get all .cpp files in the directory
# SRCS = $(wildcard *.cpp)
# # Generate object files from .cpp files
# OBJS = $(SRCS:.cpp=.o) 

# all: $(TARGET)

# $(TARGET): $(OBJS)
# 	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

# %.o: %.cpp
# 	$(CXX) $(CXXFLAGS) -c $< -o $@

# clean:
# 	rm -f $(OBJS) $(TARGET)
all : neat_so
neat_so: neat.cpp
	g++ -c -fPIC neat.cpp -o neat.o
	g++ -shared -Wl,-soname,libneat.so -o libneat.so  neat.o