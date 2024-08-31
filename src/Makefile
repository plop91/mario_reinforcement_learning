
all : neat_so

neat_so: neat.cpp
	g++ -c -fPIC neat.cpp -o neat.o
	g++ -shared -Wl,-soname,libneat.so -o libneat.so  neat.o

clean:
	rm -f neat.o libneat.so