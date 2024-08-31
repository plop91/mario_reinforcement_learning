all : genome_so

genome_so: genome.cpp
	g++ -c -fPIC genome.cpp -o genome.o
	g++ -shared -Wl,-soname,libgenome.so -o libgenome.so  genome.o

clean:
	rm -f genome.o libgenome.so