.PHONY: all clean check

CXXFLAGS+=-g -I../include -I../git_submodules/calculisto/array/include
LINK.o=$(LINK.cc)

all: check

check: tests
	./tests

tests: tests.o test-dual.o

tests.o: tests.cpp

test-dual.o: test-dual.cpp ../include/calculisto/automatic_differentiation/automatic_differentiation.hpp
	$(CXX) $(CXXFLAGS) -c -o test-dual.o test-dual.cpp

clean: 
	rm -f tests *.o 
