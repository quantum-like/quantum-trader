# Build the hard-sphere EDMD / collision-Verlet simulator.
#   make            -> ./hsdmd  (CXX defaults to c++: clang++ on macOS, g++ on Linux)
#   make CXX=g++    -> force a compiler
#   make test       -> unit self-tests
CXX      ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra

.PHONY: all clean test
all: hsdmd

hsdmd: hsdmd.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

test: hsdmd
	./hsdmd --selftest

clean:
	rm -f hsdmd
