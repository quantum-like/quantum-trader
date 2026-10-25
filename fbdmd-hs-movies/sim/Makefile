# Build the hard-sphere EDMD / collision-Verlet simulator.
#   make            -> ./hsdmd  (CXX defaults to c++: clang++ on macOS, g++ on Linux)
#   make CXX=g++    -> force a compiler
#   make test       -> unit self-tests (incl. splitting, quartic solver, bound-pair dt scans)
#   make order      -> energy-error order test on the production system (writes ../traj/<name>_order.csv,
#                      ~10 s on 4 cpus; runs from the project root with its venv, PYTHON=.venv/bin/python;
#                      extra flags via ORDER_ARGS="--coulomb dsf --name _test/x")
CXX      ?= c++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra
PYTHON   ?= .venv/bin/python

.PHONY: all clean test order
all: hsdmd

hsdmd: hsdmd.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

test: hsdmd
	./hsdmd --selftest

order: hsdmd
	cd .. && $(PYTHON) sim/order_test.py $(ORDER_ARGS)

clean:
	rm -f hsdmd
