# POSIX build of the CPU SumFold kernel (gcc/clang + OpenMP).  Windows: see ../README.md.
CC      ?= gcc
CFLAGS  ?= -O3 -march=native -funroll-loops -fopenmp -std=gnu11 -Wall
SRCS     = main.c sumfold.c sha256.c
HDRS     = gl_field.h sumfold.h sha256.h

build/sumfold_cpu: $(SRCS) $(HDRS)
	mkdir -p build
	$(CC) $(CFLAGS) -DSF_BUILD_FLAGS='"$(CFLAGS)"' -o $@ $(SRCS)

selftest: build/sumfold_cpu
	./build/sumfold_cpu --selftest

clean:
	rm -rf build

.PHONY: selftest clean
