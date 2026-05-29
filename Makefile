# Conservation Spectral SDK — OpenCL Backend

CC      = gcc
CFLAGS  = -std=c11 -Wall -Wextra -O2 -Iinclude -Isrc -DCL_TARGET_OPENCL_VERSION=120
LDFLAGS = -lOpenCL -lm

KERNELS = kernels

# test_chord.c defines CS_IMPLEMENTATION for the header-only SDK
# conservation_spectral_cl.c does NOT define CS_IMPLEMENTATION
# They must be compiled as separate translation units.

BIN_TEST = test_chord

.PHONY: all test clean install-deps

all: $(BIN_TEST)

src/conservation_spectral_cl.o: src/conservation_spectral_cl.c include/conservation_spectral_cl.h include/conservation_spectral.h
	$(CC) $(CFLAGS) -c -o $@ $<

src/test_chord.o: src/test_chord.c include/conservation_spectral.h include/conservation_spectral_cl.h
	$(CC) $(CFLAGS) -c -o $@ $<

$(BIN_TEST): src/test_chord.o src/conservation_spectral_cl.o
	$(CC) -o $@ $^ $(LDFLAGS)

test: $(BIN_TEST)
	./$(BIN_TEST)

install-deps:
	apt-get update && apt-get install -y opencl-headers ocl-icd-opencl-dev

clean:
	rm -f $(BIN_TEST) src/*.o
