all: checkEnv clean setFlags openmp_async_c openmp_async_f

.PHONY: checkEnv clean

setFlags:
ifdef CRAY_PRGENVPGI
    CFLAGS = -mp
endif
ifdef CRAY_PRGENVGNU
    CFLAGS = -fopenmp
endif
ifdef CRAY_PRGENVINTEL
    CFLAGS = -openmp
endif

openmp_async_c: launcher.c sleep.cu
	mkdir -p bin
	nvcc -arch=sm_35 -c sleep.cu
	cc $(CFLAGS) sleep.o launcher.c -o bin/openmp_async_c.out

openmp_async_f: launcher.f90 sleep.cuf
	mkdir -p bin
	ftn $(FLAGS) -acc -ta=nvidia,kepler sleep.cuf launcher.f90 -o bin/openmp_async_f.out

checkEnv:
ifndef CRAY_CUDATOOLKIT_VERSION
	$(error cudatoolkit module not loaded)
endif

clean:
	rm -rf bin
	rm -f *.o
