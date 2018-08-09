# makefiles

# HDF5 stuff, pkg-config --cflags hdf5
HDF5INC=-I/usr/include/hdf5/serial
HDF5LIB=-L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5 -lhdf5_hl

# GSL, pkg-config --libs gsl
# GSLLIB=-lgsl -lgslcblas -lm
GSLLIB=

# MKL on linux
# MKL stuff
MKLFLAGS= -m64 -I${MKLROOT}/include
MKLLIB= -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_gnu_thread -lmkl_core -lgomp -lpthread -lm -ldl
#MKLLIB= -L${MKLROOT}/lib/intel64 -Wl,--no-as-needed -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lm -ldl

# Fortran
FC=gfortran
F90FLAGS=-g -c -cpp
F90SRC= 
F90OBJ=$(F90SRC:.f90=.o)

# C++ compiler
CXX=g++
CXXFLAGS=-c -Wall -g -std=c++11 -fbounds-check 
#CXXFLAGS=-c -Wall -g -std=c++11 -DVERBOSE -DDEBUG $(MKLFLAGS) -DUSE_MKL -fbounds-check 
# CFLAGS=-c -Wall -g -std=c++11 -DUSE_MKL -DVERBOSE -fopenmp $(MKLFLAGS)
INCFLAGS=-I../ $(HDF5INC)
LDFLAGS=
CXXSRC= hdf5_ex1.cpp

CXXOBJ=$(CXXSRC:.cpp=.o)

#LIBFLAGS=-lgfortran -lblas -llapack -llapacke $(HDF5LIB) $(GSLLIB)
#LIBFLAGS=-lgfortran $(HDF5LIB) $(MKLLIB)
LIBFLAGS= $(HDF5LIB)

EXECUTABLE=a.out

all: $(F90OBJ) $(CXXOBJ) $(EXECUTABLE)
 
$(EXECUTABLE): $(F90OBJ) $(CXXOBJ)
	$(CXX) $(LDFLAGS) $(F90OBJ) $(CXXOBJ) -o $@ $(LIBFLAGS)

%.o: %.f90
	$(FC) $(F90FLAGS) $(INCFLAGS) $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCFLAGS) $< -o $@

clean:
	rm *.o a.out

