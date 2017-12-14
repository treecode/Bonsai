PROG=write_test
OBJ=write_test.o

CXX=mpicxx
LD=mpicxx

CXXFLAGS=-O3 -Wall


all: $(PROG)

$(PROG): $(OBJ)
	$(LD) $(LDFLAGS) -o $@ $<

$(OBJS): %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

run: all
	mpirun -np 4 ./write_test 5000000

clean:
	/bin/rm -f $(PROG) $(OBJ)

$(OBJS): write_snapshot.h sion_write_snapshot.h
