PROG=write_test
OBJ=write_test.o

CXX=mpiicpc
LD=mpiicpc
CXXFLAGS=-g -O2 -Wall `sionconfig --cflags --mpi --be --64`
LDFLAGS=`sionconfig --libs --mpi --be --64`

all: $(PROG)

$(PROG): $(OBJ) $(OBJS)
	$(LD) -o $@ $< $(LDFLAGS)

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -o $@ -c $<

write_test.o: write_snapshot.h sion_write_snapshot.h

run: all
	mpirun -np 4 ./write_test 5000000

clean:
	/bin/rm -f $(PROG) $(OBJ)

write_test: write_snapshot.h sion_write_snapshot.h
