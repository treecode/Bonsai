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

clean:
	/bin/rm -f $(PROG) $(OBJ)
