CC = g++ 
CFLAGS = -Wall -O3 -funroll-loops -ffast-math


GEN_IMAGE_VOXEL = gen_image_voxel.o voxel.o display.o 

all: gen_image_voxel


*.o: *.h Makefile

.cpp.o:
	$(CC) $(CFLAGS) -o $@ -c $<


.PHONY: gen_image_voxel
gen_image_voxel: $(GEN_IMAGE_VOXEL)
	$(CC) $(CFLAGS) -o $@ $^  $(LOADLIBS)


.PHONY: clean
clean:
	rm *.o
	rm *~

