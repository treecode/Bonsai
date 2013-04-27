
#include <stdio.h>
#include <stdlib.h>

void gen_bulge(int nptcl, int myseed, float *buffer, int verbose);

int main(int argc,char *argv[])
{
  gen_bulge(-1, 1, NULL, 1);
  return 0;
}
