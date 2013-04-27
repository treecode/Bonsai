
#include <stdio.h>
#include <stdlib.h>

void gen_halo(int nptcl, int myseed, float *buffer, int verbose);

int main(int argc,char *argv[])
{
  gen_halo(-1, 1, NULL, 1);
  return 0;
}
