#include <cstdio>
#include <cstdlib>
#include <vector>

struct BmpHeader
{

  unsigned int  bfSize;
  unsigned short bfReserved1;
  unsigned short bfReserved2;
  unsigned int  bfOffBits;

  unsigned int  biSize;
  unsigned int  biWidth;
  unsigned int  biHeight;
  unsigned short biPlanes;
  unsigned short biBitCount;
  unsigned int  biCompression;
  unsigned int  biSizeImage;
  unsigned int  biXPixPerMeter;
  unsigned int  biYPixPerMeter;
  unsigned int  biClrUsed;
  unsigned int  biClrImporant;

};

int main(int argc, char * argv[])
{
  fprintf(stderr, "Usage: %s  < input.bmp > output.cpp\n",
      argv[0]);
  fprintf(stderr, " Reading BMP ... \n");

  BmpHeader bmp;
  char bfType[2];

  FILE *instream = stdin;
  fread( &bfType, sizeof(char), 2, instream);
  fread( &bmp, sizeof(BmpHeader), 1, instream);
  const int width = (int)bmp.biWidth;
  const int height = (int)bmp.biHeight;
  fprintf(stderr, "width= %d  height= %d\n", width, height);
  std::vector<unsigned char> color_array(width*height*3);
  fread( &color_array[0], sizeof(unsigned char), width*height*3, instream);

  FILE *ostream = stdout;
  fprintf(ostream, "static float colorMap[%d][%d][3] = \n{\n", height, width);
  for (int h = 0; h < height; h++)
  {
    fprintf(ostream, "  { ");
    for (int w = 0; w < width; w++)
    {
      fprintf(ostream, "{%d.0f,%d.0f,%d.0f}%c", 
          (int)color_array[0 + 3*(w+h*width)],
          (int)color_array[1 + 3*(w+h*width)],
          (int)color_array[2 + 3*(w+h*width)],
          w < width-1 ? ',':' ');
    }
    fprintf(ostream, "},\n");
  }
  fprintf(ostream, "};\n");





  return 0;
}
