#include <stdio.h>
#include <stdlib.h>
#include "loadPPM.h"

Image *loadPPM(const char *filename)
{
    char buff[16];
    Image *result;
    FILE *fp;
    int maxval;

    fp = fopen(filename, "rb");
    if (!fp)
    {
	    fprintf(stderr, "Unable to open file `%s'\n", filename);
	    exit(1);
    }

    if (!fgets(buff, sizeof(buff), fp))
    {
	    fprintf(stderr, "Error opening file '%s'\n", filename);
	    exit(1);
    }

    if (buff[0] != 'P' || buff[1] != '6')
    {
	    fprintf(stderr, "Invalid image format (must be `P6')\n");
	    exit(1);
    }

    int c = fgetc(fp);
    if (c == '#') {
        // skip comment
        char str[256];
        fgets(str, 256, fp);
    } else {
        ungetc(c, fp);
    }

    result = (Image *) malloc(sizeof(Image));
    if (!result)
    {
	    fprintf(stderr, "Unable to allocate memory\n");
	    exit(1);
    }

    int r = 0;
    if (8 == sizeof(void*))
       r = fscanf(fp, "%llu %llu", &result->width, &result->height);
    else
      r = fscanf(fp, "%lu %lu", &result->width, &result->height);

    if (r != 2)
    {
	    fprintf(stderr, "Error loading image `%s'\n", filename);
	    exit(1);
    }
    while (fgetc(fp) != '\n');

    if (fscanf(fp, "%d", &maxval) != 1)
    {
	    fprintf(stderr, "Error loading image `%s'\n", filename);
	    exit(1);
    }
    while (fgetc(fp) != '\n');

    result->data = (unsigned char *) malloc(3 * result->width * result->height);
    if (!result->data)
    {
	    fprintf(stderr, "Unable to allocate memory\n");
	    exit(1);
    }

    if (fread(result->data, 3 * result->width, result->height, fp) != result->height)
    {
	    fprintf(stderr, "Error loading image `%s'\n", filename);
	    exit(1);
    }

    fprintf(stdout, "Loaded `%s', %d x %d\n", filename, result->width, result->height);

    fclose(fp);

    return result;
}
