typedef struct {
    size_t width, height;
    unsigned char *data;
} Image;

Image *loadPPM(const char *filename);
