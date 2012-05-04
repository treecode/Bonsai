#include <GL/glew.h>
#include <GL/freeglut.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "loadPPM.h"

void flipImage(Image *img)
{
    int lineSize = (int) img->width*3;
    int imageSize = lineSize * (int) img->height;

    GLubyte *top = img->data;
    GLubyte *bottom = top + (imageSize - lineSize);

    GLubyte *tempBuf = new GLubyte[lineSize];

    for(int i=0; i<img->height/2; i++) {
        // swap top and bottom
        memcpy(tempBuf, top, lineSize);
        memcpy(top, bottom, lineSize);
        memcpy(bottom, tempBuf, lineSize);

        top += lineSize;
        bottom -= lineSize;
    }

    delete [] tempBuf;
}

GLuint createCubemapTexture(Image **images)
{
    GLuint tex; 
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_CUBE_MAP, tex);

    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    //glTexParameterf(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_ANISOTROPY_EXT, aniso);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_GENERATE_MIPMAP_SGIS, GL_TRUE);

    // load face data
    for(int i=0; i<6; i++) {
        glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0,
                     GL_RGBA, (GLsizei) images[i]->width, (GLsizei) images[i]->height, 0, 
                     GL_RGB, GL_UNSIGNED_BYTE, images[i]->data);
    }

    return tex;
}

bool convertCrossToCubemap(Image *img, Image **faces)
{
    //this function only supports vertical cross format for now (3 wide by 4 high)
    if ( (img->width / 3 != img->height / 4) || (img->width % 3 != 0) || (img->height % 4 != 0) )
        return false;

    //get the source data
    GLubyte *data = img->data;
    int _width = (int) img->width;
    int _height = (int) img->height;
    int _elementSize = 3;

    int fWidth = _width / 3;
    int fHeight = _height / 4;
    printf("face size: %d x %d\n", fWidth, fHeight);
    //extract the faces

    for(int i=0; i<6; i++) {
        faces[i] = (Image *) malloc(sizeof(Image));
        faces[i]->data = new GLubyte[ fWidth * fHeight * _elementSize]; 
        faces[i]->width = fWidth;
        faces[i]->height = fHeight;
    }

    // positive X
    GLubyte *ptr = faces[0]->data;
    for (int j=0; j<fHeight; j++) {
        memcpy( ptr, &data[((_height - (fHeight + j + 1))*_width + 2 * fWidth) * _elementSize], fWidth*_elementSize);
        ptr += fWidth*_elementSize;
    }

    // negative X
    ptr = faces[1]->data;
    for (int j=0; j<fHeight; j++) {
        memcpy( ptr, &data[(_height - (fHeight + j + 1))*_width*_elementSize], fWidth*_elementSize);
        ptr += fWidth*_elementSize;
    }

    // positive Y
    ptr = faces[2]->data;
    for (int j=0; j<fHeight; j++) {
        memcpy( ptr, &data[((4 * fHeight - j - 1)*_width + fWidth)*_elementSize], fWidth*_elementSize);
        ptr += fWidth*_elementSize;
    }

    // negative Y
    ptr = faces[3]->data;
    for (int j=0; j<fHeight; j++) {
        memcpy( ptr, &data[((2*fHeight - j - 1)*_width + fWidth)*_elementSize], fWidth*_elementSize);
        ptr += fWidth*_elementSize;
    }

    // positive Z
    ptr = faces[4]->data;
    for (int j=0; j<fHeight; j++) {
        memcpy( ptr, &data[((_height - (fHeight + j + 1))*_width + fWidth) * _elementSize], fWidth*_elementSize);
        ptr += fWidth*_elementSize;
    }

    // negative Z
    ptr = faces[5]->data;
    for (int j=0; j<fHeight; j++) {
        for (int i=0; i<fWidth; i++) {
            memcpy( ptr, &data[(j*_width + 2 * fWidth - (i + 1))*_elementSize], _elementSize);
            ptr += _elementSize;
        }
    }

    return true;
}

GLuint loadCubemap(char *filenameFormat)
{
    // load faces
    Image *faces[6];
    for(int i=0; i<6; i++) {
        char filename[256];
        sprintf(filename, filenameFormat, i+1);
        faces[i] = loadPPM(filename);
        flipImage(faces[i]);
    }

    GLuint tex = createCubemapTexture(faces);

    for(int i=0; i<6; i++) {
        free(faces[i]->data);
        free(faces[i]);
    }

    return tex;
}

GLuint loadCubemapCross(char *filename)
{
    Image *cross = loadPPM(filename);
    if (!cross) {
        return 0;
    }
    flipImage(cross);

    Image *faces[6];
    bool success = convertCrossToCubemap(cross, faces);
    if (success) {
        GLuint tex = createCubemapTexture(faces);
        return tex;
    }

    return 0;
}
