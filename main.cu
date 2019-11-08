#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <stdlib.h>
#include <cuda.h>
#include <sys/time.h>
#define BLOCKY 8
#define BLOCKX 8
extern "C" {
    #include "libs/bitmap.h"
}

#define ERROR_EXIT -1

#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %s %d\n", cudaGetErrorName(code), cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Convolutional Filter Examples, each with dimension 3,
// gaussian filter with dimension 5
// If you apply another filter, remember not only to exchange
// the filter but also the filterFactor and the correct dimension.

int const sobelYFilter[] = {-1, -2, -1,
                             0,  0,  0,
                             1,  2,  1};
float const sobelYFilterFactor = (float) 1.0;

int const sobelXFilter[] = {-1, -0, -1,
                            -2,  0, -2,
                            -1,  0, -1 , 0};
float const sobelXFilterFactor = (float) 1.0;


int const laplacian1Filter[] = {  -1,  -4,  -1,
                                 -4,  20,  -4,
                                 -1,  -4,  -1};
int const laplacian1filterDim=3;
float const laplacian1FilterFactor = (float) 1.0;

int const laplacian2Filter[] = { 0,  1,  0,
                                 1, -4,  1,
                                 0,  1,  0};
float const laplacian2FilterFactor = (float) 1.0;

int const laplacian3Filter[] = { -1,  -1,  -1,
                                  -1,   8,  -1,
                                  -1,  -1,  -1};
float const laplacian3FilterFactor = (float) 1.0;


//Bonus Filter:

int const gaussianFilter[] = { 1,  4,  6,  4, 1,
                               4, 16, 24, 16, 4,
                               6, 24, 36, 24, 6,
                               4, 16, 24, 16, 4,
                               1,  4,  6,  4, 1 };

float const gaussianFilterFactor = (float) 1.0 / 256.0;

#define PIXEL(i,j) ((i)+(j)*XSIZE)
// Apply convolutional filter on image data
__global__ void applyFilter(unsigned char *in, unsigned char *out, unsigned int XSIZE, unsigned int YSIZE, int *filter, unsigned int filterDim, float filterFactor) {
  unsigned int const filterCenter = (filterDim / 2);

  __shared__ unsigned char st[BLOCKX*BLOCKY];
  __shared__ int kerneld[laplacian1filterDim*laplacian1filterDim];

  int i = blockIdx.x*BLOCKX+threadIdx.x;
  int j = blockIdx.y*BLOCKY + threadIdx.y;


  if(i>0&&i<XSIZE&&j>0&&j<YSIZE)
  {
        int threadsum=threadIdx.x+threadIdx.y*BLOCKX;
        if(threadsum<filterDim*filterDim)
        {
          kerneld[threadsum]=filter[threadsum];
        }
        __syncthreads();
        st[threadsum]=in[PIXEL(i,j)];
        __syncthreads();


     int aggregate =0;
     for (unsigned int ky = 0; ky < filterDim; ky++) {
        int nky = filterDim - 1 - ky;
        for (unsigned int kx = 0; kx < filterDim; kx++) {
          int nkx = filterDim - 1 - kx;

          int yy = threadIdx.y + (ky - filterCenter);
          int xx = threadIdx.x + (kx - filterCenter);
          int mod= kerneld[nky * filterDim + nkx];
          if (xx >= 0 && xx < BLOCKX && yy >=0 && yy < BLOCKY)
          {
            aggregate += st[xx+yy*BLOCKX] * mod;
          }
          else
          {
            yy = j + (ky - filterCenter);
            xx = i + (kx - filterCenter);
            aggregate += in[PIXEL(xx,yy)] * mod;
          }

        }

      }
      aggregate *= filterFactor;
      if (aggregate > 0) {
        out[PIXEL(i,j)] = (aggregate > 255) ? 255 : aggregate;
      } else {
        out[PIXEL(i,j)] = 0;
      }

  }

}

__global__ void applyFilterNormal(unsigned char *in, unsigned char *out, unsigned int XSIZE, unsigned int YSIZE, int *filter, unsigned int filterDim, float filterFactor) {
  unsigned int const filterCenter = (filterDim / 2);

  int i = blockIdx.x*BLOCKX+threadIdx.x;
  int j = blockIdx.y*BLOCKY + threadIdx.y;

  if(i>0&&i<XSIZE&&j>0&&j<YSIZE)
  {
     int aggregate =0;

     for (unsigned int ky = 0; ky < filterDim; ky++) {
        int nky = filterDim - 1 - ky;
        for (unsigned int kx = 0; kx < filterDim; kx++) {
          int nkx = filterDim - 1 - kx;

          int yy = j + (ky - filterCenter);
          int xx = i + (kx - filterCenter);
          aggregate += in[PIXEL(xx,yy)] * filter[nky * filterDim + nkx];

        }

      }
      aggregate *= filterFactor;
      if (aggregate > 0) {
        out[PIXEL(i,j)] = (aggregate > 255) ? 255 : aggregate;
      } else {
        out[PIXEL(i,j)] = 0;
      }
  }

}






void help(char const *exec, char const opt, char const *optarg) {
    FILE *out = stdout;
    if (opt != 0) {
        out = stderr;
        if (optarg) {
            fprintf(out, "Invalid parameter - %c %s\n", opt, optarg);
        } else {
            fprintf(out, "Invalid parameter - %c\n", opt);
        }
    }
    fprintf(out, "%s [options] <input-bmp> <output-bmp>\n", exec);
    fprintf(out, "\n");
    fprintf(out, "Options:\n");
    fprintf(out, "  -i, --iterations <iterations>    number of iterations (1)\n");

    fprintf(out, "\n");
    fprintf(out, "Example: %s in.bmp out.bmp -i 10000\n", exec);
}

int main(int argc, char **argv) {
  /*
    Parameter parsing, don't change this!
   */
  unsigned int iterations = 1;
  char *output = NULL;
  char *input = NULL;
  int ret = 0;

  static struct option const long_options[] =  {
      {"help",       no_argument,       0, 'h'},
      {"iterations", required_argument, 0, 'i'},
      {0, 0, 0, 0}
  };

  static char const * short_options = "hi:";
  {
    char *endptr;
    int c;
    int option_index = 0;
    while ((c = getopt_long(argc, argv, short_options, long_options, &option_index)) != -1) {
      switch (c) {
      case 'h':
        help(argv[0],0, NULL);
        return 0;
      case 'i':
        iterations = strtol(optarg, &endptr, 10);
        if (endptr == optarg) {
          help(argv[0], c, optarg);
          return ERROR_EXIT;
        }
        break;
      default:
        abort();
      }
    }
  }

  if (argc <= (optind+1)) {
    help(argv[0],' ',"Not enough arugments");
    return ERROR_EXIT;
  }
  input = (char *)calloc(strlen(argv[optind]) + 1, sizeof(char));
  strncpy(input, argv[optind], strlen(argv[optind]));
  optind++;

  output = (char *)calloc(strlen(argv[optind]) + 1, sizeof(char));
  strncpy(output, argv[optind], strlen(argv[optind]));
  optind++;

  /*
    End of Parameter parsing!
   */

  /*
    Create the BMP image and load it from disk.
   */
  bmpImage *image = newBmpImage(0,0);
  if (image == NULL) {
    fprintf(stderr, "Could not allocate new image!\n");
  }

  if (loadBmpImage(image, input) != 0) {
    fprintf(stderr, "Could not load bmp image '%s'!\n", input);
    freeBmpImage(image);
    return ERROR_EXIT;
  }


  // Create a single color channel image. It is easier to work just with one color
  bmpImageChannel *imageChannel = newBmpImageChannel(image->width, image->height);
  if (imageChannel == NULL) {
    fprintf(stderr, "Could not allocate new image channel!\n");
    freeBmpImage(image);
    return ERROR_EXIT;
  }

  // Extract from the loaded image an average over all colors - nothing else than
  // a black and white representation
  // extractImageChannel and mapImageChannel need the images to be in the exact
  // same dimensions!
  // Other prepared extraction functions are extractRed, extractGreen, extractBlue
  if(extractImageChannel(imageChannel, image, extractAverage) != 0) {
    fprintf(stderr, "Could not extract image channel!\n");
    freeBmpImage(image);
    freeBmpImageChannel(imageChannel);
    return ERROR_EXIT;
  }


  //Here we do the actual computation!
  // imageChannel->data is a 2-dimensional array of unsigned char which is accessed row first ([y][x])
  int XSIZE=imageChannel->width;
  int YSIZE=imageChannel->height;
  unsigned char *cudachannel1;
  unsigned char *cudachannel2;
  int *kernel;

  int size = sizeof(unsigned char)*XSIZE*YSIZE;

  cudaMalloc(&cudachannel1, size);
  cudaMalloc(&cudachannel2, size);
  cudaMalloc(&kernel, sizeof(laplacian1Filter));


  int iXSIZE=XSIZE;
  int iYSIZE=YSIZE;
  while(iXSIZE%BLOCKX!=0)
  {
     iXSIZE++;
  }
  while(iYSIZE%BLOCKY!=0)
  {
     iYSIZE++;
  }

  dim3 gridBlock(iXSIZE/BLOCKX, iYSIZE/BLOCKY);
  dim3 threadBlock(BLOCKX, BLOCKY);

  unsigned char *deviceMem = (unsigned char*)malloc(size);
  for(int x=0;x<XSIZE;x++)
  {
    for(int y=0;y<YSIZE;y++)
    {
      deviceMem[y*XSIZE+x]=imageChannel->data[y][x];
    }
  }
  cudaMemcpy(kernel, laplacian1Filter, sizeof(laplacian1Filter),cudaMemcpyHostToDevice);

  for (unsigned int i = 0; i < iterations; i ++) {
    cudaMemcpy(cudachannel1,deviceMem, size,cudaMemcpyHostToDevice);
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
    applyFilter<<<gridBlock,threadBlock>>>(cudachannel1,cudachannel2,XSIZE,YSIZE,kernel,3, laplacian1FilterFactor);
    clock_gettime(CLOCK_REALTIME, &end);

    if (end.tv_nsec < start.tv_nsec) {
              end.tv_nsec += 1000000000;
              end.tv_sec--;
          }

          printf("%ld.%09ld  GPU time\n", (long)(end.tv_sec - start.tv_sec),
              end.tv_nsec - start.tv_nsec);
    cudaMemcpy(deviceMem, cudachannel2, size,cudaMemcpyDeviceToHost);

  }
  cudaFree(kernel);
  cudaFree(cudachannel1);
  cudaFree(cudachannel2);

  for(int x=0;x<XSIZE;x++)
  {
    for(int y=0;y<YSIZE;y++)
    {
      imageChannel->data[y][x]=deviceMem[y*XSIZE+x];
    }
  }
  free(deviceMem);
  // Map our single color image back to a normal BMP image with 3 color channels
  // mapEqual puts the color value on all three channels the same way
  // other mapping functions are mapRed, mapGreen, mapBlue


  if (mapImageChannel(image, imageChannel, mapEqual) != 0) {
    fprintf(stderr, "Could not map image channel!\n");
    freeBmpImage(image);
    freeBmpImageChannel(imageChannel);
    return ERROR_EXIT;
  }
  freeBmpImageChannel(imageChannel);

  //Write the image back to disk
  if (saveBmpImage(image, output) != 0) {
    fprintf(stderr, "Could not save output to '%s'!\n", output);
    freeBmpImage(image);
    return ERROR_EXIT;
  };

  ret = 0;
  if (input)
    free(input);
  if (output)
    free(output);
  return ret;
};
