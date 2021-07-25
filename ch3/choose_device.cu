#include "../common/book.h"

int main(void) {
  cudaDeviceProp prop;
  int dev;

  HANDLE_ERROR(cudaGetDevice(&dev));
  printf("ID of current CUDA device: %d\n", dev);

  memset(&prop, 0, sizeof(cudaDeviceProp));
  prop.major = 7;
  prop.minor = 5;

  HANDLE_ERROR(cudaChooseDevice(&dev, &prop));
  printf("ID of CUDA device closest to revision %d.%d: %d\n", prop.major,
         prop.minor, dev);
  HANDLE_ERROR(cudaSetDevice(dev));
}