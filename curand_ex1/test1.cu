#include <cstdio>
#include <cstdlib>
#include <vector>
#include <memory>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

__device__ float generate(curandState* globalState, int ind)
{
    //int ind = threadIdx.x;
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState;
    return RANDOM;
}

__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x;
    curand_init ( seed, id, 0, &state[id] );
}

__global__ void addToCount(int N, int *y, curandState* globalState)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
while (id < N)
{
    int number = generate(globalState, id) * 1000000;
    printf("%i\n", number);

    atomicAdd(&(y[0]), number);
    id += blockDim.x * gridDim.x;
}
}

int main(void)
{
  int N = 10;
  int *y, *d_y;
  y = (int*)malloc(N*sizeof(int));

  cudaMalloc(&d_y, N * sizeof(int));
  cudaMemcpy(d_y, y, N * sizeof(int), cudaMemcpyHostToDevice);

  curandState* devStates;
  cudaMalloc (&devStates, N * sizeof(curandState));
  srand(time(0));
  /** ADD THESE TWO LINES **/
  int seed = rand();
  setup_kernel<<<2, N>>>(devStates,seed);
  /** END ADDITION **/
  addToCount<<<2, N>>>(N, d_y, devStates);

  cudaMemcpy(y, d_y, N*sizeof(int), cudaMemcpyDeviceToHost);
  printf("final = %i\n", *y);
}
