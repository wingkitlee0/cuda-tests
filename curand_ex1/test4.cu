#include <cstdio>
#include <cstdlib>
#include <vector>
#include <memory>
#include <iostream>
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

__device__ float3 generate3(curandState* globalState, int ind)
{
    /*
        generate random x, y, z position for particles
    */
    float3 newposition;
    curandState localState = globalState[ind];
    newposition.x = curand_uniform( &localState ); 
    newposition.y = curand_uniform( &localState ); 
    newposition.z = curand_uniform( &localState ); 
    globalState[ind] = localState;
    return newposition;
}

__global__ void initPosition(int N, float4 *d_par, curandState* globalState)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    while (id < N)
    {
        auto position = generate3(globalState, id);
        d_par[id].w = 1.0f;        // mass
        d_par[id].x = 2.0*position.x-1.0;
        d_par[id].y = 2.0*position.y-1.0;
        d_par[id].z = 2.0*position.z-1.0;      
        id += blockDim.x * gridDim.x;  
    }
}

int main(int argc, char** argv)
{
  int N = 20;
  int *d_y;
  float4 *d_par; // particle positions and mass
  std::vector<int> y(N);
  std::vector<float4> par(N);

  cudaMalloc(&d_y, N * sizeof(int));
  //cudaMemcpy(d_y, &y[0], N * sizeof(int), cudaMemcpyHostToDevice);

  // allocate memory on device
  cudaMalloc(&d_par, N * sizeof(float4)); 
  
  curandState* devStates;
  cudaMalloc (&devStates, N * sizeof(curandState));
  
  //srand(time(0));
  srand(1234);
  /** ADD THESE TWO LINES **/
  int seed = rand();
  setup_kernel<<<2, 100>>>(devStates,seed);
  /** END ADDITION **/
  addToCount<<<2, 100>>>(N, d_y, devStates);
  initPosition<<<2, 100>>>(N, d_par, devStates);

  cudaMemcpy(&y[0], d_y, N*sizeof(int), cudaMemcpyDeviceToHost);
  //cudaMemcpy(y, d_y, N*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(&par[0], d_par, N*sizeof(float4), cudaMemcpyDeviceToHost);

  printf("final = %i\n", y[0]);

  std::cout << "particles:" << std::endl;
  for (auto particle : par) {
      printf("%15.7f %15.7f %15.7f\n", particle.x, particle.y, particle.z);
  }
}
