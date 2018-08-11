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
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform( &localState );
    globalState[ind] = localState;
    return RANDOM;
}

__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
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
    float3 newposition = make_float3(0.0,0.0,0.0);
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
    if (id < N) // final block may not have all the threads
    {
        auto position = generate3(globalState, id);
        d_par[id].w = (float) id / 1000.0;        // mass
        d_par[id].x = 2.0*position.x-1.0;
        d_par[id].y = 2.0*position.y-1.0;
        d_par[id].z = 2.0*position.z-1.0;      
        //id += blockDim.x * gridDim.x; 
    }
}

int main(int argc, char** argv)
{
    int N = 768;
    int *d_y;
    float4 *d_par; // particle positions and mass
    std::vector<int> y(N);
    std::vector<float4> par(N);

    int blocksize = 256; // value usually chosen by tuning and hardware constraints
    int nblocks = ceil( (float) N/blocksize);

    printf("nblocks = %i\n", nblocks);

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
    setup_kernel<<<nblocks, blocksize>>>(devStates,seed);
    /** END ADDITION **/
    addToCount<<<nblocks, blocksize>>>(N, d_y, devStates);
    initPosition<<<nblocks, blocksize>>>(N, d_par, devStates);
    cudaDeviceSynchronize();

    cudaMemcpy(&y[0], d_y, N*sizeof(int), cudaMemcpyDeviceToHost);
    //cudaMemcpy(y, d_y, N*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&par[0], d_par, N*sizeof(float4), cudaMemcpyDeviceToHost);

    printf("final = %i\n", y[0]);

    printf("# %i particles:\n", N);
    for (auto particle : par) {
        printf("%15.7f %15.7f %15.7f %15.7f\n", particle.w, 
        particle.x, particle.y, particle.z);
    }

    // Free the GPU memory here
    cudaFree(d_y);
    cudaFree(d_par);
    cudaFree(devStates);
}
