#include<cuda.h>
#include<stdio.h>
#include<stdlib.h>
#include <cuda_runtime.h> 

cudaEvent_t start, stop;
float elapsed_time_ms;

__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width)
{
	int tx = threadIdx.x + blockIdx.x * blockDim.x;
	int ty = threadIdx.y + blockIdx.y * blockDim.y;

	float Pvalue = 0;

	for (int k = 0; k < Width; ++k) {

		float Mdelement = Md[ty * Width + k];
		float Ndelement = Nd[k * Width + tx];
		Pvalue += (Mdelement * Ndelement);
	}

	Pd[ty * Width + tx] = Pvalue;
}

void MatrixMultiplication(float* M, float* N, float* P, int Width)
{
	int size = Width * Width * sizeof(float);
	float* Md, * Nd, * Pd;
	int k = 128;
	int l = 128;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMalloc((void**)&Md, size);
	cudaMemcpy(Md, M, size, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&Nd, size);
	cudaMemcpy(Nd, N, size, cudaMemcpyHostToDevice);

	cudaMalloc((void**)&Pd, size);

	dim3 dimBlock((k - 1) / Width + 1, (l - 1) / Width + 1);
	dim3 dimGrid(Width, Width);

	cudaEventRecord(start, 0);

	MatrixMulKernel << <dimGrid, dimBlock >> > (Md, Nd, Pd, Width);

	cudaMemcpy(P, Pd, size, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed_time_ms, start, stop);

	cudaFree(Md);
	cudaFree(Nd);
	cudaFree(Pd);
}

int main(void)
{

	void MatrixMultiplication(float*, float*, float*, int);

	const int Width = 10000;
	float* M = new float[Width * Width];
	float* N = new float[Width * Width];
	float* P = new float[Width * Width];

	for (int i = 0; i < (Width * Width); i++) {
		M[i] = static_cast<float>(std::rand()) ;
		N[i] = static_cast<float>(std::rand()) ;
		P[i] = 0;
	}

	MatrixMultiplication(M, N, P, Width);

	delete[] M;
	delete[] N;
	delete[] P;

	printf("\n");
	printf("계산 시간: %f ms.", elapsed_time_ms);

	return 0;
}
