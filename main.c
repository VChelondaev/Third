#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <malloc.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(int argc, char** argv) {

	double accuracy = 0.000001;
	int N, ITER_MAX;
	accuracy = atof(argv[1]);
	N = atoi(argv[2]);
	ITER_MAX = atoi(argv[3]);

	double* arr = (double*)calloc(N * N, sizeof(double));
	double* arr_new = (double*)calloc(N * N, sizeof(double));
	double step = 10.0 / (N-1);
	

	arr[0] = 10;
	arr[N-1] = 20;
	arr[N * (N - 1)] = 20;
	arr[N * N -1] = 30;


	for (int i = 1; i < N; i++) {
		arr[i] = arr[0] + step * i;
		arr[N * (N - 1) + i] = arr[N - 1] + step * i;
		arr[(N * i)] = arr[0] + step * i;
		arr[N - 1 + i * N] = arr[N - 1] + step * i;
	}

	memcpy(arr_new, arr, N * N * sizeof(double));

	int size = N * N;
	int iter = 0;
	double error = 1.0;

	cublasHandle_t handler;
	cublasStatus_t status;

	status = cublasCreate(&handler);

	clock_t start = clock();
#pragma acc enter data copyin(arr[0:size],arr_new[0:size])   //Выделение и копирование памяти на видеокарту
	{
	for (; ((iter < ITER_MAX) && (error > accuracy)); iter++) {  //Условия выполнения кода
		double alpha = -1.0;
		int idx = 0;
	#pragma acc data present(arr,arr_new)  //Обновление указателей
	#pragma acc parallel loop independent collapse(2) vector vector_length(256) gang num_gangs(256)                          // "acc parallel loop"  распараллеливаем,
		for (int i = 1; i < N - 1; i++) {                                                                               // "independent" говорим, что циклы независимы, 
			for (int j = 1; j < N - 1; j++) {                                                                      // "collapse(2)" разворачиваем циклы, 
				int n = i * N + j;                                                      
				arr_new[n] = 0.25 * (arr[n - 1] + arr[n + 1] + arr[(i - 1) * N + j] + arr[(i + 1) * N + j]);
			}
		}
		if (iter % 100 == 0) {
#pragma acc data present (arr, arr_new)                     //Обновление указателей
#pragma acc host_data use_device(arr, arr_new)              //Используем  данные указатели на девайсе
			{
				status = cublasDaxpy(handler, size, &alpha, arr_new, 1, arr, 1);
				status = cublasIdamax(handler, size, arr, 1, &idx);
			}

#pragma acc update host(arr[idx - 1])        //выгружаем значение максимума с девайса на хост
			error = abs(arr[idx - 1]);

#pragma acc host_data use_device(arr, arr_new)    
			status = cublasDcopy(handler, size, arr_new, 1, arr, 1);
		}

		double* temp = arr;
		arr = arr_new;
		arr_new = temp;
		}

	clock_t end = clock();
	printf("%lf\n", 1.0 * (end - start) / CLOCKS_PER_SEC); 
	}
	printf("%0.15lf, %d\n", error, iter);
	
	free(arr);
	free(arr_new);
	cublasDestroy(handler);

	return 0;
}
