#include <algorithm>
#include <iostream>
#include <sstream>
#include <fstream>
#include <utility>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <cmath>
#include <cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <ctime>
#include <cassert>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#define WIN32_LEAN_AND_MEAN
#define pb push_back 
#define all(c) (c).begin(),(c).end()
#include <Windows.h>
#include <MMSystem.h>
#pragma comment(lib, "winmm.lib")
using namespace std;

#define _DTH cudaMemcpyDeviceToHost
#define _HTD cudaMemcpyHostToDevice
#define _DTD cudaMemcpyDeviceToDevice

#define THREADS 256
#define INF (1<<30)
#define NUM_CARS 400
#define MXC NUM_CARS*2
#define DO_GPU 1

bool InitMMTimer(UINT wTimerRes);
void DestroyMMTimer(UINT wTimerRes, bool init);

inline int _3d_flat(int i, int j, int k, int D1,int D0){return D0*(i*D1+j)+k;}//Ciao ai miei amici italiani!
void _gen_random_test_set(int *CarExitOrder, const int num_cars);
//const int EXAMPLE_0_EXIT_ORDER[NUM_CARS]={0, 2, 30, 6, 8, 3, 12, 14, 16, 18, 20, 22, 24, 26, 28, 4, 32, 41, 36, 38, 40, 42, 44, 46, 48, 29, 10, 5, 7, 9, 49, 13, 15, 17, 19, 21, 23, 25, 27, 1, 31, 33, 35, 37, 39, 34, 43, 45, 47, 11};

int CPU_version(const int *corder, int *DP, const int N){
	int ans=INF;
	for(int i=0;i<=N;i++)for(int j=0;j<=N;j++)for(int k=0;k<=N;k++){
		DP[_3d_flat(i,j,k,N+1,N+1)]=MXC;
	}
	DP[_3d_flat(0,N,N,N+1,N+1)]=0;
	for(int i=0;i<N;i++){
		for(int j=0;j<=N;j++){
			for(int k=0;k<=N;k++){
				if(j<=corder[i])DP[_3d_flat(i+1,j,k,N+1,N+1)]=min(DP[_3d_flat(i+1,j,k,N+1,N+1)],1+DP[_3d_flat(i,j,k,N+1,N+1)]);
				else
					DP[_3d_flat(i+1,corder[i],k,N+1,N+1)]=min(DP[_3d_flat(i+1,corder[i],k,N+1,N+1)],DP[_3d_flat(i,j,k,N+1,N+1)]);

				if(k<=corder[i])DP[_3d_flat(i+1,j,k,N+1,N+1)]=min(DP[_3d_flat(i+1,j,k,N+1,N+1)],1+DP[_3d_flat(i,j,k,N+1,N+1)]);
				else
					DP[_3d_flat(i+1,j,corder[i],N+1,N+1)]=min(DP[_3d_flat(i+1,j,corder[i],N+1,N+1)],DP[_3d_flat(i,j,k,N+1,N+1)]);
			}
		}
	}
	for(int j=0;j<=N;j++)for(int k=0;k<=N;k++){
		ans=min(ans,DP[_3d_flat(N,j,k,N+1,N+1)]);
	}

	return ans;
}
__device__ int D_3d_flat(int i, int j, int k, int D1,int D0){return D0*(i*D1+j)+k;}

__global__ void GPU_mem_op(int *DP,const int bound,const int zloc){
	const int offset=threadIdx.x+blockIdx.x*blockDim.x;
	if(offset<bound){
		DP[offset]= (offset!=zloc) ? MXC:0;
	}
}

__global__ void GPU_step0(int *DP, const int corder_val, const int i, const int N){
	const int j=threadIdx.x+blockIdx.x*blockDim.x;
	if(j>N)return;
	const int k=blockIdx.y;
	const int idx1=D_3d_flat(i+1,j,k,N+1,N+1);
	const int pval=DP[D_3d_flat(i,j,k,N+1,N+1)];

	if(j<=corder_val)atomicMin(&DP[idx1],(1+pval));
	else
		atomicMin(&DP[D_3d_flat(i+1,corder_val,k,N+1,N+1)],pval);

	if(k<=corder_val)atomicMin(&DP[idx1],(1+pval));
	else
		atomicMin(&DP[D_3d_flat(i+1,j,corder_val,N+1,N+1)],pval);
}

__global__ void GPU_step1(const int *DP, const int N,int *ans){//Hallo aan mijn Nederlandse vrienden!
	const int j=threadIdx.x+blockIdx.x*blockDim.x;
	const int k=blockIdx.y;

	__shared__ volatile int best[THREADS];

	best[threadIdx.x]= (j>N) ? MXC:(DP[D_3d_flat(N,j,k,N+1,N+1)]);
	__syncthreads();

	if(threadIdx.x<128){
		best[threadIdx.x] = (best[threadIdx.x+128] < best[threadIdx.x]) ? best[threadIdx.x+128] : best[threadIdx.x];
	}
	__syncthreads();

	if(threadIdx.x<64){
		best[threadIdx.x] = (best[threadIdx.x+64] < best[threadIdx.x]) ? best[threadIdx.x+64] : best[threadIdx.x];
	}
	__syncthreads();

	if(threadIdx.x<32){
		best[threadIdx.x] = (best[threadIdx.x+32] < best[threadIdx.x]) ? best[threadIdx.x+32] : best[threadIdx.x];
		best[threadIdx.x] = (best[threadIdx.x+16] < best[threadIdx.x]) ? best[threadIdx.x+16] : best[threadIdx.x];
		best[threadIdx.x] = (best[threadIdx.x+8] < best[threadIdx.x]) ? best[threadIdx.x+8] : best[threadIdx.x];
		best[threadIdx.x] = (best[threadIdx.x+4] < best[threadIdx.x]) ? best[threadIdx.x+4] : best[threadIdx.x];
		best[threadIdx.x] = (best[threadIdx.x+2] < best[threadIdx.x]) ? best[threadIdx.x+2] : best[threadIdx.x];
		best[threadIdx.x] = (best[threadIdx.x+1] < best[threadIdx.x]) ? best[threadIdx.x+1] : best[threadIdx.x];	
	}
	__syncthreads();

	if(threadIdx.x==0){
		atomicMin(&ans[0],best[0]);
	}
}


int main(){
	char ch;
	srand(time(NULL));
	const int DPSPACE=(NUM_CARS+1)*(NUM_CARS+1)*(NUM_CARS+1);
	const unsigned int num_bytes=DPSPACE*sizeof(int);
	int *orderMapping=(int *)malloc(NUM_CARS*sizeof(int));
	int *CarExitOrder=(int *)malloc(NUM_CARS*sizeof(int));
	int *H_DP=(int *)malloc(num_bytes);

	_gen_random_test_set(CarExitOrder,NUM_CARS);
	//pre-processing for both routines(not counted because less than 1 ms)
	for(int i=0;i<NUM_CARS;i++){
		orderMapping[CarExitOrder[i]]=i;
	}

	int CPU_ans=MXC,GPU_ans=MXC;
	cout<<"\nRunning CPU implementation..\n";
	UINT wTimerRes = 0;
	DWORD CPU_time=0,GPU_time=0;
	bool init = InitMMTimer(wTimerRes);
	DWORD startTime=timeGetTime();

	CPU_ans=CPU_version(orderMapping,H_DP,NUM_CARS);

	DWORD endTime = timeGetTime();
	CPU_time=endTime-startTime;
	cout<<"CPU solution timing: "<<CPU_time<<" , answer = "<<CPU_ans<<'\n';
	DestroyMMTimer(wTimerRes, init);

	//GPU
	int compute_capability=0;
	cudaDeviceProp deviceProp;
	cudaError_t err=cudaGetDeviceProperties(&deviceProp, compute_capability);
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	string ss= (deviceProp.major>=3 && deviceProp.minor>=5) ? "Capable!\n":"Not Sufficient compute capability!\n";
	cout<<ss;

	/*err=cudaDeviceReset();
	if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}*/
	cout<<"\nRunning GPU implementation..\n";//Hourra! Le français évidemment ne pense pas que tous les Américains sont des idiots!
	if(DO_GPU && (deviceProp.major>=3 && deviceProp.minor>=5)){
		const int bound0=DPSPACE,N=NUM_CARS;
		const int zloc=_3d_flat(0,N,N,N+1,N+1);
		int ii=0;
		dim3 dimGrid0(((N+1)+THREADS-1)/THREADS,N+1,1);
		int *D_DP,*D_ans;
		err=cudaMalloc((void**)&D_DP,num_bytes);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaMalloc((void**)&D_ans,sizeof(int));
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		//Deutsch Bier ist so gut wie deutsche Frauen!
		wTimerRes = 0;
		init = InitMMTimer(wTimerRes);
		startTime = timeGetTime();
		
		GPU_mem_op<<<((bound0+THREADS-1)/THREADS),THREADS>>>(D_DP,bound0,zloc);
		err = cudaThreadSynchronize();
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		err=cudaMemcpy(D_ans,&GPU_ans,sizeof(int),_HTD);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

	
		for(;ii<N;ii++){
			GPU_step0<<<dimGrid0,THREADS>>>(D_DP,orderMapping[ii],ii,N);
			err = cudaThreadSynchronize();
			if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		}

		GPU_step1<<<dimGrid0,THREADS>>>(D_DP,N,D_ans);

		err=cudaMemcpy(&GPU_ans,D_ans,sizeof(int),_DTH);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}

		endTime = timeGetTime();
		GPU_time=endTime-startTime;
		cout<<"GPU timing: "<<GPU_time<<" , answer = "<<GPU_ans<<'\n';
	

		err=cudaFree(D_DP);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
		err=cudaFree(D_ans);
		if(err!=cudaSuccess){printf("%s in %s at line %d\n",cudaGetErrorString(err),__FILE__,__LINE__);}
	}
	if(CPU_ans==GPU_ans){
		cout<<"Success! CPU answer matches GPU answer! The CUDA GPU implementation was "<<double(CPU_time)/double(GPU_time)<<" faster than the serial CPU implementation!\n";
	}else{
		cout<<"Error in calculation!\n";
	}

	free(orderMapping);
	free(CarExitOrder);
	free(H_DP);
	std::cin>>ch;
	return 0;
}

bool InitMMTimer(UINT wTimerRes){
	TIMECAPS tc;
	if (timeGetDevCaps(&tc, sizeof(TIMECAPS)) != TIMERR_NOERROR) {return false;}
	wTimerRes = min(max(tc.wPeriodMin, 1), tc.wPeriodMax);
	timeBeginPeriod(wTimerRes); 
	return true;
}
void DestroyMMTimer(UINT wTimerRes, bool init){
	if(init)
		timeEndPeriod(wTimerRes);
}
void _gen_random_test_set(int *CarExitOrder, const int num_cars){
	bool *b=(bool *)malloc(num_cars*sizeof(bool));
	memset(b,false,num_cars*sizeof(bool));
	int w=-1;
	for(int i=0;i<num_cars;i++){
		do{
			w=rand()%num_cars;
		}while(b[w]);
		b[w]=true;
		CarExitOrder[i]=w;
	}
	free(b);
}






