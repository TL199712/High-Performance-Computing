#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>

/*
	version note:
	1. This version employs the threads and shared memory
		each column is a block, each entry of which is calculated by a thread within this block 
	2. SOR_update cannot be used due to the racing conditions.
   */

/*
	int main(int argc, const char* argv[])

	Compute the diffustion equation with sinks and sources
	
inputs: argc should be 5
argv[1]: grid size (N)
argv[2]: relaxation parameter (omega)
argv[3]: tolerence (tol)
argv[4]: maximum number of iterations (K)
 */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double get_max(double*a,int n);
__global__ void SOR_update_r(double*u,double*r_u,double*omega,double*coef,double*dx,double*lambda);
__global__ void SOR_update_b(double*u,double*r_u,double*omega,double*coef,double*dx,double*lambda);

int main(int argc, char* argv[]){
	if(argc<5){printf("Too few arguments.\n");return 1;}
	if(argc>5){printf("Too many argumentsn\n");return 1;}

	double lambda = 100.0;
	double pi = M_PI;
	double coef = 10.0 * lambda / sqrt(pi); // coefficient of sinks and sources

	int N,K;
	double omega,tol;

	N = atoi(argv[1]);
	omega = atof(argv[2]);
	tol = atof(argv[3]);
	K = atoi(argv[4]);

	if(N>256){printf("Try a smaller N.\n");return 1;}

	printf("N=%d\n",N);
	printf("omega=%f\n",omega);
	printf("tol=%f\n",tol);
	printf("K=%d\n",K);

	// Initialize in the host
	double *u,*r_u;

	u = (double*)malloc((2*N-1)*N*sizeof(double));
	r_u = (double*)malloc((2*N-1)*N*sizeof(double));

	for(int i=0;i<(2*N-1)*N;i++){*(u+i)=0;*(r_u+i)=0;}

	double dx = 2.0/(N-1);

	// Initialize in the device
	double* dev_u, *dev_r_u;
	
	cudaMalloc((void**)&dev_u,(2*N-1)*N*sizeof(double));
	cudaMalloc((void**)&dev_r_u,(2*N-1)*N*sizeof(double));

	double *dev_dx,*dev_coef,*dev_omega,*dev_lambda;

	cudaMalloc((void**)&dev_dx,sizeof(double));
	cudaMalloc((void**)&dev_coef,sizeof(double));
	cudaMalloc((void**)&dev_omega,sizeof(double));
	cudaMalloc((void**)&dev_lambda,sizeof(double));

	// Copy data from the host to the device
	cudaMemcpy(dev_u,u,(2*N-1)*N*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_r_u,r_u,(2*N-1)*N*sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_dx,&dx,sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_coef,&coef,sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_omega,&omega,sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_lambda,&lambda,sizeof(double),cudaMemcpyHostToDevice);

	// Compute the solution in device
	
	double err=1.0;
	int iters=0;
	dim3 blockmesh(2*N-1), threadmesh(N);

	while(err>tol && iters<K){
		SOR_update_r<<<blockmesh,threadmesh>>>(dev_u,dev_r_u,dev_omega,dev_coef,dev_dx,dev_lambda);
		SOR_update_b<<<blockmesh,threadmesh>>>(dev_u,dev_r_u,dev_omega,dev_coef,dev_dx,dev_lambda);
		iters++;

		// check the error every 5 loops
		if(iters%5==0){
			cudaMemcpy(r_u,dev_r_u,(2*N-1)*N*sizeof(double),cudaMemcpyDeviceToHost);
			err = get_max(r_u,(2*N-1)*N);
		}
	}
	printf("err=%1.15f\n",err);
	printf("%d iterations to achieve the desired tolerance.\n",iters);
	// Copy the data from the device to the host
	cudaMemcpy(u,dev_u,(2*N-1)*N*sizeof(double),cudaMemcpyDeviceToHost);

	FILE *fpt = fopen("Sources.out","wb");
	fwrite(u,sizeof(double),(2*N-1)*N,fpt);
	fclose(fpt);

	// Free all the pointers
	cudaFree(dev_dx);
	cudaFree(dev_coef);
	cudaFree(dev_omega);
	cudaFree(dev_lambda);
	cudaFree(dev_u);
	cudaFree(dev_r_u);

	free(u);
	free(r_u);

	return 0;
}

double get_max(double*a,int n){
	double temp = 0;
	for(int i=0;i<n;i++){if(abs(*(a+i))>temp) temp=abs(*(a+i));}
	return temp;
}

__global__ void SOR_update_r(double*u,double*r_u,double*omega,double*coef,double*dx,double*lambda){
	int xid=blockIdx.x, yid=threadIdx.x;
	double x = -2.0+xid*(*dx), y=-1.0+yid*(*dx);	
	double source = (*coef)*exp(-(*lambda)*(*lambda)*y*y)*(exp(-(*lambda)*(*lambda)*(x-1)*(x-1))-exp(-(*lambda)*(*lambda)*(x+1)*(x+1)));
	double res=0;

	__shared__ double left[256],right[256],localu[256];

	if(xid>0&&xid<gridDim.x){
		left[yid]=u[xid-1+yid*gridDim.x];
		right[yid]=u[xid+1+yid*gridDim.x];
	}
	else if(xid==0){
		right[yid]=u[xid+1+yid*gridDim.x];
	}
	else{
		left[yid]=u[xid-1+yid*gridDim.x];
	}
	localu[yid]=u[xid+yid*gridDim.x];

	__syncthreads();

	// calculate the residuals and update the solution

	int br_indicator = xid+yid;
	if(1==br_indicator%2){
		// left bound
		if(xid==0&&yid>0&&yid<blockDim.x-1){
			res = 1.0/4*(right[yid]-localu[yid]) + 1.0/4*(localu[yid+1]-2*localu[yid]+localu[yid-1])-(*dx)*(*dx)/4*source;
		}
		// right bound
		else if(xid==gridDim.x-1&&yid>0&&yid<blockDim.x-1){
			res = 1.0/4*(left[yid]-localu[yid]) + 1.0/4*(localu[yid+1]-2*localu[yid]+localu[yid-1])-(*dx)*(*dx)/4*source;
		}
		//internal
		else if(yid>0&&yid<blockDim.x-1){
			res = 1.0/4*(left[yid]-2*localu[yid]+right[yid]) + 1.0/4*(localu[yid+1]-2*localu[yid]+localu[yid-1])-(*dx)*(*dx)/4*source;
		}

		__syncthreads();
		// update
		r_u[xid+yid*gridDim.x]=res;
		u[xid+yid*gridDim.x]+=(*omega)*res;
	}
}
__global__ void SOR_update_b(double*u,double*r_u,double*omega,double*coef,double*dx,double*lambda){
	int xid=blockIdx.x, yid=threadIdx.x;
	double x = -2.0+xid*(*dx), y=-1.0+yid*(*dx);	
	double source = (*coef)*exp(-(*lambda)*(*lambda)*y*y)*(exp(-(*lambda)*(*lambda)*(x-1)*(x-1))-exp(-(*lambda)*(*lambda)*(x+1)*(x+1)));
	double res=0;

	__shared__ double left[256],right[256],localu[256];

	if(xid>0&&xid<gridDim.x){
		left[yid]=u[xid-1+yid*gridDim.x];
		right[yid]=u[xid+1+yid*gridDim.x];
	}
	else if(xid==0){
		right[yid]=u[xid+1+yid*gridDim.x];
	}
	else{
		left[yid]=u[xid-1+yid*gridDim.x];
	}
	localu[yid]=u[xid+yid*gridDim.x];

	__syncthreads();

	// calculate the residuals and update the solution

	int br_indicator = xid+yid;
	if(0==br_indicator%2){
		// left bound
		if(xid==0&&yid>0&&yid<blockDim.x-1){
			res = 1.0/4*(right[yid]-localu[yid]) + 1.0/4*(localu[yid+1]-2*localu[yid]+localu[yid-1])-(*dx)*(*dx)/4*source;
		}
		// right bound
		else if(xid==gridDim.x-1&&yid>0&&yid<blockDim.x-1){
			res = 1.0/4*(left[yid]-localu[yid]) + 1.0/4*(localu[yid+1]-2*localu[yid]+localu[yid-1])-(*dx)*(*dx)/4*source;
		}
		//internal
		else if(yid>0&&yid<blockDim.x-1){
			res = 1.0/4*(left[yid]-2*localu[yid]+right[yid]) + 1.0/4*(localu[yid+1]-2*localu[yid]+localu[yid-1])-(*dx)*(*dx)/4*source;
		}
		__syncthreads();

		// update
		r_u[xid+yid*gridDim.x]=res;
		u[xid+yid*gridDim.x]+=(*omega)*res;
	}
}
