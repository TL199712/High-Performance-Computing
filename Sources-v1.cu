#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<cuda.h>
#include<cuda_runtime.h>

/*
	version note:
	1. This version only uses the blocks, which is not so efficient.
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
__global__ void SOR_update(double*u,double*r_u,double*omega,double*coef,double*dx,double*lambda);
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
	dim3 meshDim(2*N-1,N);

	while(err>tol && iters<K){
		SOR_update_r<<<meshDim,1>>>(dev_u,dev_r_u,dev_omega,dev_coef,dev_dx,dev_lambda);
		SOR_update_b<<<meshDim,1>>>(dev_u,dev_r_u,dev_omega,dev_coef,dev_dx,dev_lambda);
		//SOR_update<<<meshDim,1>>>(dev_u,dev_r_u,dev_omega,dev_coef,dev_dx,dev_lambda);
		iters++;
		cudaMemcpy(r_u,dev_r_u,(2*N-1)*N*sizeof(double),cudaMemcpyDeviceToHost);
		err = get_max(r_u,(2*N-1)*N);
	}
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

__global__ void SOR_update(double*u,double*r_u,double*omega,double*coef,double*dx,double*lambda){
	int xid=blockIdx.x, yid=blockIdx.y;
	int br_indicator = xid+yid;
	double x = -2.0+xid*(*dx), y=-1.0+yid*(*dx);

	double source = (*coef)*exp(-(*lambda)*(*lambda)*y*y)*(exp(-(*lambda)*(*lambda)*(x-1)*(x-1))-exp(-(*lambda)*(*lambda)*(x+1)*(x+1)));

	// compute black points
	if(br_indicator % 2 == 0){
		// upper and lower bounds
		if(yid == 0 || yid == gridDim.y-1){
			r_u[xid+yid*gridDim.x] = 0;
			u[xid+yid*gridDim.x]+=(*omega)*r_u[xid+yid*gridDim.x];
		}
		// left bound
		else if(xid == 0&& yid>0 && yid<gridDim.y-1){
			r_u[xid+yid*gridDim.x]=1.0/4 * (u[xid+1+yid*gridDim.x]-u[xid+yid*gridDim.x]) + 1.0/4 * (u[xid+(yid+1)*gridDim.x]-2*u[xid+yid*gridDim.x]+u[xid+(yid-1)*gridDim.x]) - (*dx)*(*dx)/4*source;
			u[xid+yid*gridDim.x]+=(*omega)*r_u[xid+yid*gridDim.x];
		}
		else if(xid == gridDim.x-1 && yid>0 && yid<gridDim.y-1){
			r_u[xid+yid*gridDim.x]=1.0/4 * (u[xid-1+yid*gridDim.x]-u[xid+yid*gridDim.x]) + 1.0/4 * (u[xid+(yid+1)*gridDim.x]-2*u[xid+yid*gridDim.x]+u[xid+(yid-1)*gridDim.x]) - (*dx)*(*dx)/4*source;
			u[xid+yid*gridDim.x]+=(*omega)*r_u[xid+yid*gridDim.x];
		}
		else if(xid>0 && xid<gridDim.x-1 && yid>0 && yid<gridDim.y-1){
			r_u[xid+yid*gridDim.x]=1.0/4 * (u[xid-1+yid*gridDim.x]-2*u[xid+yid*gridDim.x]+u[xid+1+yid*gridDim.x]) + 1.0/4 * (u[xid+(yid+1)*gridDim.x]-2*u[xid+yid*gridDim.x]+u[xid+(yid-1)*gridDim.x]) - (*dx)*(*dx)/4*source;
			u[xid+yid*gridDim.x]+=(*omega)*r_u[xid+yid*gridDim.x];
		}
	
	}
	
	__syncthreads();

	// compute black points
	if(br_indicator % 2 == 1){
		// upper and lower bounds
		if(yid == 0 || yid == gridDim.y-1){
			r_u[xid+yid*gridDim.x] = 0;
			u[xid+yid*gridDim.x]+=(*omega)*r_u[xid+yid*gridDim.x];
		}
		// left	 bound
		else if(xid == 0&& yid>0 && yid<gridDim.y-1){
			r_u[xid+yid*gridDim.x]=1.0/4 * (u[xid+1+yid*gridDim.x]-u[xid+yid*gridDim.x]) + 1.0/4 * (u[xid+(yid+1)*gridDim.x]-2*u[xid+yid*gridDim.x]+u[xid+(yid-1)*gridDim.x]) - (*dx)*(*dx)/4*source;
			u[xid+yid*gridDim.x]+=(*omega)*r_u[xid+yid*gridDim.x];
		}
		else if(xid == gridDim.x-1 && yid>0 && yid<gridDim.y-1){
			r_u[xid+yid*gridDim.x]=1.0/4 * (u[xid-1+yid*gridDim.x]-u[xid+yid*gridDim.x]) + 1.0/4 * (u[xid+(yid+1)*gridDim.x]-2*u[xid+yid*gridDim.x]+u[xid+(yid-1)*gridDim.x]) - (*dx)*(*dx)/4*source;
			u[xid+yid*gridDim.x]+=(*omega)*r_u[xid+yid*gridDim.x];
		}
		else if(xid>0 && xid<gridDim.x-1 && yid>0 && yid<gridDim.y-1){
			r_u[xid+yid*gridDim.x]=1.0/4 * (u[xid-1+yid*gridDim.x]-2*u[xid+yid*gridDim.x]+u[xid+1+yid*gridDim.x]) + 1.0/4 * (u[xid+(yid+1)*gridDim.x]-2*u[xid+yid*gridDim.x]+u[xid+(yid-1)*gridDim.x]) - (*dx)*(*dx)/4*source;
			u[xid+yid*gridDim.x]+=(*omega)*r_u[xid+yid*gridDim.x];
		}
	}
}

__global__ void SOR_update_r(double*u,double*r_u,double*omega,double*coef,double*dx,double*lambda){
	int xid=blockIdx.x, yid=blockIdx.y;
	double x = -2.0+xid*(*dx), y=-1.0+yid*(*dx);	
	double source = (*coef)*exp(-(*lambda)*(*lambda)*y*y)*(exp(-(*lambda)*(*lambda)*(x-1)*(x-1))-exp(-(*lambda)*(*lambda)*(x+1)*(x+1)));


	int br_indicator = xid+yid;
	if(br_indicator%2==1){
	// upper and lower bounds
		if(yid == 0 || yid == gridDim.y-1){
			r_u[xid+yid*gridDim.x] = 0;
			u[xid+yid*gridDim.x]+=(*omega)*r_u[xid+yid*gridDim.x];
		}
		// left	 bound
		else if(xid == 0&& yid>0 && yid<gridDim.y-1){
			r_u[xid+yid*gridDim.x]=1.0/4 * (u[xid+1+yid*gridDim.x]-u[xid+yid*gridDim.x]) + 1.0/4 * (u[xid+(yid+1)*gridDim.x]-2*u[xid+yid*gridDim.x]+u[xid+(yid-1)*gridDim.x]) - (*dx)*(*dx)/4*source;
			u[xid+yid*gridDim.x]+=(*omega)*r_u[xid+yid*gridDim.x];
		}
		else if(xid == gridDim.x-1 && yid>0 && yid<gridDim.y-1){
			r_u[xid+yid*gridDim.x]=1.0/4 * (u[xid-1+yid*gridDim.x]-u[xid+yid*gridDim.x]) + 1.0/4 * (u[xid+(yid+1)*gridDim.x]-2*u[xid+yid*gridDim.x]+u[xid+(yid-1)*gridDim.x]) - (*dx)*(*dx)/4*source;
			u[xid+yid*gridDim.x]+=(*omega)*r_u[xid+yid*gridDim.x];
		}
		else if(xid>0 && xid<gridDim.x-1 && yid>0 && yid<gridDim.y-1){
			r_u[xid+yid*gridDim.x]=1.0/4 * (u[xid-1+yid*gridDim.x]-2*u[xid+yid*gridDim.x]+u[xid+1+yid*gridDim.x]) + 1.0/4 * (u[xid+(yid+1)*gridDim.x]-2*u[xid+yid*gridDim.x]+u[xid+(yid-1)*gridDim.x]) - (*dx)*(*dx)/4*source;
			u[xid+yid*gridDim.x]+=(*omega)*r_u[xid+yid*gridDim.x];
		}
	}

}
__global__ void SOR_update_b(double*u,double*r_u,double*omega,double*coef,double*dx,double*lambda){
	int xid=blockIdx.x, yid=blockIdx.y;
	double x = -2.0+xid*(*dx), y=-1.0+yid*(*dx);
	double source = (*coef)*exp(-(*lambda)*(*lambda)*y*y)*(exp(-(*lambda)*(*lambda)*(x-1)*(x-1))-exp(-(*lambda)*(*lambda)*(x+1)*(x+1)));


	int br_indicator = xid+yid;
	if(br_indicator%2==0){
	// upper and lower bounds
		if(yid == 0 || yid == gridDim.y-1){
			r_u[xid+yid*gridDim.x] = 0;
			u[xid+yid*gridDim.x]+=(*omega)*r_u[xid+yid*gridDim.x];
		}
		// left	 bound
		else if(xid == 0&& yid>0 && yid<gridDim.y-1){
			r_u[xid+yid*gridDim.x]=1.0/4 * (u[xid+1+yid*gridDim.x]-u[xid+yid*gridDim.x]) + 1.0/4 * (u[xid+(yid+1)*gridDim.x]-2*u[xid+yid*gridDim.x]+u[xid+(yid-1)*gridDim.x]) - (*dx)*(*dx)/4*source;
			u[xid+yid*gridDim.x]+=(*omega)*r_u[xid+yid*gridDim.x];
		}
		else if(xid == gridDim.x-1 && yid>0 && yid<gridDim.y-1){
			r_u[xid+yid*gridDim.x]=1.0/4 * (u[xid-1+yid*gridDim.x]-u[xid+yid*gridDim.x]) + 1.0/4 * (u[xid+(yid+1)*gridDim.x]-2*u[xid+yid*gridDim.x]+u[xid+(yid-1)*gridDim.x]) - (*dx)*(*dx)/4*source;
			u[xid+yid*gridDim.x]+=(*omega)*r_u[xid+yid*gridDim.x];
		}
		else if(xid>0 && xid<gridDim.x-1 && yid>0 && yid<gridDim.y-1){
			r_u[xid+yid*gridDim.x]=1.0/4 * (u[xid-1+yid*gridDim.x]-2*u[xid+yid*gridDim.x]+u[xid+1+yid*gridDim.x]) + 1.0/4 * (u[xid+(yid+1)*gridDim.x]-2*u[xid+yid*gridDim.x]+u[xid+(yid-1)*gridDim.x]) - (*dx)*(*dx)/4*source;
			u[xid+yid*gridDim.x]+=(*omega)*r_u[xid+yid*gridDim.x];
		}
	}

}
