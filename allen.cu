#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<cuda.h>
#include<cufft.h>
#include<complex.h>

#ifndef M_PI
#define M_PI 3.1415956535
#endif
/*
 *
 * int main():
 * 
 * solves the Allen-Cahn equation using a random initial condition, the random seed of which can be specified by users
 *
 * argc should be 7 or 8:
 * argv[1]: grid size (N)
 * argv[2]: flowing speed in x direction (vx)
 * argv[3]: flowing speed in y direction (vy)
 * argv[4]: diffustion parameters (b)
 * argv[5]: nonlinearity (W)
 * argv[6]: number of iterations (K)
 * argv[7]: (optional) random seed
 */

__global__ void diff(cufftDoubleComplex*f,cufftDoubleComplex*f_x,cufftDoubleComplex*f_y,cufftDoubleComplex*f_xx,cufftDoubleComplex*f_yy);
__global__ void RK_update(cufftDoubleComplex*u, cufftDoubleComplex*u_temp, cufftDoubleComplex*u_x,cufftDoubleComplex*u_y,cufftDoubleComplex*u_xx,cufftDoubleComplex*u_yy,double*coef,double*vx,double*vy,double*b,double*W);

int main(int argc, char* argv[]){
	
	// check input and initialize
	if(argc<7){printf("Too few arguments.\n"); return 1;}
	if(argc>8){printf("Too many arguments.\n"); return 1;}
	
	int N,K;
	double vx,vy,b,W;

	N = atoi(argv[1]);
	vx = atof(argv[2]);
	vy = atof(argv[3]);
	b = atof(argv[4]);
	W = atof(argv[5]);
	K = atoi(argv[6]);
	srand48((long int)time(NULL));

	if(N>128){printf("Try a smaller N.\n");return 1;}

	if(argc==8){
		long int seed = atoi(argv[7]);
		srand48(seed);
	}

	printf("N=%d\n",N);
	printf("vx=%f\n",vx);
	printf("vy=%f\n",vy);
	printf("b=%f\n",b);
	printf("W=%f\n",W);
	printf("K=%d\n",K);

	// initialize in the host
	
	cufftDoubleComplex *u;

	u = (cufftDoubleComplex*)malloc(N*N*sizeof(cufftDoubleComplex));

	for(int i = 0;i<N*N;i++){u[i].x=2*drand48()-1;}
	
	double T_final = 5.0;
	double dt = T_final/K;
	double coef1 = dt/4, coef2 = dt/3, coef3 = dt/2;
	

	// initilize in the device
	cufftDoubleComplex *dev_u, *dev_u_x, *dev_u_y, *dev_u_xx, *dev_u_yy;
	cufftDoubleComplex *dev_f, *dev_f_x, *dev_f_y, *dev_f_xx, *dev_f_yy;
	cufftDoubleComplex *dev_u_temp;

	cudaMalloc((void**)&dev_u,N*N*sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&dev_u_x,N*N*sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&dev_u_y,N*N*sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&dev_u_xx,N*N*sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&dev_u_yy,N*N*sizeof(cufftDoubleComplex));

	cudaMalloc((void**)&dev_f,N*N*sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&dev_f_x,N*N*sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&dev_f_y,N*N*sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&dev_f_xx,N*N*sizeof(cufftDoubleComplex));
	cudaMalloc((void**)&dev_f_yy,N*N*sizeof(cufftDoubleComplex));

	cudaMalloc((void**)&dev_u_temp,N*N*sizeof(cufftDoubleComplex));

	double *dev_coef1, *dev_coef2, *dev_coef3, *dev_coef4, *dev_vx, *dev_vy, *dev_b, *dev_W;

	cudaMalloc((void**)&dev_coef1,sizeof(double));
	cudaMalloc((void**)&dev_coef2,sizeof(double));
	cudaMalloc((void**)&dev_coef3,sizeof(double));
	cudaMalloc((void**)&dev_coef4,sizeof(double));
	cudaMalloc((void**)&dev_vx,sizeof(double));
	cudaMalloc((void**)&dev_vy,sizeof(double));
	cudaMalloc((void**)&dev_b,sizeof(double));
	cudaMalloc((void**)&dev_W,sizeof(double));

	// copy from the host to the device
	cudaMemcpy(dev_u,u,N*N*sizeof(cufftDoubleComplex),cudaMemcpyHostToDevice);
	
	cudaMemcpy(dev_coef1,&coef1,sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_coef2,&coef2,sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_coef3,&coef3,sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_coef4,&dt   ,sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vx, &vx, sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vy, &vy, sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, &b, sizeof(double),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_W, &W, sizeof(double),cudaMemcpyHostToDevice);


	// compute the solution in the device

	cufftHandle fplan, bplan;
	cufftPlan2d(&fplan, N, N, CUFFT_Z2Z);
	cufftPlan2d(&bplan, N, N, CUFFT_Z2Z);

	dim3 gridmesh(N), blockmesh(N);

	cudaMemcpy(dev_u_temp,dev_u,N*N*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToDevice);
	for(int i=0; i<K; i++){
		cufftExecZ2Z(fplan, dev_u_temp, dev_f, CUFFT_FORWARD);
		diff<<<gridmesh,blockmesh>>>(dev_f,dev_f_x,dev_f_y,dev_f_xx,dev_f_yy);
		cufftExecZ2Z(bplan, dev_f_x, dev_u_x, CUFFT_INVERSE);
		cufftExecZ2Z(bplan, dev_f_y, dev_u_y, CUFFT_INVERSE);
		cufftExecZ2Z(bplan, dev_f_xx, dev_u_xx, CUFFT_INVERSE);
		cufftExecZ2Z(bplan, dev_f_yy, dev_u_yy, CUFFT_INVERSE);
		RK_update<<<gridmesh,blockmesh>>>(dev_u,dev_u_temp,dev_u_x,dev_u_y,dev_u_xx,dev_u_yy,dev_coef1,dev_vx,dev_vy,dev_b,dev_W);
	
		cufftExecZ2Z(fplan, dev_u_temp, dev_f, CUFFT_FORWARD);
		diff<<<gridmesh,blockmesh>>>(dev_f,dev_f_x,dev_f_y,dev_f_xx,dev_f_yy);
		cufftExecZ2Z(bplan, dev_f_x, dev_u_x, CUFFT_INVERSE);
		cufftExecZ2Z(bplan, dev_f_y, dev_u_y, CUFFT_INVERSE);
		cufftExecZ2Z(bplan, dev_f_xx, dev_u_xx, CUFFT_INVERSE);
		cufftExecZ2Z(bplan, dev_f_yy, dev_u_yy, CUFFT_INVERSE);
		RK_update<<<gridmesh,blockmesh>>>(dev_u,dev_u_temp,dev_u_x,dev_u_y,dev_u_xx,dev_u_yy,dev_coef2,dev_vx,dev_vy,dev_b,dev_W);
		
		cufftExecZ2Z(fplan, dev_u_temp, dev_f, CUFFT_FORWARD);
		diff<<<gridmesh,blockmesh>>>(dev_f,dev_f_x,dev_f_y,dev_f_xx,dev_f_yy);
		cufftExecZ2Z(bplan, dev_f_x, dev_u_x, CUFFT_INVERSE);
		cufftExecZ2Z(bplan, dev_f_y, dev_u_y, CUFFT_INVERSE);
		cufftExecZ2Z(bplan, dev_f_xx, dev_u_xx, CUFFT_INVERSE);
		cufftExecZ2Z(bplan, dev_f_yy, dev_u_yy, CUFFT_INVERSE);
		RK_update<<<gridmesh,blockmesh>>>(dev_u,dev_u_temp,dev_u_x,dev_u_y,dev_u_xx,dev_u_yy,dev_coef3,dev_vx,dev_vy,dev_b,dev_W);
	
		cufftExecZ2Z(fplan, dev_u_temp, dev_f, CUFFT_FORWARD);
		diff<<<gridmesh,blockmesh>>>(dev_f,dev_f_x,dev_f_y,dev_f_xx,dev_f_yy);
		cufftExecZ2Z(bplan, dev_f_x, dev_u_x, CUFFT_INVERSE);
		cufftExecZ2Z(bplan, dev_f_y, dev_u_y, CUFFT_INVERSE);
		cufftExecZ2Z(bplan, dev_f_xx, dev_u_xx, CUFFT_INVERSE);
		cufftExecZ2Z(bplan, dev_f_yy, dev_u_yy, CUFFT_INVERSE);
		RK_update<<<gridmesh,blockmesh>>>(dev_u,dev_u_temp,dev_u_x,dev_u_y,dev_u_xx,dev_u_yy,dev_coef4,dev_vx,dev_vy,dev_b,dev_W);
	
		cudaMemcpy(dev_u, dev_u_temp,N*N*sizeof(cufftDoubleComplex), cudaMemcpyDeviceToDevice);
	}

	// save data
	
	cudaMemcpy(u,dev_u,N*N*sizeof(cufftDoubleComplex),cudaMemcpyDeviceToHost);
	
	double *soln = (double*)malloc(N*N*sizeof(double));

	for(int i=0;i<N*N;i++){*(soln+i)=u[i].x;}

	FILE *fpt=fopen("Allen.out","wb");
	fwrite(soln,sizeof(double),N*N,fpt);
	fclose(fpt);

	cudaFree(dev_u);
	cudaFree(dev_u_x);
	cudaFree(dev_u_y);
	cudaFree(dev_u_xx);
	cudaFree(dev_u_yy);
	cudaFree(dev_f);
	cudaFree(dev_f_x);
	cudaFree(dev_f_y);
	cudaFree(dev_f_xx);
	cudaFree(dev_f_yy);
	cudaFree(dev_u_temp);
	cudaFree(dev_coef1);
	cudaFree(dev_coef2);
	cudaFree(dev_coef3);
	cudaFree(dev_coef4);
	cudaFree(dev_vx);
	cudaFree(dev_vy);
	cudaFree(dev_b);
	cudaFree(dev_W);

	free(u);

	return 0;
}

__global__ void diff(cufftDoubleComplex*f,cufftDoubleComplex*f_x,cufftDoubleComplex*f_y,cufftDoubleComplex*f_xx,cufftDoubleComplex*f_yy){

	int xid = blockIdx.x, yid = threadIdx.x;
	int x_pos, y_pos;
	y_pos = xid<(gridDim.x+1.0)/2?xid:xid-gridDim.x;
	x_pos = yid<blockDim.x/2?-1-yid:blockDim.x-yid;

	__shared__ cufftDoubleComplex local_f[128],local_diff[128];

	local_f[yid] = f[xid+yid*gridDim.x];

	// calculate u_x
	local_diff[yid].x = -x_pos*local_f[yid].y/gridDim.x/gridDim.x;
	local_diff[yid].y = x_pos*local_f[yid].x/gridDim.x/gridDim.x;
	f_x[xid+yid*gridDim.x] = local_diff[yid];
	__syncthreads();

	// calculate u_y
	local_diff[yid].x = -y_pos*local_f[yid].y/gridDim.x/gridDim.x;
	local_diff[yid].y = y_pos*local_f[yid].x/gridDim.x/gridDim.x;
	f_y[xid+yid*gridDim.x] = local_diff[yid];
	__syncthreads();

	// calculate u_xx
	local_diff[yid].x = -x_pos*x_pos*local_f[yid].x/gridDim.x/gridDim.x;
	local_diff[yid].y = -x_pos*x_pos*local_f[yid].y/gridDim.x/gridDim.x; 
	f_xx[xid+yid*gridDim.x] = local_diff[yid];
	__syncthreads();

	// calculate u_yy
	local_diff[yid].x = -y_pos*y_pos*local_f[yid].x/gridDim.x/gridDim.x;
	local_diff[yid].y = -y_pos*y_pos*local_f[yid].y/gridDim.x/gridDim.x;
	f_yy[xid+yid*gridDim.x] = local_diff[yid];
	__syncthreads();
}

__global__ void RK_update(cufftDoubleComplex*u, cufftDoubleComplex*u_temp,cufftDoubleComplex*u_x,cufftDoubleComplex*u_y,cufftDoubleComplex*u_xx,cufftDoubleComplex*u_yy,double*coef,double*vx,double*vy,double*b,double*W){
	int xid = blockIdx.x, yid = threadIdx.x;
	__shared__ cufftDoubleComplex local_u[128];

	local_u[yid] = u_temp[xid+yid*gridDim.x];
	u_temp[xid+yid*gridDim.x].x=u[xid+yid*gridDim.x].x+*coef*(*b)*(local_u[yid].x*(1-local_u[yid].x*local_u[yid].x)+3*local_u[yid].x*local_u[yid].y*local_u[yid].y)/(*W)/(*W);
	u_temp[xid+yid*gridDim.x].y=u[xid+yid*gridDim.x].y+*coef*(*b)*(-3*local_u[yid].x*local_u[yid].x*local_u[yid].y+local_u[yid].y+local_u[yid].y*local_u[yid].y)/(*W)/(*W);

	local_u[yid] = u_x[xid+yid*gridDim.x];
	u_temp[xid+yid*gridDim.x].x-=*coef*local_u[yid].x*(*vx);
	u_temp[xid+yid*gridDim.x].y-=*coef*local_u[yid].y*(*vx);

	local_u[yid] = u_y[xid+yid*gridDim.x];
	u_temp[xid+yid*gridDim.x].x-=*coef*local_u[yid].x*(*vy);
	u_temp[xid+yid*gridDim.x].y-=*coef*local_u[yid].y*(*vy);

	local_u[yid] = u_xx[xid+yid*gridDim.x];
	u_temp[xid+yid*gridDim.x].x+=*coef*local_u[yid].x*(*b);
	u_temp[xid+yid*gridDim.x].y+=*coef*local_u[yid].y*(*b);

	local_u[yid] = u_yy[xid+yid*gridDim.x];
	u_temp[xid+yid*gridDim.x].x+=*coef*local_u[yid].x*(*b);
	u_temp[xid+yid*gridDim.x].y+=*coef*local_u[yid].y*(*b);
}
