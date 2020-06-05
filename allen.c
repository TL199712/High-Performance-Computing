//
//  allen.c
//  allen
//
//
//  Copyright © 2020 Tao. All rights reserved.
//



//
// This code can be improved by not having so many initialization in the function.

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include "fftw3-mpi.h"
#include "mpi.h"

ptrdiff_t N,M;
double vx,vy,b,W;

void derivatives_calc(fftw_complex*,fftw_complex*,fftw_complex*,fftw_complex*,fftw_complex*,ptrdiff_t*,ptrdiff_t*,int,ptrdiff_t,ptrdiff_t);

int main(int argc, char * argv[]) {
    //
    // Set-up
    //
    int my_rank, p; // rank of processors & total #processors
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Status status;
    fftw_mpi_init();
    //
    // check input arguments
    //
    if(argc<7){
        printf("Not Enough Parameters. Stop!\n");
        return 1;
    }
    if(argc>8){
        printf("Too Many Parameters. Stop!\n");
        return 1;
    }
    
    N = atoi(argv[1]);
    vx = atof(argv[2]); vy=atof(argv[3]);
    b = atof(argv[4]); W=atof(argv[5]);
    M=atoi(argv[6]);
    
    if(argc==8){
        srand48(atoi(argv[7])+my_rank);
        printf("Seed = %d.\n",atoi(argv[7]));
    }
    else{
        long int seed = time(NULL);
        srand48(seed+my_rank);
    }
    
    if(my_rank==0){
        printf("N = %td.\n",N);
        printf("vx = %f.\n",vx);
        printf("vy = %f.\n",vy);
        printf("b = %f.\n",b);
        printf("W = %f.\n",W);
        printf("M = %td.\n",M);
    }
    
    //
    // Initialization
    //
    
    double time_start=MPI_Wtime();
    
    ptrdiff_t alloc_local, local_n0, local_0_start;
    alloc_local = fftw_mpi_local_size_2d(N, N/2+1, MPI_COMM_WORLD, &local_n0, &local_0_start);
    ptrdiff_t x_span = local_n0, y_span = N;
    
    fftw_complex *num_soln,*temp,*deriv_x, *deriv_y, *deriv_xx, *deriv_yy;
    double* output,*output_all;
    
    // output: used in the last in MPI_WRITE
    // output_all: not used
    // num_soln: store the numerical solution at each time step
    // temp: help to record
    // deriv_*: store the derivatives
    
    output = (double*)malloc(x_span*y_span*sizeof(double));
    output_all = (double*)malloc(N*N*sizeof(double));
    num_soln = (fftw_complex*)malloc(x_span*y_span*sizeof(fftw_complex));
    temp = (fftw_complex*)malloc(x_span*y_span*sizeof(fftw_complex));
    deriv_x = (fftw_complex*)malloc(x_span*y_span*sizeof(fftw_complex));
    deriv_y = (fftw_complex*)malloc(x_span*y_span*sizeof(fftw_complex));
    deriv_xx = (fftw_complex*)malloc(x_span*y_span*sizeof(fftw_complex));
    deriv_yy = (fftw_complex*)malloc(x_span*y_span*sizeof(fftw_complex));
    
    // setting initials for num_soln
    
    for(int i=0;i<x_span;i++){
        for(int j=0;j<y_span;j++){
            *(num_soln+i*y_span+j)=2*drand48()-1;
        }
    }
    
    // store wave numbers in each position
    
    ptrdiff_t x_wave[N],y_wave[N];
    for(ptrdiff_t i=0;i<ceil((1.+N)/2);i++){
        x_wave[i]=i;
        y_wave[i]=i;
    }
    for(ptrdiff_t i=N-1;i>=ceil((1.+N)/2);i--){
        x_wave[i]=i-N;
        y_wave[i]=i-N;
    }
    
    double T_final=5.0;
    double dt = T_final/M;
    
    //
    // Iterating
    //
    for(int step=0; step<M; step++){
        
        for(int i=0;i<x_span*y_span;i++){
            *(temp+i)=*(num_soln+i);
        }
        
        // calculate derivatives using Fn and update num_soln to F1
        derivatives_calc(num_soln,deriv_x,deriv_y,deriv_xx,deriv_yy,x_wave,y_wave,my_rank, x_span, y_span);
        for(int i=0;i<x_span*y_span;i++){
            *(num_soln+i)=*(temp+i)+dt/4*(-vx*(*(deriv_x+i))-vy*(*(deriv_y+i))+b*((*(deriv_xx+i))+(*(deriv_yy+i))+(*(num_soln+i))*(1.0-(*(num_soln+i))*(*(num_soln+i)))/W/W));
        }
        MPI_Barrier(MPI_COMM_WORLD); //sync after 1st stage of RK4
        
        // calculate derivatives using F1 and update num_soln to F2
        derivatives_calc(num_soln,deriv_x,deriv_y,deriv_xx,deriv_yy,x_wave,y_wave,my_rank, x_span, y_span);
        for(int i=0;i<x_span*y_span;i++){
            *(num_soln+i)=*(temp+i)+dt/3*(-vx*(*(deriv_x+i))-vy*(*(deriv_y+i))+b*((*(deriv_xx+i))+(*(deriv_yy+i))+(*(num_soln+i))*(1.0-(*(num_soln+i))*(*(num_soln+i)))/W/W));
        }
        MPI_Barrier(MPI_COMM_WORLD); //sync after 2nd stage of RK4
        
        // calculate derivatives using F2 and update num_soln to F3
        derivatives_calc(num_soln,deriv_x,deriv_y,deriv_xx,deriv_yy,x_wave,y_wave,my_rank, x_span, y_span);
        for(int i=0;i<x_span*y_span;i++){
            *(num_soln+i)=*(temp+i)+dt/2*(-vx*(*(deriv_x+i))-vy*(*(deriv_y+i))+b*((*(deriv_xx+i))+(*(deriv_yy+i))+(*(num_soln+i))*(1.0-(*(num_soln+i))*(*(num_soln+i)))/W/W));
        }
        MPI_Barrier(MPI_COMM_WORLD); //sync after 3rd stage of RK4
        
        // calculate derivatives using F3 and update num_soln to F4(F_{n+1})
        derivatives_calc(num_soln,deriv_x,deriv_y,deriv_xx,deriv_yy,x_wave,y_wave,my_rank, x_span, y_span);
        for(int i=0;i<x_span*y_span;i++){
            *(num_soln+i)=*(temp+i)+dt*(-vx*(*(deriv_x+i))-vy*(*(deriv_y+i))+b*((*(deriv_xx+i))+(*(deriv_yy+i))+(*(num_soln+i))*(1.0-(*(num_soln+i))*(*(num_soln+i)))/W/W));
        }
        MPI_Barrier(MPI_COMM_WORLD); //sync after 4th stage of RK4
    }
    for(int i=0;i<x_span*y_span;i++){
        *(output+i)=creal(*(num_soln+i));
    }
    
    double time_used = MPI_Wtime()-time_start;
    printf("Processor %d execution time %f\n",my_rank,time_used);
    
    MPI_File fID;
    MPI_File_open(MPI_COMM_WORLD, "Allen.out", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fID);
    MPI_File_write_ordered(fID, output, x_span*y_span, MPI_DOUBLE, &status);
    MPI_File_close(&fID);
    
    // Set everything to NULL and release the memory
    
    fftw_free(num_soln);
    fftw_free(temp);
    fftw_free(deriv_x);
    fftw_free(deriv_y);
    fftw_free(deriv_xx);
    fftw_free(deriv_yy);
    free(output);
    
    num_soln = NULL;
    temp = NULL;
    deriv_x = NULL;
    deriv_y = NULL;
    deriv_xx = NULL;
    deriv_yy = NULL;
    output = NULL;
    
    MPI_Finalize();
    return 0;
}

void derivatives_calc(fftw_complex* num_soln,fftw_complex* deriv_x,fftw_complex*deriv_y,fftw_complex*deriv_xx,fftw_complex*deriv_yy,ptrdiff_t* x_wave,ptrdiff_t*y_wave,int my_rank, ptrdiff_t x_span,ptrdiff_t y_span){
    
    fftw_complex* frequency_space;
    frequency_space = (fftw_complex*)malloc(x_span*y_span*sizeof(fftw_complex));
    
    fftw_plan plan_f,plan_x,plan_y,plan_xx,plan_yy;
    plan_f=fftw_mpi_plan_dft_2d(N, N, num_soln, frequency_space, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
    plan_x=fftw_mpi_plan_dft_2d(N, N, frequency_space, deriv_x, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
    plan_y=fftw_mpi_plan_dft_2d(N, N, frequency_space, deriv_y, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
    plan_xx=fftw_mpi_plan_dft_2d(N, N, frequency_space, deriv_xx, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
    plan_yy=fftw_mpi_plan_dft_2d(N, N, frequency_space, deriv_yy, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);
    
    fftw_execute(plan_f);
    for(int i=0;i<x_span;i++){
        for(int j=0;j<y_span;j++){
            *(frequency_space+i*y_span+j) *= 1.0*I*x_wave[i+x_span*my_rank]/N/N;
        }
    }
    fftw_execute(plan_x);
    for(int i=0;i<x_span;i++){
        for(int j=0;j<y_span;j++){
            *(deriv_x+i*y_span+j) = creal(*(deriv_x+i*y_span+j));
        }
    }
    
    fftw_execute(plan_f);
    for(int i=0;i<x_span;i++){
        for(int j=0;j<y_span;j++){
            *(frequency_space+i*y_span+j) *= 1.0*I*y_wave[j]/N/N;
        }
    }
    fftw_execute(plan_y);
    for(int i=0;i<x_span;i++){
        for(int j=0;j<y_span;j++){
            *(deriv_y+i*y_span+j) = creal(*(deriv_y+i*y_span+j));
        }
    }
    
    fftw_execute(plan_f);
    for(int i=0;i<x_span;i++){
        for(int j=0;j<y_span;j++){
            *(frequency_space+i*y_span+j) *= -1.0*x_wave[i+x_span*my_rank]*x_wave[i+x_span*my_rank]/N/N;
        }
    }
    fftw_execute(plan_xx);
    for(int i=0;i<x_span;i++){
        for(int j=0;j<y_span;j++){
            *(deriv_xx+i*y_span+j) = creal(*(deriv_xx+i*y_span+j));
        }
    }

    
    fftw_execute(plan_f);
    for(int i=0;i<x_span;i++){
        for(int j=0;j<y_span;j++){
            *(frequency_space+i*y_span+j) *= -1.0*y_wave[j]*y_wave[j]/N/N;
        }
    }
    fftw_execute(plan_yy);
    for(int i=0;i<x_span;i++){
        for(int j=0;j<y_span;j++){
            *(deriv_yy+i*y_span+j) = creal(*(deriv_yy+i*y_span+j));
        }
    }
    
    free(frequency_space);
    frequency_space=NULL;
    fftw_destroy_plan(plan_f);
    fftw_destroy_plan(plan_x);
    fftw_destroy_plan(plan_y);
    fftw_destroy_plan(plan_xx);
    fftw_destroy_plan(plan_yy);
}
