//
//  SOR.c
//  SOR
//
//  Created by Tao on 5/27/20.
//  Copyright © 2020 Tao. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "mpi.h"

double lambda = 100;
double left = -2.0;
double right = 2.0;
double top = 1.0;
double bottom = -1.0;

int prod(int*,int);
double source(double,double);
double error_two_steps(double*,double*,int);
void int_mem_init(int*,int);
void double_mem_init(double*,int);
void send_bound(int,int,int,int,double*,double*,double*);

int main(int argc, char * argv[]) {
    //
    // Set-up
    //
    int my_rank, p; // rank of processors & total #processors
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Status status;
    
    //
    // check input arguments
    //
    double  omega, tol; // SOR strength and tolerence
    int N,K; // Grid parameter, max step
    if(argc>5){printf("Too many argumentsn\n"); return 1;}
    else if(argc<5){printf("Too few arguments.\n"); return 1;}
    else{
        N = atoi(argv[1]);
        omega = atof(argv[2]);
        tol = atof(argv[3]);
        K = atoi(argv[4]);
    }
    
    if(my_rank==0){
        printf("N:%d\n",N);
        printf("omega:%f\n",omega);
        printf("tol:%*.*f\n",1,15,tol);
        printf("K:%d\n",K);
    }
    //
    // Initialization
    //
    
    double time_start = MPI_Wtime();
    
    int x_span=2*N-1, y_span=N/p; // #grid in x and y direction
    double dx, dy; // spacing
    dx = (right-left)/(2*N-2);
    dy = (top-bottom)/(N-1);
    double y_coef, x_coef, source_coef;
    y_coef = dx*dx/(dx*dx+dy*dy)/2;
    x_coef = dy*dy/(dx*dx+dy*dy)/2;
    source_coef = dx*dx*dy*dy/(dx*dx+dy*dy)/2;
    
    double *num_soln,*updated_soln,*upper_data, *lower_data, *resid;
    num_soln = (double*)malloc(x_span*y_span*sizeof(double));
    updated_soln = (double*)malloc(x_span*y_span*sizeof(double));
    resid = (double*)malloc(x_span*y_span*sizeof(double));
    upper_data = (double*)malloc(x_span*sizeof(double));
    lower_data = (double*)malloc(x_span*sizeof(double));
    
    double_mem_init(num_soln, x_span*y_span);// Setting all initial values to 0.
    double_mem_init(updated_soln, x_span*y_span);
    double_mem_init(upper_data, x_span);
    double_mem_init(lower_data, x_span);
    double_mem_init(resid, x_span*y_span);
    
    int* stopping_signal;// used to record which processor is done actually only used in rank 0
    stopping_signal = (int*)malloc(p*sizeof(int));
    int_mem_init(stopping_signal, p);
    
    int flag = 0; // used to record if this processor satisfies the stopping critiria
    
    int start_x,start_y; // store indices of the header
    start_x = 0;
    start_y = my_rank*y_span;
    //
    // use SOR to solve the quation
    //
    int count = 0;
    while(count<K && prod(stopping_signal, p)==0){
        
        send_bound(p,my_rank,x_span,y_span,num_soln,lower_data,upper_data);
        
        // Calculation
        
        /* updating the red points */
        
        // rank 0 (first, bottom)
        if(my_rank==0){
            // updating the upper and lower boundary in this processor
            *updated_soln=0;
            *(updated_soln+x_span-1)=0;
            for(int i=1;i<x_span-1;i++){
                // lower bound
                if((start_x+i+start_y)%2==0){
                    *(updated_soln+i)=0;
                }
                // upper bound
                if((start_x+i+start_y+y_span-1)%2==0){
                    *(updated_soln+(y_span-1)*x_span+i)=(1-omega)*(*(num_soln+(y_span-1)*x_span+i)) \
                    + omega * y_coef *(*(num_soln+(y_span-2)*x_span+i)+*(upper_data+i)) \
                    + omega * x_coef *(*(num_soln+(y_span-1)*x_span+i-1)+*(num_soln+(y_span-1)*x_span+i+1))\
                    - omega * source_coef * source(left+i*dx, bottom+(y_span-1)*dy);
                    
                }
            }
            // updating the internal points
            for(int i=1;i<x_span-1;i++){
                for(int j=1;j<y_span-1;j++){
                    if((start_x+i+start_y+j)%2==0){
                        *(updated_soln+j*x_span+i)=(1-omega)*(*(num_soln+j*x_span+i)) \
                        + omega * y_coef *(*(num_soln+(j-1)*x_span+i)+*(num_soln+(j+1)*x_span+i))\
                        + omega * x_coef *(*(num_soln+j*x_span+i-1)+*(num_soln+j*x_span+i+1))\
                        - omega * source_coef * source(left+i*dx, bottom+my_rank*dy*y_span+j*dy);
                    }
                }
            }
            
            // updating two upper corners
            if((start_x+0+start_y+y_span-1)%2==0){
                *(updated_soln+(y_span-1)*x_span) = (1-omega+omega*x_coef)*(*(num_soln+(y_span-1)*x_span)) \
                + omega * y_coef *(*(num_soln+(y_span-2)*x_span)+*(upper_data))\
                + omega * x_coef *(*(num_soln+(y_span-1)*x_span+1))\
                - omega * source_coef * source(left, bottom+my_rank*dy*y_span+(y_span-1)*dy);
            }
            if((start_x+x_span-1+start_y+y_span-1)%2==0){
                *(updated_soln+(y_span-1)*x_span+x_span-1) = (1-omega+omega*x_coef)*(*(num_soln+(y_span-1)*x_span+x_span-1)) \
                + omega * y_coef *(*(num_soln+(y_span-2)*x_span+x_span-1)+*(upper_data+x_span-1))\
                + omega * x_coef *(*(num_soln+(y_span-1)*x_span+x_span-2))\
                - omega * source_coef * source(right, bottom+my_rank*dy*y_span+(y_span-1)*dy);
            }
            // updating the left and right boundaries
            for(int j=1;j<y_span-1;j++){
                if((start_x+0+start_y+j)%2==0){
                    *(updated_soln+j*x_span)=(1-omega+omega*x_coef)*(*(num_soln+j*x_span)) \
                    + omega * y_coef *(*(num_soln+(j-1)*x_span)+*(num_soln+(j+1)*x_span))\
                    + omega * x_coef *(*(num_soln+j*x_span+1))\
                    - omega * source_coef * source(left, bottom+my_rank*dy*y_span+j*dy);
                }
                if((start_x+x_span-1+start_y+j)%2==0){
                    *(updated_soln+j*x_span+x_span-1)=(1-omega+omega*x_coef)*(*(num_soln+j*x_span+x_span-1)) \
                    + omega * y_coef *(*(num_soln+(j-1)*x_span+x_span-1)+*(num_soln+(j+1)*x_span+x_span-1))\
                    + omega * x_coef *(*(num_soln+j*x_span+x_span-2))\
                    - omega * source_coef * source(right, bottom+my_rank*dy*y_span+j*dy);
                }
            }
        }
        // rank p-1 (last, top)
        else if(my_rank==p-1){
            // updating the upper and lower boundary in this processor
            *(updated_soln+(y_span-1)*x_span)=0;
            *(updated_soln+(y_span-1)*x_span+x_span-1)=0;
            for(int i=1;i<x_span-1;i++){
                if((start_x+i+start_y+y_span-1)%2==0){
                    *(updated_soln+(y_span-1)*x_span+i)=0;
                }
                if((start_x+i+start_y+0)%2==0){
                    *(updated_soln+i)=(1-omega)*(*(num_soln+i)) \
                    + omega * y_coef *(*(num_soln+x_span+i)+*(lower_data+i)) \
                    + omega * x_coef *(*(num_soln+i-1)+*(num_soln+i+1)) \
                    - omega * source_coef * source(left+i*dx, bottom+my_rank*dy*y_span);
                }
            }
            // updating the internal points
            for(int i=1;i<x_span-1;i++){
                for(int j=1;j<y_span-1;j++){
                    if((start_x+i+start_y+j)%2==0){
                        *(updated_soln+j*x_span+i)=(1-omega)*(*(num_soln+j*x_span+i)) \
                        + omega * y_coef *(*(num_soln+(j-1)*x_span+i)+*(num_soln+(j+1)*x_span+i))\
                        + omega * x_coef *(*(num_soln+j*x_span+i-1)+*(num_soln+j*x_span+i+1))\
                        - omega * source_coef * source(left+i*dx, bottom+my_rank*dy*y_span+j*dy);
                    }
                }
            }
            // updating two lower corners
            if((start_x+start_y)%2==0){
                *(updated_soln) = (1-omega+omega*x_coef)*(*(num_soln)) \
                + omega * y_coef *(*(num_soln+x_span)+*(lower_data))\
                + omega * x_coef *(*(num_soln+1))\
                - omega * source_coef * source(left, bottom+my_rank*dy*y_span);
            }
            if((start_x+x_span-1+start_y)%2==0){
                *(updated_soln+x_span-1) = (1-omega+omega*x_coef)*(*(num_soln+x_span-1)) \
                + omega * y_coef *(*(num_soln+x_span+x_span-1)+*(lower_data+x_span-1))\
                + omega * x_coef *(*(num_soln+x_span-2))\
                - omega * source_coef * source(right, bottom+my_rank*dy*y_span);
            }
            
            // updating the left and right boundaries
            for(int j=1;j<y_span-1;j++){
                if((start_x+0+start_y+j)%2==0){
                    *(updated_soln+j*x_span)=(1-omega+omega*x_coef)*(*(num_soln+j*x_span)) \
                    + omega * y_coef *(*(num_soln+(j-1)*x_span)+*(num_soln+(j+1)*x_span))\
                    + omega * x_coef *(*(num_soln+j*x_span+1))\
                    - omega * source_coef * source(left, bottom+my_rank*dy*y_span+j*dy);
                }
                if((start_x+x_span-1+start_y+j)%2==0){
                    *(updated_soln+j*x_span+x_span-1)=(1-omega+omega*x_coef)*(*(num_soln+j*x_span+x_span-1)) \
                    + omega * y_coef *(*(num_soln+(j-1)*x_span+x_span-1)+*(num_soln+(j+1)*x_span+x_span-1))\
                    + omega * x_coef *(*(num_soln+j*x_span+x_span-2))\
                    - omega * source_coef * source(right, bottom+my_rank*dy*y_span+j*dy);
                }
            }
        }
        else{
            // updating the upper and lower boundary in this processor
            for(int i=1;i<x_span-1;i++){
                if((start_x+i+start_y+y_span-1)%2==0){
                    *(updated_soln+(y_span-1)*x_span+i)=(1-omega)*(*(num_soln+(y_span-1)*x_span+i)) \
                    + omega * y_coef *(*(num_soln+(y_span-2)*x_span+i)+*(upper_data+i)) \
                    + omega * x_coef *(*(num_soln+(y_span-1)*x_span+i-1)+*(num_soln+(y_span-1)*x_span+i+1))\
                    - omega * source_coef * source(left+i*dx, bottom+my_rank*dy*y_span+(y_span-1)*dy);
                }
                if((start_x+i+start_y)%2==0){
                    *(updated_soln+i)=(1-omega)*(*(num_soln+i)) \
                    + omega * y_coef *(*(num_soln+x_span+i)+*(lower_data+i)) \
                    + omega * x_coef *(*(num_soln+i-1)+*(num_soln+i+1)) \
                    - omega * source_coef * source(left+i*dx, bottom+my_rank*dy*y_span);
                }
            }
            // updating the internal points
            for(int i=1;i<x_span-1;i++){
                for(int j=1;j<y_span-1;j++){
                    if((start_x+i+start_y+j)%2==0){
                        *(updated_soln+j*x_span+i)=(1-omega)*(*(num_soln+j*x_span+i)) \
                        + omega * y_coef *(*(num_soln+(j-1)*x_span+i)+*(num_soln+(j+1)*x_span+i))\
                        + omega * x_coef *(*(num_soln+j*x_span+i-1)+*(num_soln+j*x_span+i+1))\
                        - omega * source_coef * source(left+i*dx, bottom+my_rank*dy*y_span+j*dy);
                    }
                }
            }
            // updating two lower corners
            if((start_x+start_y)%2==0){
                *(updated_soln) = (1-omega+omega*x_coef)*(*(num_soln)) \
                + omega * y_coef *(*(num_soln+x_span)+*(lower_data))\
                + omega * x_coef *(*(num_soln+1))\
                - omega * source_coef * source(left, bottom+my_rank*dy*y_span);
            }
            if((start_x+x_span-1+start_y)%2==0){
                *(updated_soln+x_span-1) = (1-omega+omega*x_coef)*(*(num_soln+x_span-1)) \
                + omega * y_coef *(*(num_soln+x_span+x_span-1)+*(lower_data+x_span-1))\
                + omega * x_coef *(*(num_soln+x_span-2))\
                - omega * source_coef * source(right, bottom+my_rank*dy*y_span);
            }
            // updating two upper corners
            if((start_x+start_y+y_span-1)%2==0){
                *(updated_soln+(y_span-1)*x_span) = (1-omega+omega*x_coef)*(*(num_soln+(y_span-1)*x_span)) \
                + omega * y_coef *(*(num_soln+(y_span-2)*x_span)+*(upper_data))\
                + omega * x_coef *(*(num_soln+(y_span-1)*x_span+1))\
                - omega * source_coef * source(left, bottom+my_rank*dy*y_span+(y_span-1)*dy);
            }
            if((start_x+x_span-1+start_y+y_span-1)%2==0){
                *(updated_soln+(y_span-1)*x_span+x_span-1) = (1-omega+omega*x_coef)*(*(num_soln+(y_span-1)*x_span+x_span-1)) \
                + omega * y_coef *(*(num_soln+(y_span-2)*x_span+x_span-1)+*(upper_data+x_span-1))\
                + omega * x_coef *(*(num_soln+(y_span-1)*x_span+x_span-2))\
                - omega * source_coef * source(right, bottom+my_rank*dy*y_span+(y_span-1)*dy);
            }
            // updating the left and right boundaries
            for(int j=1;j<y_span-1;j++){
                if((start_x+0+start_y+j)%2==0){
                    *(updated_soln+j*x_span)=(1-omega+omega*x_coef)*(*(num_soln+j*x_span)) \
                    + omega * y_coef *(*(num_soln+(j-1)*x_span)+*(num_soln+(j+1)*x_span))\
                    + omega * x_coef *(*(num_soln+j*x_span+1))\
                    - omega * source_coef * source(left, bottom+my_rank*dy*y_span+j*dy);
                }
                if((start_x+x_span-1+start_y+j)%2==0){
                    *(updated_soln+j*x_span+x_span-1)=(1-omega+omega*x_coef)*(*(num_soln+j*x_span+x_span-1)) \
                    + omega * y_coef *(*(num_soln+(j-1)*x_span+x_span-1)+*(num_soln+(j+1)*x_span+x_span-1))\
                    + omega * x_coef *(*(num_soln+j*x_span+x_span-2))\
                    - omega * source_coef * source(right, bottom+my_rank*dy*y_span+j*dy);
                }
            }
        }
        for(int i=0;i<x_span*y_span;i++){
            *(num_soln+i) = *(updated_soln+i);
        }
        
        /* updating the black points */
        
        send_bound(p,my_rank, x_span, y_span,num_soln,lower_data,upper_data);
        
        // rank 0 (first, bottom)
        if(my_rank==0){
            // updating the upper and lower boundary in this processor
            *updated_soln=0;
            *(updated_soln+x_span-1)=0;
            for(int i=1;i<x_span-1;i++){
                // lower bound
                if((start_x+i+start_y)%2==1){
                    *(updated_soln+i)=0;
                }
                // upper bound
                if((start_x+i+start_y+y_span-1)%2==1){
                    *(updated_soln+(y_span-1)*x_span+i)=(1-omega)*(*(num_soln+(y_span-1)*x_span+i)) \
                    + omega * y_coef *(*(num_soln+(y_span-2)*x_span+i)+*(upper_data+i)) \
                    + omega * x_coef *(*(num_soln+(y_span-1)*x_span+i-1)+*(num_soln+(y_span-1)*x_span+i+1))\
                    - omega * source_coef * source(left+i*dx, bottom+(y_span-1)*dy);
                    
                }
            }
            // updating the internal points
            for(int i=1;i<x_span-1;i++){
                for(int j=1;j<y_span-1;j++){
                    if((start_x+i+start_y+j)%2==1){
                        *(updated_soln+j*x_span+i)=(1-omega)*(*(num_soln+j*x_span+i)) \
                        + omega * y_coef *(*(num_soln+(j-1)*x_span+i)+*(num_soln+(j+1)*x_span+i))\
                        + omega * x_coef *(*(num_soln+j*x_span+i-1)+*(num_soln+j*x_span+i+1))\
                        - omega * source_coef * source(left+i*dx, bottom+my_rank*dy*y_span+j*dy);
                    }
                }
            }
            
            // updating two upper corners
            if((start_x+0+start_y+y_span-1)%2==1){
                *(updated_soln+(y_span-1)*x_span) = (1-omega+omega*x_coef)*(*(num_soln+(y_span-1)*x_span)) \
                + omega * y_coef *(*(num_soln+(y_span-2)*x_span)+*(upper_data))\
                + omega * x_coef *(*(num_soln+(y_span-1)*x_span+1))\
                - omega * source_coef * source(left, bottom+my_rank*dy*y_span+(y_span-1)*dy);
            }
            if((start_x+x_span-1+start_y+y_span-1)%2==1){
                *(updated_soln+(y_span-1)*x_span+x_span-1) = (1-omega+omega*x_coef)*(*(num_soln+(y_span-1)*x_span+x_span-1)) \
                + omega * y_coef *(*(num_soln+(y_span-2)*x_span+x_span-1)+*(upper_data+x_span-1))\
                + omega * x_coef *(*(num_soln+(y_span-1)*x_span+x_span-2))\
                - omega * source_coef * source(right, bottom+my_rank*dy*y_span+(y_span-1)*dy);
            }
            // updating the left and right boundaries
            for(int j=1;j<y_span-1;j++){
                if((start_x+0+start_y+j)%2==1){
                    *(updated_soln+j*x_span)=(1-omega+omega*x_coef)*(*(num_soln+j*x_span)) \
                    + omega * y_coef *(*(num_soln+(j-1)*x_span)+*(num_soln+(j+1)*x_span))\
                    + omega * x_coef *(*(num_soln+j*x_span+1))\
                    - omega * source_coef * source(left, bottom+my_rank*dy*y_span+j*dy);
                }
                if((start_x+x_span-1+start_y+j)%2==1){
                    *(updated_soln+j*x_span+x_span-1)=(1-omega+omega*x_coef)*(*(num_soln+j*x_span+x_span-1)) \
                    + omega * y_coef *(*(num_soln+(j-1)*x_span+x_span-1)+*(num_soln+(j+1)*x_span+x_span-1))\
                    + omega * x_coef *(*(num_soln+j*x_span+x_span-2))\
                    - omega * source_coef * source(right, bottom+my_rank*dy*y_span+j*dy);
                }
            }
        }
        // rank p-1 (last, top)
        else if(my_rank==p-1){
            // updating the upper and lower boundary in this processor
            *(updated_soln+(y_span-1)*x_span)=0;
            *(updated_soln+(y_span-1)*x_span+x_span-1)=0;
            for(int i=1;i<x_span-1;i++){
                if((start_x+i+start_y+y_span-1)%2==1){
                    *(updated_soln+(y_span-1)*x_span+i)=0;
                }
                if((start_x+i+start_y+0)%2==1){
                    *(updated_soln+i)=(1-omega)*(*(num_soln+i)) \
                    + omega * y_coef *(*(num_soln+x_span+i)+*(lower_data+i)) \
                    + omega * x_coef *(*(num_soln+i-1)+*(num_soln+i+1)) \
                    - omega * source_coef * source(left+i*dx, bottom+my_rank*dy*y_span);
                }
            }
            // updating the internal points
            for(int i=1;i<x_span-1;i++){
                for(int j=1;j<y_span-1;j++){
                    if((start_x+i+start_y+j)%2==1){
                        *(updated_soln+j*x_span+i)=(1-omega)*(*(num_soln+j*x_span+i)) \
                        + omega * y_coef *(*(num_soln+(j-1)*x_span+i)+*(num_soln+(j+1)*x_span+i))\
                        + omega * x_coef *(*(num_soln+j*x_span+i-1)+*(num_soln+j*x_span+i+1))\
                        - omega * source_coef * source(left+i*dx, bottom+my_rank*dy*y_span+j*dy);
                    }
                }
            }
            // updating two lower corners
            if((start_x+start_y)%2==1){
                *(updated_soln) = (1-omega+omega*x_coef)*(*(num_soln)) \
                + omega * y_coef *(*(num_soln+x_span)+*(lower_data))\
                + omega * x_coef *(*(num_soln+1))\
                - omega * source_coef * source(left, bottom+my_rank*dy*y_span);
            }
            if((start_x+x_span-1+start_y)%2==1){
                *(updated_soln+x_span-1) = (1-omega+omega*x_coef)*(*(num_soln+x_span-1)) \
                + omega * y_coef *(*(num_soln+x_span+x_span-1)+*(lower_data+x_span-1))\
                + omega * x_coef *(*(num_soln+x_span-2))\
                - omega * source_coef * source(right, bottom+my_rank*dy*y_span);
            }
            
            // updating the left and right boundaries
            for(int j=1;j<y_span-1;j++){
                if((start_x+0+start_y+j)%2==1){
                    *(updated_soln+j*x_span)=(1-omega+omega*x_coef)*(*(num_soln+j*x_span)) \
                    + omega * y_coef *(*(num_soln+(j-1)*x_span)+*(num_soln+(j+1)*x_span))\
                    + omega * x_coef *(*(num_soln+j*x_span+1))\
                    - omega * source_coef * source(left, bottom+my_rank*dy*y_span+j*dy);
                }
                if((start_x+x_span-1+start_y+j)%2==1){
                    *(updated_soln+j*x_span+x_span-1)=(1-omega+omega*x_coef)*(*(num_soln+j*x_span+x_span-1)) \
                    + omega * y_coef *(*(num_soln+(j-1)*x_span+x_span-1)+*(num_soln+(j+1)*x_span+x_span-1))\
                    + omega * x_coef *(*(num_soln+j*x_span+x_span-2))\
                    - omega * source_coef * source(right, bottom+my_rank*dy*y_span+j*dy);
                }
            }
        }
        else{
            // updating the upper and lower boundary in this processor
            for(int i=1;i<x_span-1;i++){
                if((start_x+i+start_y+y_span-1)%2==1){
                    *(updated_soln+(y_span-1)*x_span+i)=(1-omega)*(*(num_soln+(y_span-1)*x_span+i)) \
                    + omega * y_coef *(*(num_soln+(y_span-2)*x_span+i)+*(upper_data+i)) \
                    + omega * x_coef *(*(num_soln+(y_span-1)*x_span+i-1)+*(num_soln+(y_span-1)*x_span+i+1))\
                    - omega * source_coef * source(left+i*dx, bottom+my_rank*dy*y_span+(y_span-1)*dy);
                }
                if((start_x+i+start_y)%2==1){
                    *(updated_soln+i)=(1-omega)*(*(num_soln+i)) \
                    + omega * y_coef *(*(num_soln+x_span+i)+*(lower_data+i)) \
                    + omega * x_coef *(*(num_soln+i-1)+*(num_soln+i+1)) \
                    - omega * source_coef * source(left+i*dx, bottom+my_rank*dy*y_span);
                }
            }
            // updating the internal points
            for(int i=1;i<x_span-1;i++){
                for(int j=1;j<y_span-1;j++){
                    if((start_x+i+start_y+j)%2==1){
                        *(updated_soln+j*x_span+i)=(1-omega)*(*(num_soln+j*x_span+i)) \
                        + omega * y_coef *(*(num_soln+(j-1)*x_span+i)+*(num_soln+(j+1)*x_span+i))\
                        + omega * x_coef *(*(num_soln+j*x_span+i-1)+*(num_soln+j*x_span+i+1))\
                        - omega * source_coef * source(left+i*dx, bottom+my_rank*dy*y_span+j*dy);
                    }
                }
            }
            // updating two lower corners
            if((start_x+start_y)%2==1){
                *(updated_soln) = (1-omega+omega*x_coef)*(*(num_soln)) \
                + omega * y_coef *(*(num_soln+x_span)+*(lower_data))\
                + omega * x_coef *(*(num_soln+1))\
                - omega * source_coef * source(left, bottom+my_rank*dy*y_span);
            }
            if((start_x+x_span-1+start_y)%2==1){
                *(updated_soln+x_span-1) = (1-omega+omega*x_coef)*(*(num_soln+x_span-1)) \
                + omega * y_coef *(*(num_soln+x_span+x_span-1)+*(lower_data+x_span-1))\
                + omega * x_coef *(*(num_soln+x_span-2))\
                - omega * source_coef * source(right, bottom+my_rank*dy*y_span);
            }
            // updating two upper corners
            if((start_x+start_y+y_span-1)%2==1){
                *(updated_soln+(y_span-1)*x_span) = (1-omega+omega*x_coef)*(*(num_soln+(y_span-1)*x_span)) \
                + omega * y_coef *(*(num_soln+(y_span-2)*x_span)+*(upper_data))\
                + omega * x_coef *(*(num_soln+(y_span-1)*x_span+1))\
                - omega * source_coef * source(left, bottom+my_rank*dy*y_span+(y_span-1)*dy);
            }
            if((start_x+x_span-1+start_y+y_span-1)%2==1){
                *(updated_soln+(y_span-1)*x_span+x_span-1) = (1-omega+omega*x_coef)*(*(num_soln+(y_span-1)*x_span+x_span-1)) \
                + omega * y_coef *(*(num_soln+(y_span-2)*x_span+x_span-1)+*(upper_data+x_span-1))\
                + omega * x_coef *(*(num_soln+(y_span-1)*x_span+x_span-2))\
                - omega * source_coef * source(right, bottom+my_rank*dy*y_span+(y_span-1)*dy);
            }
            // updating the left and right boundaries
            for(int j=1;j<y_span-1;j++){
                if((start_x+0+start_y+j)%2==1){
                    *(updated_soln+j*x_span)=(1-omega+omega*x_coef)*(*(num_soln+j*x_span)) \
                    + omega * y_coef *(*(num_soln+(j-1)*x_span)+*(num_soln+(j+1)*x_span))\
                    + omega * x_coef *(*(num_soln+j*x_span+1))\
                    - omega * source_coef * source(left, bottom+my_rank*dy*y_span+j*dy);
                }
                if((start_x+x_span-1+start_y+j)%2==1){
                    *(updated_soln+j*x_span+x_span-1)=(1-omega+omega*x_coef)*(*(num_soln+j*x_span+x_span-1)) \
                    + omega * y_coef *(*(num_soln+(j-1)*x_span+x_span-1)+*(num_soln+(j+1)*x_span+x_span-1))\
                    + omega * x_coef *(*(num_soln+j*x_span+x_span-2))\
                    - omega * source_coef * source(right, bottom+my_rank*dy*y_span+j*dy);
                }
            }
        }
        
        MPI_Barrier(MPI_COMM_WORLD);// sync after calculation
        
        // calculating the error and tell all processors if it is OK to stop
        if(error_two_steps(num_soln, updated_soln, x_span*y_span)<tol){
            flag=1;
        }
        else{
            flag=0;
        }
        MPI_Barrier(MPI_COMM_WORLD); // sync after validation
        
        MPI_Allgather(&flag, 1, MPI_INT, stopping_signal, 1, MPI_INT, MPI_COMM_WORLD);
        for(int i=0;i<x_span*y_span;i++){
            *(num_soln+i) = *(updated_soln+i);
        }
        
        count += 1;
    }
    
    double time_used = MPI_Wtime()-time_start;
    
    printf("Processor %d execution time %f\n:",my_rank,time_used);
    
    if(my_rank==0){
        printf("%d steps to satisify TOL<%1.15f.\n",count,tol);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // writing data
    MPI_File fID;
    MPI_File_open(MPI_COMM_WORLD, "Sources.out", MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &fID);
    MPI_File_write_ordered(fID, num_soln, x_span*y_span, MPI_DOUBLE, &status);
    MPI_File_close(&fID);
    
    // Set everything to NULL and release the memory
    num_soln = NULL;
    updated_soln = NULL;
    lower_data = NULL;
    upper_data = NULL;
    stopping_signal = NULL;
    free(num_soln);
    free(updated_soln);
    free(lower_data);
    free(upper_data);
    free(stopping_signal);
    
    MPI_Finalize();
    return 0;
}

int prod(int*flag, int n){
    // find the product of first n terms in array *flag
    // flag: an array (should only contain 0 and 1 in this project)
    // n: size of flag
    int temp = 1;
    for(int i=0;i<n;i++){
        temp *= *(flag+i);
    }
    return temp;
}

double source(double x,double y){
    // the source function
    // x,y: corresponding coordinates
    return 10*lambda/sqrt(M_PI)*exp(-lambda*lambda*y*y)*(exp(-lambda*lambda*(x-1)*(x-1))-exp(-lambda*lambda*(x+1)*(x+1)));
}

double error_two_steps(double*a,double*b,int n){
    // find the inf norm of a-b
    // a,b: two array of the same size n
    // n: size of a and b
    double temp=0, diff;
    for(int i=0;i<n;i++){
        diff = fabs(*(a+i)-*(b+i));
        if(diff>temp){temp=diff;}
    }
    return temp;
}

void int_mem_init(int*a,int n){
    for(int i=0;i<n;i++){
        *(a+i)=0;
    }
}

void double_mem_init(double*a,int n){
    for(int i=0;i<n;i++){
        *(a+i)=0.0;
    }
}

void send_bound(int p,int my_rank,int x_span,int y_span,double*num_soln,double*lower_data,double*upper_data){
    // sending boundaries upward (safe, no deadlock)
    MPI_Status status;
    if(my_rank==0){
        MPI_Send(num_soln+(y_span-1)*x_span, x_span, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    }
    else if(my_rank==p-1){
        MPI_Recv(lower_data, x_span, MPI_DOUBLE, p-2, p-2, MPI_COMM_WORLD, &status);
    }
    else{
        MPI_Recv(lower_data, x_span, MPI_DOUBLE, my_rank-1, my_rank-1, MPI_COMM_WORLD, &status);
        MPI_Send(num_soln+(y_span-1)*x_span, x_span, MPI_DOUBLE, my_rank+1, my_rank, MPI_COMM_WORLD);
    }
    
    //seding boundaries downward (safe, no deadlock)
    if(my_rank==p-1){
        MPI_Send(num_soln, x_span, MPI_DOUBLE, p-2, p-1, MPI_COMM_WORLD);
    }
    else if(my_rank==0){
        MPI_Recv(upper_data, x_span, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &status);
    }
    else{
        MPI_Recv(upper_data, x_span, MPI_DOUBLE, my_rank+1, my_rank+1, MPI_COMM_WORLD, &status);
        MPI_Send(num_soln, x_span, MPI_DOUBLE, my_rank-1, my_rank, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD); // sync after exchanging data(not necessary because we used blocked Send/Recv)
}
