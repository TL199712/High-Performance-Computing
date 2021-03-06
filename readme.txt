Sources-v1/Sources-v2.cu:

Use CUDA to calculate the solution of diffustion with sinks/sources

u_xx+u_yy=g(x)
u_x=0 on left and right boundary
u=0 on upper and lower boundary

Use the following command to compile and run:
$ nvcc -o Sources Sources-v1.cu
$ ./Sources 128 1.8 1e-9 10000

allen.c:

This is the source file of the MPI program calculating the numerical solution of Allen-Cahn eation. In order to make an executable and get the output, one can simply use the following command in bash:

$ touch Allen.out
$ nano makefile
//Put the following in the makefile
allen: allen.o
	mpicc -std=c99 -o allen allen.o -lfftw3_mpi -lfftw3 -lm
allen.o: allen.c
	mpicc -std=c99 -c allen.c
// save and exit 
$ make
$ mpirun -np <p> allen <N> <vx> <vy> <b> <W> <M> <Seed> (or $mpirun -np <p> ./allen <N> <vx> <vy> <b> <W> <M> <Seed>)

 int p: #processors
 int N: #grids in both directions
 double vx, vy: flow direction
 double b, W: equation parameters
 int M: #steps
 (optional) long int Seed: a preset random seed

SOR.c:

This is the source file of the MPI program using SOR to calculate the numerical solution of Diffusion equation. In order to make an executable and get the output, one can simply use the following command in bash:

$ touch Sources.out
$ mpicc -o SOR SOR.c -lm
$ mpirun -np <p> SOR <N> <omega> <TOL> <M>(or mpirun -np <p> SOR <N> <omega> <TOL> <M>)

 int p: #processors
 int N: #grids in y directions
 double omega: Successive over-relaxation parameter
 double TOL: tolerance
 int M: #max steps
