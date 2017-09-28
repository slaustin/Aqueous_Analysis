#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define CHECK(call) {   const cudaError_t error = call; if (error != cudaSuccess) { printf("Error: %s:%d, ", __FILE__, __LINE__); printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); exit(1); }}

__global__ void filldensity (int *a,int na,float *xx,float *yy,float *zz,int x_min,int y_min,int z_min,int x_max,int y_max,int z_max,int x_side,int y_side,int z_side,float *d_field,int num_points,float *atom_radius) {

        int k=threadIdx.x + blockDim.x * blockIdx.x;
        int j,kk,R,xxx,yyy,zzz;
        float curr_dist,radius,dimensionality;
        float pi;

        pi=3.1415926535897931;

	if(k<num_points){
		//Convert Point to XYZ Coords
		zzz=k/(x_side*y_side);
		R=k%(x_side*y_side);
		yyy=R/x_side;
		xxx=R%x_side;
		xxx=xxx+x_min;
		yyy=yyy+y_min;
		zzz=zzz+z_min;
		for(j=0;j<na;j++){
			kk=a[j]-1;
			curr_dist=sqrtf(((xx[kk]-float(xxx))*(xx[kk]-float(xxx)))+((yy[kk]-float(yyy))*(yy[kk]-float(yyy)))+((zz[kk]-float(zzz))*(zz[kk]-float(zzz))));
			if(curr_dist <= 6.0){
				radius=powf(((atom_radius[kk])/2.0),2.0);
				dimensionality=(-3.0/2.0);
				d_field[k]+=powf((2.0*pi*radius),dimensionality)*(expf((-1.0*curr_dist*curr_dist)/(2.0*radius)));
			}
		}
	}

} // End of Global

extern "C" void water_wrapper_2_(int *frame_count, double *x,double *y,double *z,int *min_x,int *min_y,int *min_z,int *max_x, int *max_y, int *max_z,int *sidex,int *sidey,int *sidez,double *grid_array,int *natoma,int *lista,int *natim,double *radius_array)
{//main

int k,r,rr,rrr,blocks,threads;
long int num_points,n_of_grid;
int devCount;
int num_atom_a=*natoma;
int n_atim=*natim;
int minn_x=*min_x;
int minn_y=*min_y;
int minn_z=*min_z;
int maxx_x=*max_x;
int maxx_y=*max_y;
int maxx_z=*max_z;
int x_side=*sidex;
int y_side=*sidey;
int z_side=*sidez;
int *local_lista;
float *local_density_field;
float *local_x, *local_y, *local_z;
float *local_radius;
int *dev_lista;
float *dev_density_field;
float *dev_x,*dev_y,*dev_z;
float *dev_radius;

//CHECK (cudaSetDevice ( 0 ) );

/////////////////////////////////////////////////////////////////

cudaGetDeviceCount(&devCount);
//printf("CUDA Device Query...\n");
//printf("There are %d CUDA devices.\n", devCount);

// Iterate through devices
for (int i = 0; i < devCount; ++i){
        // Get device properties
        //printf("\nCUDA Device #%d\n", i);
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        //printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
        threads=devProp.maxThreadsPerBlock;
}

blocks=0;
num_points=0;
num_points=x_side*y_side*z_side;

blocks=ceil(float(num_points)/float(threads))+1;

local_lista=(int *)malloc(num_atom_a*sizeof(int));
if(local_lista == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}

local_x=(float *)malloc(n_atim*sizeof(float));
if(local_x == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}

local_y=(float *)malloc(n_atim*sizeof(float));
if(local_y == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}

local_z=(float *)malloc(n_atim*sizeof(float));
if(local_z == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}


local_radius=(float *)malloc(n_atim*sizeof(float));
if(local_radius == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}

local_density_field=(float *)malloc(num_points*sizeof(float));
if(local_density_field == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}


memset(local_density_field,0,num_points*sizeof(float));
///////////////////////////////////////////////////////


// Transfer Pointers from Fortran to Local C Arrays
for (k=0;k<num_atom_a;k++){
     local_lista[k]=lista[k];
}

for (k=0;k<n_atim;k++){
     local_x[k]=x[k];
     local_y[k]=y[k];
     local_z[k]=z[k];
     local_radius[k]=radius_array[k];
}

///////////////////////////////////////////////////////

CHECK (cudaMalloc((void **) &dev_lista, num_atom_a*sizeof(int)) );
CHECK (cudaMalloc((void **) &dev_density_field, num_points*sizeof(float)) );
CHECK (cudaMalloc((void **) &dev_x, n_atim*sizeof(float)) );
CHECK (cudaMalloc((void **) &dev_y, n_atim*sizeof(float)) );
CHECK (cudaMalloc((void **) &dev_z, n_atim*sizeof(float)) );
CHECK (cudaMalloc((void **) &dev_radius, n_atim*sizeof(float)) );


CHECK (cudaMemcpy(dev_x, local_x, n_atim*sizeof(float), cudaMemcpyHostToDevice) );
CHECK (cudaMemcpy(dev_y, local_y, n_atim*sizeof(float), cudaMemcpyHostToDevice) );
CHECK (cudaMemcpy(dev_z, local_z, n_atim*sizeof(float), cudaMemcpyHostToDevice) );
CHECK (cudaMemcpy(dev_radius, local_radius, n_atim*sizeof(float), cudaMemcpyHostToDevice) );
CHECK (cudaMemcpy(dev_lista, local_lista, num_atom_a*sizeof(int), cudaMemcpyHostToDevice) );
CHECK (cudaMemcpy(dev_density_field, local_density_field, num_points*sizeof(float), cudaMemcpyHostToDevice) );

///////////////////////////////////////////////////////

//printf("Launch!\n");
//printf("Blocks=%i\n",blocks);
//printf("Threads=%i\n",threads);


filldensity<<<blocks,threads>>>(dev_lista,num_atom_a,dev_x,dev_y,dev_z,minn_x,minn_y,minn_z,maxx_x,maxx_y,maxx_z,x_side,y_side,z_side,dev_density_field,num_points,dev_radius);

CHECK (cudaMemcpy(local_density_field, dev_density_field, num_points*sizeof(float), cudaMemcpyDeviceToHost) );

CHECK (cudaFree(dev_lista) ); 
CHECK (cudaFree(dev_x) ); 
CHECK (cudaFree(dev_y) ); 
CHECK (cudaFree(dev_z) ); 
CHECK (cudaFree(dev_radius) ); 
CHECK (cudaFree(dev_density_field) );
CHECK (cudaDeviceReset());

///////////////////////////////////////////////////////

n_of_grid=0;
//Update Grid Array
for (r=minn_x;r<maxx_x+1;r++){
	for (rr=minn_y;rr<maxx_y+1;rr++){
		for (rrr=minn_z;rrr<maxx_z+1;rrr++){
			n_of_grid=(r-minn_x)+((rr-minn_y)*x_side)+((rrr-minn_z)*x_side*y_side);
                        grid_array[n_of_grid]+=local_density_field[n_of_grid];
		}
	}
}

///////////////////////////////////////////////////////

free(local_lista);
free(local_x);
free(local_y);
free(local_z);
free(local_radius);
free(local_density_field);

}//Main
