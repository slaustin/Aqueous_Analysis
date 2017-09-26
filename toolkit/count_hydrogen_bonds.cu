#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define CHECK(call) {   const cudaError_t error = call; if (error != cudaSuccess) { printf("Error: %s:%d, ", __FILE__, __LINE__); printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); exit(1); }}

__global__ void countagua (int *a,int *b,int *c,int na,int nb,float *xx,float *yy,float *zz,int *convers,int natim,int x_min,int y_min,int z_min,int x_max,int y_max,int z_max,int x_side,int y_side,int z_side,int *space,int num_points,int num_protons) {

        int k=threadIdx.x + blockDim.x * blockIdx.x;
        int r=blockIdx.x;
        int kk,j,jj,q,xxx,yyy,zzz,rx,ry,rz;
        long int n_of_grid;
        float curr_dist,dist_a;

	if (k<na){
		kk=a[k]-1;
		for(j=0;j<nb;j++){
			jj=b[j]-1;
			if(kk != jj){
				if(convers[jj] == 1){
					curr_dist=0.0;
					curr_dist=sqrtf(((xx[kk]-xx[jj])*(xx[kk]-xx[jj]))+((yy[kk]-yy[jj])*(yy[kk]-yy[jj]))+((zz[kk]-zz[jj])*(zz[kk]-zz[jj])));
					if(curr_dist < 3.5){
						//Proton Loop
						for(q=1;q<=num_protons;q++){
							jj=b[j+q]-1;
							dist_a=0.0;
							dist_a=sqrtf(((xx[kk]-xx[jj])*(xx[kk]-xx[jj]))+((yy[kk]-yy[jj])*(yy[kk]-yy[jj]))+((zz[kk]-zz[jj])*(zz[kk]-zz[jj])));
							if(dist_a < 2.45){
								//Acceptor
								if ((xx[kk] >= x_min)&&(yy[kk] >= y_min)&&(zz[kk] >= z_min)&&(xx[kk] <= x_max)&&(yy[kk] <= y_max)&&(zz[kk] <= z_max)){
									c[k]+=1;}
								xxx=floor(xx[kk]);
								yyy=floor(yy[kk]);
								zzz=floor(zz[kk]);
								if ((xxx >= x_min)&&(yyy >= y_min)&&(zzz >= z_min)&&(xxx <= x_max)&&(yyy <= y_max)&&(zzz <= z_max)){
									for(rx=xxx-1;rx<xxx+2;rx++){
									for(ry=yyy-1;ry<yyy+2;ry++){
									for(rz=zzz-1;rz<zzz+2;rz++){
									if ((rx >= x_min)&&(ry >= y_min)&&(rz >= z_min)&&(rx <= x_max)&&(ry <= y_max)&&(rz <= z_max)){
										n_of_grid=(rx-x_min)+((ry-y_min)*x_side)+((rz-z_min)*x_side*y_side)+(r*x_side*y_side*z_side);
										if((n_of_grid >= 0)&&(n_of_grid < num_points)){space[n_of_grid]+=1;}
									}
									}
									}
									}
								}
								//Donor
								if ((xx[jj] >= x_min)&&(yy[jj] >= y_min)&&(zz[jj] >= z_min)&&(xx[jj] <= x_max)&&(yy[jj] <= y_max)&&(zz[jj] <= z_max)){
									c[k]+=1;}
								xxx=floor(xx[jj]);
								yyy=floor(yy[jj]);
								zzz=floor(zz[jj]);
								if ((xxx >= x_min)&&(yyy >= y_min)&&(zzz >= z_min)&&(xxx <= x_max)&&(yyy <= y_max)&&(zzz <= z_max)){
									for(rx=xxx-1;rx<xxx+2;rx++){
									for(ry=yyy-1;ry<yyy+2;ry++){
									for(rz=zzz-1;rz<zzz+2;rz++){
									if ((rx >= x_min)&&(ry >= y_min)&&(rz >= z_min)&&(rx <= x_max)&&(ry <= y_max)&&(rz <= z_max)){
										n_of_grid=(rx-x_min)+((ry-y_min)*x_side)+((rz-z_min)*x_side*y_side)+(r*x_side*y_side*z_side);
										if((n_of_grid >= 0)&&(n_of_grid < num_points)){space[n_of_grid]+=1;}
									}
									}
									}
									}
								}
							}
						}
					}
				}
			}
		}
	}

} // End of Global

extern "C" void water_wrapper_(int *frame_count, double *x,double *y,double *z,int *min_x,int *min_y,int *min_z,int *max_x, int *max_y, int *max_z,int *sidex,int *sidey,int *sidez, int *grid_array,int *natoma,int *lista,int *natomb,int *listb,int *conversion,int *natim, int *prot_num, int *curr_count)
{//main

int k,r,rr,rrr,t,blocks,threads,sum;
long int num_points,num_points2,n_of_grid,n_of_grid2;
int devCount;
int num_atom_a=*natoma;
int num_atom_b=*natomb;
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
int num_protons=*prot_num;
int *local_lista, *local_listb, *local_listc, *local_conversion, *local_space, *local_space2;
float *local_x, *local_y, *local_z;
int *dev_lista, *dev_listb, *dev_listc, *dev_conversion, *dev_space;
float *dev_x,*dev_y,*dev_z;

/////////////////////////////////////////////////////////////////

cudaGetDeviceCount(&devCount);

for (int i = 0; i < devCount; ++i){
        cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, i);
        threads=devProp.maxThreadsPerBlock;
}

blocks=0;
num_points=0;
num_points2=0;
blocks=ceil(float(num_atom_a)/float(threads))+1;
num_points=x_side*y_side*z_side*blocks;
num_points2=x_side*y_side*z_side;

local_lista=(int *)malloc(num_atom_a*sizeof(int));
if(local_lista == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}

local_listb=(int *)malloc(num_atom_b*sizeof(int));
if(local_listb == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}

local_listc=(int *)malloc(num_atom_a*sizeof(int));
if(local_listc == NULL){
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

local_conversion=(int *)malloc(n_atim*sizeof(int));
if(local_conversion == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}

local_space=(int *)malloc(num_points*sizeof(int));
if(local_space == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}

local_space2=(int *)malloc(num_points2*sizeof(int));
if(local_space2 == NULL){
   printf("Error: %s:%d, ", __FILE__, __LINE__);
   exit(1);}

memset(local_space,0,num_points*sizeof(int));
memset(local_space2,0,num_points2*sizeof(int));
memset(local_listc,0,num_atom_a*sizeof(int));
///////////////////////////////////////////////////////


// Transfer Pointers from Fortran to Local C Arrays
for (k=0;k<num_atom_a;k++){
     local_lista[k]=lista[k];
}

for (k=0;k<num_atom_b;k++){
     local_listb[k]=listb[k];
}

for (k=0;k<n_atim;k++){
     local_x[k]=x[k];
     local_y[k]=y[k];
     local_z[k]=z[k];
     local_conversion[k]=conversion[k];
}

///////////////////////////////////////////////////////

CHECK (cudaMalloc((void **) &dev_lista, num_atom_a*sizeof(int)) );
CHECK (cudaMalloc((void **) &dev_listb, num_atom_b*sizeof(int)) );
CHECK (cudaMalloc((void **) &dev_listc, num_atom_a*sizeof(int)) );
CHECK (cudaMalloc((void **) &dev_space, num_points*sizeof(int)) );
CHECK (cudaMalloc((void **) &dev_x, n_atim*sizeof(float)) );
CHECK (cudaMalloc((void **) &dev_y, n_atim*sizeof(float)) );
CHECK (cudaMalloc((void **) &dev_z, n_atim*sizeof(float)) );
CHECK (cudaMalloc((void **) &dev_conversion, n_atim*sizeof(int)) );


CHECK (cudaMemcpy(dev_x, local_x, n_atim*sizeof(float), cudaMemcpyHostToDevice) );
CHECK (cudaMemcpy(dev_y, local_y, n_atim*sizeof(float), cudaMemcpyHostToDevice) );
CHECK (cudaMemcpy(dev_z, local_z, n_atim*sizeof(float), cudaMemcpyHostToDevice) );
CHECK (cudaMemcpy(dev_lista, local_lista, num_atom_a*sizeof(int), cudaMemcpyHostToDevice) );
CHECK (cudaMemcpy(dev_listb, local_listb, num_atom_b*sizeof(int), cudaMemcpyHostToDevice) );
CHECK (cudaMemcpy(dev_listc, local_listc, num_atom_a*sizeof(int), cudaMemcpyHostToDevice) );
CHECK (cudaMemcpy(dev_conversion, local_conversion, n_atim*sizeof(int), cudaMemcpyHostToDevice) );
CHECK (cudaMemcpy(dev_space, local_space, num_points*sizeof(int), cudaMemcpyHostToDevice) );

///////////////////////////////////////////////////////

//printf("Launch!\n");
//printf("Blocks=%i\n",blocks);
//printf("Threads=%i\n",threads);

countagua<<<blocks,threads>>>(dev_lista,dev_listb,dev_listc,num_atom_a,num_atom_b,dev_x,dev_y,dev_z,dev_conversion,n_atim,minn_x,minn_y,minn_z,maxx_x,maxx_y,maxx_z,x_side,y_side,z_side,dev_space,num_points,num_protons);

CHECK (cudaMemcpy(local_space, dev_space, num_points*sizeof(int), cudaMemcpyDeviceToHost) );
CHECK (cudaMemcpy(local_listc, dev_listc, num_atom_a*sizeof(int), cudaMemcpyDeviceToHost) );

CHECK (cudaFree(dev_lista) ); 
CHECK (cudaFree(dev_listb) ); 
CHECK (cudaFree(dev_listc) ); 
CHECK (cudaFree(dev_x) ); 
CHECK (cudaFree(dev_y) ); 
CHECK (cudaFree(dev_z) ); 
CHECK (cudaFree(dev_conversion) ); 
CHECK (cudaFree(dev_space) );
CHECK (cudaDeviceReset());

///////////////////////////////////////////////////////

n_of_grid=0;
n_of_grid2=0;
sum=0;

for (k=0;k<num_atom_a;k++){
    sum+=local_listc[k];
}

*curr_count+=sum;

sum=0;
//Collect Grid Counts Across Blocks
for (t=0;t<blocks;t++){
	for (r=minn_x;r<maxx_x+1;r++){
		for (rr=minn_y;rr<maxx_y+1;rr++){
			for (rrr=minn_z;rrr<maxx_z+1;rrr++){
				n_of_grid=(r-minn_x)+((rr-minn_y)*x_side)+((rrr-minn_z)*x_side*y_side)+(t*x_side*y_side*z_side);
				n_of_grid2=(r-minn_x)+((rr-minn_y)*x_side)+((rrr-minn_z)*x_side*y_side);
				local_space2[n_of_grid2]+=local_space[n_of_grid];
			}
		}
	}
}

//Update Grid Array
for (r=minn_x;r<maxx_x+1;r++){
	for (rr=minn_y;rr<maxx_y+1;rr++){
		for (rrr=minn_z;rrr<maxx_z+1;rrr++){
			n_of_grid=(r-minn_x)+((rr-minn_y)*x_side)+((rrr-minn_z)*x_side*y_side);
                        grid_array[n_of_grid]+=local_space2[n_of_grid];
		}
	}
}

///////////////////////////////////////////////////////

free(local_lista);
free(local_listb);
free(local_listc);
free(local_x);
free(local_y);
free(local_z);
free(local_conversion);
free(local_space);
free(local_space2);

}//Main
