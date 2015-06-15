//__author__ = "Konstantin Klementiev, Roman Chernikov"
//__date__ = "10 Apr 2015"

#pragma OPENCL EXTENSION cl_khr_fp64: enable

void GetLock(__global int* lock) {
   int status = atomic_xchg(lock, 1);
   while(status > 0)
   {
     status = atomic_xchg(lock, 1);
   }
}
//-----------------------------------------------
void ReleaseLock(__global int* lock)
{
   int status = atomic_xchg(lock, 0);
}
//-----------------------------------------------
__kernel void hist_1d(  const double a,
                        const double b,
                        const int max_bin, 
                        __global double* x,
                        __global double* weights,
                        __global double* hist_n,
                        __global int* loc_ng
                        )
{
        unsigned int ii = get_global_id(0);
        int hist_i = floor(a * x[ii] - b);
        if (hist_i >= 0 && hist_i < max_bin)
          {
              GetLock(&loc_ng[hist_i]);
              hist_n[hist_i]+=weights[ii];
              ReleaseLock(&loc_ng[hist_i]);
          }
        
           
}
//-----------------------------------------------
__kernel void hist_2d(  const double a_x,
                        const double b_x,
                        const double a_y,
                        const double b_y,
                        const int max_bin_x,
                        const int max_bin_y,
                        __global double* x,
                        __global double* y,
                        __global double* weights,
                        __global double* hist_n,
                        __global int* loc_ng
                        )
{
        unsigned int ii = get_global_id(0);

        int hist_ix = floor(a_x * x[ii] - b_x);
        int hist_iy = floor(a_y * y[ii] - b_y);

        if (hist_ix >= 0 && hist_ix < max_bin_x && hist_iy >= 0 && hist_iy < max_bin_y)
          {
              GetLock(&loc_ng[hist_iy * max_bin_x + hist_ix]);
              hist_n[hist_iy * max_bin_x + hist_ix] += weights[ii];
              ReleaseLock(&loc_ng[hist_iy * max_bin_x + hist_ix]);
          }   
}