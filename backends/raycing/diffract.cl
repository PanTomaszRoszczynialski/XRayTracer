//#pragma OPENCL EXTENSION cl_khr_fp64: enable
//__author__ = "Konstantin Klementiev, Roman Chernikov"
//__date__ = "10 Apr 2015"

//#include "materials.cl"
//__constant double twoPi = 6.283185307179586476;
__constant double fourPi = 12.566370614359172;
__constant double2 cmp0 = (double2)(0,0);
__constant double2 cmpi1 = (double2)(0,1);

double2 prod_c(double2 a, double2 b)
  {
    return (double2)(a.x*b.x - a.y*b.y, a.y*b.x + a.x*b.y);
  }
double2 conj_c(double2 a)
  {
    return (double2)(a.x, -a.y);
  }
double abs_c(double2 a)
  {
    return sqrt(a.x*a.x + a.y*a.y);
  }
double abs_c2(double2 a)
  {
    return (a.x*a.x + a.y*a.y);
  }

__kernel void integrate_kirchhoff(
                    const unsigned int imageLength,
                    const unsigned int fullnrays,
                    const double chbar,
                    __global double* cosGamma,
                    __global double* x_glo,
                    __global double* y_glo,
                    __global double* z_glo,
                    __global double2* Es,
                    __global double2* Ep,
                    __global double* E_loc,
                    __global double3* beamOEglo,
                    __global double3* oe_surface_normal,
                    __global double* beam_OE_loc_path,
                    __global double2* KirchS_gl,
                    __global double2* KirchP_gl)

{
    unsigned int i;

    double3 beam_coord_glo, beam_angle_glo;
    double2 gi, KirchS_loc, KirchP_loc;
    double pathAfter, cosAlpha, cr;
    unsigned int ii_screen = get_global_id(0);
//    double wavelength;
    double k, phase;

    KirchS_loc = cmp0;
    KirchP_loc = cmp0;

    beam_coord_glo.x = x_glo[ii_screen];
    beam_coord_glo.y = y_glo[ii_screen];
    beam_coord_glo.z = z_glo[ii_screen];
    for (i=0; i<fullnrays; i++)
      {
        beam_angle_glo = beam_coord_glo - beamOEglo[i];
        pathAfter = length(beam_angle_glo);
        cosAlpha = dot(beam_angle_glo, oe_surface_normal[i]) / pathAfter;
        k = E_loc[i] / chbar * 1e7;
//        wavelength = twoPi / k;
        phase = k * (pathAfter + beam_OE_loc_path[i]);
        cr = (cosAlpha + cosGamma[i]) / pathAfter;
        gi = (double2)(cr * cos(phase), cr * sin(phase));
        KirchS_loc += prod_c(gi, Es[i]);
        KirchP_loc += prod_c(gi, Ep[i]);
      }
  mem_fence(CLK_LOCAL_MEM_FENCE);

  KirchS_gl[ii_screen] = -prod_c(cmpi1*k/fourPi, KirchS_loc);
  KirchP_gl[ii_screen] = -prod_c(cmpi1*k/fourPi, KirchP_loc);
  mem_fence(CLK_LOCAL_MEM_FENCE);
}

__kernel void integrate_fraunhofer(
                    const unsigned int imageLength,
                    const unsigned int fullnrays,
                    const double chbar,
                    __global double* cosGamma,
                    __global double* a_glo,
                    __global double* b_glo,
                    __global double* c_glo,
                    __global double2* Es,
                    __global double2* Ep,
                    __global double* E_loc,
                    __global double3* beamOEglo,
                    __global double3* oe_surface_normal,
                    __global double* beam_OE_loc_path,
                    __global double2* KirchS_gl,
                    __global double2* KirchP_gl)

{
    unsigned int i;

//    double3 beam_coord_glo;
    double3 beam_angle_glo;
    double2 gi, KirchS_loc, KirchP_loc;
    double pathAfter, cosAlpha, cr;
    unsigned int ii_screen = get_global_id(0);
//    double wavelength;
    double k, phase;

    KirchS_loc = (double2)(0, 0);
    KirchP_loc = KirchS_loc;

    beam_angle_glo.x = a_glo[ii_screen];
    beam_angle_glo.y = b_glo[ii_screen];
    beam_angle_glo.z = c_glo[ii_screen];
    //if (ii_screen==128 || ii_screen==129) printf("Pix %i, beam_coord_glo %0.32v3f\n",ii_screen, beam_coord_glo);
    for (i=0; i<fullnrays; i++)
      {
        //printf("point %i\n",i);
        pathAfter = -dot(beam_angle_glo, beamOEglo[i]);
        cosAlpha = dot(beam_angle_glo, oe_surface_normal[i]);
        k = E_loc[i] / chbar * 1.e7;
//        wavelength = twoPi / k;
        phase = k * (pathAfter + beam_OE_loc_path[i]);
        cr = (cosAlpha + cosGamma[i]) / pathAfter;
        gi = (double2)(cr * cos(phase), cr * sin(phase));
        KirchS_loc += prod_c(gi, Es[i]);
        KirchP_loc += prod_c(gi, Ep[i]);
      }
  mem_fence(CLK_LOCAL_MEM_FENCE);

  KirchS_gl[ii_screen] = -prod_c(cmpi1*k/fourPi, KirchS_loc);
  KirchP_gl[ii_screen] = -prod_c(cmpi1*k/fourPi, KirchP_loc);
  mem_fence(CLK_LOCAL_MEM_FENCE);
}
