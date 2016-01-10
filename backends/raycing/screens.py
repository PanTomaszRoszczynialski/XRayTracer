# -*- coding: utf-8 -*-
r"""
Screens
-------

Module :mod:`~xrt.backends.raycing.screens` defines a flat screen and a
hemispheric screen that intercept a beam and give its image.

.. autoclass:: xrt.backends.raycing.screens.MonitorPoints()
   :members: __init__, expose_wave_prepare, expose_wave

.. autoclass:: xrt.backends.raycing.screens.Screen()
   :members: __init__, expose, expose_wave_prepare, expose_wave

.. autoclass:: xrt.backends.raycing.screens.HemisphericScreen()
   :members: __init__

.. _waves:

Wave propagation (diffraction)
------------------------------

Time dependent diffraction
~~~~~~~~~~~~~~~~~~~~~~~~~~

We start from the Kirchhoff integral theorem in the general (time-dependent)
form [Born & Wolf]:

    .. math::
        V(r,t)=\frac 1{4\pi }\int _S\left\{[V]\frac{\partial }{\partial n}
        \left(\frac 1 s\right)-\frac 1{\mathit{cs}}\frac{\partial s}{\partial
        n}\left[\frac{\partial V}{\partial t}\right]-\frac 1 s\left[\frac
        {\partial V}{\partial n}\right]\right\}\mathit{dS},

where the integration is performed over the selected surface :math:`S`,
:math:`s` is the distance between the point :math:`r` and the surface
:math:`S`, :math:`\frac{\partial }{\partial n}` denotes differentiation along
the normal on the surface and the square brackets on :math:`V` terms denote
retarded values, i.e. the values at time :math:`t − s/c`. :math:`V` is a scalar
wave here but can represent any component of the actual electromagnetic wave
provided that the observation point is much further than the wave length
(surface currents are neglected here). :math:`V` depends on position and time;
this is something what we do not have in ray tracing. We obtain it from ray
characteristics by:

    .. math::
        V(s,t)=\frac 1{\sqrt{2\pi }}\int U_{\omega }(s)e^{-i\omega t}d\omega,

where :math:`U_{\omega }(s)` is interpreted as a monochromatic wave field and
therefore can be associated with a ray. Here, this is any component of the ray
polarization vector times its propagation factor :math:`e^{ikl(s)}`.
Substituting it into the Kirchhoff integral yields :math:`V(r,t)`. As we
visualize the wave fields in space and energy, the obtained :math:`V(r,t)` must
be back-Fourier transformed to the frequency domain represented by a new
re-sampled energy grid:

    .. math::
        U_{\omega '}(r)=\frac 1{\sqrt{2\pi }}\int V(r,t)e^{i\omega 't}
        \mathit{dt}.

Ingredients:

    .. math::
        [V]\frac{\partial }{\partial n}\left(\frac 1 s\right)=-\frac{(\hat
        {\vec s}\cdot \vec n)}{s^2}[V],

        \frac 1{\mathit{cs}}\frac{\partial s}{\partial n}\left[\frac{\partial
        V}{\partial t}\right]=\frac{ik} s(\hat{\vec s}\cdot \vec n)[V],

        \frac 1 s\left[\frac{\partial V}{\partial n}\right]=\frac{ik} s(\hat
        {\vec l}\cdot \vec n)[V],

where the hatted vectors are unit vectors: :math:`\hat {\vec l}` for incoming
direction, :math:`\hat {\vec s}` for outgoing direction, both being variable
over the diffracting surface. As :math:`1/s\ll k`, the 1st term is negligible
as compared to the second one.

Finally,

    .. math::
        U_{\omega '}(r)=\frac{-i}{8\pi ^2\hbar ^2c}\int e^{i(\omega '-\omega)t}
        \mathit{dt}\int \frac E s\left((\hat{\vec s}\cdot \vec n)+(\hat{\vec l}
        \cdot \vec n)\right)U_{\omega }(s)e^{ik(l(s)+s)}\mathit{dS}\mathit{dE}.

The time-dependent diffraction integral is not yet implemented in xrt.

Stationary diffraction
~~~~~~~~~~~~~~~~~~~~~~

If the time interval :math:`t` is infinite, the forward and back Fourier
transforms give unity. The Kirchhoff integral theorem is reduced then to its
monochromatic form. In this case the energy of the reconstructed wave is the
same as that of the incoming one. We can still use the general equation, where
we substitute:

    .. math::
        \delta (\omega -\omega ')=\frac 1{2\pi }\int e^{i(\omega '-\omega )t}
        \mathit{dt},

which yields:

    .. math::
        U_{\omega }(r)=-\frac {i k}{4\pi }\int \frac1 s\left((\hat{\vec s}\cdot
        \vec n)+(\hat{\vec l}\cdot \vec n)\right)U_{\omega }(s)e^{ik(l(s)+s)}
        \mathit{dS}.

How we treat the non-monochromaticity? We repeat the sequence of ray-tracing
from the source down to the diffracting surface for each energy individually.
For synchrotron sources, we also assume a single electron trajectory. This
single energy contributes fully coherently into the diffraction integral.
Different energies contribute incoherently, i.e. we add their intensities, not
amplitudes.

The input field amplitudes can, in principle, be taken from ray-tracing, as it
was done by [Shi_Reininger]_ as :math:`U_\omega(s) = \sqrt{I_{ray}(s)}`. This
has, however, a fundamental difficulty. The notion "intensity" in many ray
tracing programs, as in Shadow used in [Shi_Reininger]_, is different from the
physical meaning of intensity: "intensity" in Shadow is a placeholder for
reflectivity and transmittivity. The real intensity is represented by the
*density* of rays – this is the way the rays were sampled, while each ray has
:math:`I_{ray}(x, z) = 1` at the source [shadowGuide]_, regardless of the
actual intensity profile. Therefore the actual intensity must be reconstructed.
We tried to overcome this difficulty by computing the density of rays by (a)
histogramming and (b) kernel density estimation [KDE]_. However, the easiest
approach is to sample the source with uniform ray density (and not proportional
to intensity) and to assign to each ray its physical wave amplitudes as *s* and
*p* projections. In this case we do not have to reconstruct the physical
intensity. The uniform ray density is an option for the synchrotron sources in
:mod:`~xrt.backends.raycing.sources`.

Notice that this formulation does not require paraxial propagation and thus xrt
is more general than other wave propagation codes. For instance, it can work
with gratings and FZPs where the deflection angles may become large.

.. [Shi_Reininger] X. Shi, R. Reininger, M. Sanchez del Rio & L. Assoufid,
   A hybrid method for X-ray optics simulation: combining geometric ray-tracing
   and wavefront propagation, J. Synchrotron Rad. **21** (2014) 669–678.

.. [shadowGuide] F. Cerrina, "SHADOW User’s Guide" (1998).

.. [KDE] Michael G. Lerner (mglerner) (2013) http://www.mglerner.com/blog/?p=28

Usage
~~~~~

    .. warning::
        You need a good graphics card for running these calculations!

The Kirchhoff integral can be calculated at 3D points in the global coordinate
system or at points on a screen. In the former case, one uses the class
:class:`MonitorPoints` and in the latter case one uses the classes
:class:`Screen` or :class:`HemisphericScreen`.

The user must invoke two methods: (1) :meth:`expose_wave_prepare` to create the
diffracted beam arrays and (2) :meth:`expose_wave` to actually calculate the
diffraction. The two methods are split because the diffraction calculations
frequently need several repeats in order to accumulate enough wave samples for
attaining dark field at the image periphery. The second method can reside in a
loop that will accumulate the complex valued field amplitudes in the same beam
arrays defined by the first method.

Normalization
~~~~~~~~~~~~~

The amplitude factors in the Kirchhoff integral assure that the diffracted wave
has correct intensity and flux. This fact appears to be very handy in
calculating the efficiency of a grating or an FZP in a particular diffraction
order. Without proper amplitude factors one would need to calculate all the
significant orders and renormalize their total flux to the incoming one.

The resulting amplitude is correct provided the amplitudes on the diffracting
surface are properly normalized. The latter are normalized as follows. First,
the normalization constant :math:`X` is found from the flux integral:

    .. math::
        F = X^2 \int \left(|E_s|^2 + |E_p|^2 \right) (\hat{\vec l}\cdot \vec n)
        \mathit{dS}

by means of its Monte-Carlo representation:

    .. math::
        X^2 = \frac{F N }{\sum \left(|E_s|^2 + |E_p|^2 \right)
        (\hat{\vec l}\cdot \vec n) S} \equiv \frac{F N }{\Sigma_{J\angle} S}.

The area :math:`S` can be calculated by the user or it can be calculated
automatically by constructing a convex hull over the impact points of the
incoming rays. The voids, as in the case of a grating (shadowed areas) or an
FZP (the opaque zones) cannot be calculated by the convex hull and such cases
must be carefully considered by the user.

With the above normalization factors, the Kirchhoff integral calculated by
Monte-Carlo sampling gives the polarization components :math:`E_s` and
:math:`E_p` as (:math:`\gamma = s, p`):

    .. math::
        E_\gamma(r) = \frac{\sum K(r, s) E_\gamma(s) X S}{N} =
        \sum{K(r, s) E_\gamma(s)} \sqrt{\frac{F S} {\Sigma_{J\angle} N}}.

Finally, the Kirchhoff intensity :math:`\left(|E_s|^2 + |E_p|^2\right)(r)` must
be integrated over the screen area to give the flux.

    .. note::
        The above normalization steps are automatically done inside
        :meth:`expose_wave`. The integration of the Kirchhoff intensity over a
        flat screen or a hemispheric screen is provided by the array summation
        within the plotter, and :meth:`expose_wave` provides the surface
        element to make the sum an integral. In the case of calculations at 3D
        points in the global coordinate system (class :class:`MonitorPoints`)
        *the user* must provide the proper integration, as the connectivity of
        the points is generally unknown.
"""
__author__ = "Konstantin Klementiev, Roman Chernikov"
__date__ = "10 Apr 2015"
import numpy as np
from scipy.spatial import ConvexHull

from .. import raycing
from . import sources as rs
from . import materials as rm
from . import oes as roe
from . import apertures as ra
import time
import os
try:
    import pyopencl as cl
    isOpenCL = True
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
except ImportError:
    isOpenCL = False

_DEBUG = 20


class MonitorPoints(object):
    """This class is used to calculate the Kirchhoff integral for 3D points in
    the global coordinate system. One should first run
    :meth:`expose_wave_prepare` for the diffracting OE and the specified 3D
    points. The actual calculation of the Kirchhoff field is done by
    :meth:`expose_wave`."""
    def __init__(self, name='', targetOpenCL=None):
        u"""
        *name*: str.

        *targetOpenCL*:
            pyopencl can accelerate the calculation of diffraction integrals.
            If pyopencl is used, *targetOpenCL* is a tuple (iPlatform, iDevice)
            of indices in the lists cl.get_platforms() and
            platform.get_devices(), see the section :ref:`calculations_on_GPU`.
            None, if pyopencl is not wanted. Ignored if pyopencl is not
            installed.
        """
        self.name = name
        self.set_cl(targetOpenCL)
        self.bgW = None

    def set_cl(self, targetOpenCL):
        if (targetOpenCL is not None) and not isOpenCL:
            print "pyopencl is not available!"
        if (targetOpenCL is not None) and isOpenCL:
            iPlatform, iDevice = targetOpenCL
            platform = cl.get_platforms()[iPlatform]
            device = platform.get_devices()[iDevice]
            print('    Device - Name:  ' + device.name)
            self.cl_ctx = cl.Context(devices=[device])
            self.cl_queue = cl.CommandQueue(self.cl_ctx)
            #cl_file = os.path.join(os.path.dirname(__file__),
            #                       r'diffract.cl')
            cl_file = os.path.join(os.path.dirname(__file__),
                                   r'diffract.cl')
            kernelsource = open(cl_file).read()
            self.cl_program = cl.Program(self.cl_ctx, kernelsource).build()
            self.cl_mf = cl.mem_flags
        else:
            self.cl_ctx = None

    def expose_wave_prepare(self, oe, xglo, yglo, zglo, oeArea='auto'):
        """Creates the beam arrays used in wave diffraction calculations. *oe*
        is a diffracting element: a descendant from
        :class:`~xrt.backends.raycing.oes.OE`,
        :class:`~xrt.backends.raycing.apertures.RectangularAperture` or
        :class:`~xrt.backends.raycing.apertures.RoundAperture`. *xglo*, *yglo*
        and *zglo* are 1D arrays of equal length and represent 3D points in the
        global coordinate system.

        *oeArea*, if given, it is used for surface integration, otherwise the
        area is calculated from the impact coordinates by constructing a convex
        hull.
        """
        bgW = rs.Beam(nrays=len(xglo), forceState=1, withAmplitudes=True)
        bgW.EsTemp = np.zeros_like(bgW.Es)
        bgW.EpTemp = np.zeros_like(bgW.Ep)
        bgW.Jss[:] = 0
        bgW.x, bgW.y, bgW.z = xglo, yglo, zglo
        bgW.a[:] = (bgW.x - oe.center[0])
        bgW.b[:] = (bgW.y - oe.center[1])
        bgW.c[:] = (bgW.z - oe.center[2])
        path = (bgW.a**2 + bgW.b**2 + bgW.c**2)**0.5
        bgW.a[:] /= path
        bgW.b[:] /= path
        bgW.c[:] /= path
        bgW.path = path
        self.bgW = bgW
        self.oe = oe
        if isinstance(oeArea, str):
            self.externalArea = False
            self.oeArea = 0.
        else:
            self.externalArea = True
            self.oeArea = oeArea
        self.beamReflRays = 0L
#        self.beamInRays = 0L
#        self.beamInSumJ = 0.
        self.beamReflSumJ = 0.
        self.beamReflSumJnl = 0.
        self.expose_wave_repeats = 0L

    def expose_wave(self, beam):
        """Exposes the screen to the beam by calculating the Kirchhoff
        integral. *beam* is in global system, the returned beams are: a global
        and a local ones in ray reflection regime at the given oe and a global
        one in wave propagation regime at the given 3D points. The calculation
        points must be given beforehand by invoking
        :meth:`expose_wave_prepare`."""
        oe = self.oe
        if self.bgW is None:
            raise ValueError('Run "expose_wave_prepare" before "expose_wave"!')
        if _DEBUG > 10:
            self.t0 = time.time()

        if isinstance(oe, roe.OE):
            oeGlobal, oeLocal = oe.reflect(beam)
            secondDim = oeLocal.y
        elif isinstance(oe, (ra.RectangularAperture, ra.RoundAperture)):
            oeGlobal, oeLocal = oe.propagate(beam, needNewGlobal=True)
            secondDim = oeLocal.z
        else:
            raise ValueError('Wrong oe type!')

        good = oeLocal.state == 1
        goodlen = good.sum()
        lost = ~good
        oeGlobal.Es[lost] = 0
        oeGlobal.Ep[lost] = 0
        oeGlobal.Jss[lost] = 0
        oeGlobal.Jpp[lost] = 0
        oeGlobal.Jsp[lost] = 0
        if goodlen < 1e2:
            print "Not enough good rays: {0} of {1}".format(
                goodlen, len(oeGlobal.x))
            return oeGlobal, oeLocal, self.bgW
#        goodIn = beam.state == 1
#        self.beamInRays += goodIn.sum()
#        self.beamInSumJ += (beam.Jss[goodIn] + beam.Jpp[goodIn]).sum()

        if not self.externalArea:
            impactPoints = np.vstack((oeLocal.x[good], secondDim[good])).T
            try:  # convex hull
                hull = ConvexHull(impactPoints)
            except:  # QhullError
                raise ValueError('cannot normalize this way!')
            outerPts = impactPoints[hull.vertices, :]
            lines = np.hstack([outerPts, np.roll(outerPts, -1, axis=0)])
            area = 0.5 * abs(sum(x1*y2-x2*y1 for x1, y1, x2, y2 in lines))
            self.oeArea = max(self.oeArea, area)
            oe.surfaceArea = self.oeArea

        nglo = rs.Beam(0)  # container for the normals in BL coordinates
        if isinstance(oe, roe.OE):
            n = oe.local_n(oeLocal.x, oeLocal.y)
            nglo.a, nglo.b, nglo.c =\
                np.asarray([n[-3]]), np.asarray([n[-2]]), np.asarray([n[-1]])
            nl = (oeLocal.a*nglo.a + oeLocal.b*nglo.b +
                  oeLocal.c*nglo.c).flatten()
# rotate the normal from the local oe system to the local BL system:
            raycing.rotate_beam(
                nglo, None, rotationSequence=oe.rotationSequence,
                pitch=oe.pitch, roll=oe.roll+oe.positionRoll, yaw=oe.yaw,
                skip_xyz=True)
            if oe.extraPitch or oe.extraRoll or oe.extraYaw:
                raycing.rotate_beam(
                    nglo, None, rotationSequence=oe.extraRotationSequence,
                    pitch=oe.extraPitch, roll=oe.extraRoll, yaw=oe.extraYaw,
                    skip_xyz=True)
        else:  # aperture
            nglo.a, nglo.b, nglo.c = 0, 1, 0
            nl = oeGlobal.a*nglo.a + oeGlobal.b*nglo.b + oeGlobal.c*nglo.c

        self.expose_wave_repeats += 1
        self.beamReflRays += goodlen
        self.beamReflSumJ += (oeLocal.Jss[good] + oeLocal.Jpp[good]).sum()
        self.beamReflSumJnl += abs(((oeLocal.Jss[good] + oeLocal.Jpp[good]) *
                                   nl[good]).sum())

        if self.cl_ctx is None:
            kcode = self._diffraction_integral_conv
        else:
            kcode = self._diffraction_integral_CL

        Es, Ep = kcode(oe, oeGlobal, nglo, nl, good)
        self.bgW.EsTemp += Es
        self.bgW.EpTemp += Ep
        self.bgW.E[:] = beam.E[0]
        self.bgW.Es[:] = self.bgW.EsTemp
        self.bgW.Ep[:] = self.bgW.EpTemp
        self.bgW.Jss[:] = (self.bgW.Es * np.conj(self.bgW.Es)).real
        self.bgW.Jpp[:] = (self.bgW.Ep * np.conj(self.bgW.Ep)).real
        self.bgW.Jsp[:] = self.bgW.Es * np.conj(self.bgW.Ep)

        norm = self.oeArea * self.beamReflSumJ
        de = self.beamReflRays * self.beamReflSumJnl * self.expose_wave_repeats
        if de > 0:
            norm /= de
        self.bgW.Jss *= norm
        self.bgW.Jpp *= norm
        self.bgW.Jsp *= norm
        self.bgW.Es *= norm**0.5
        self.bgW.Ep *= norm**0.5
        if hasattr(beam, 'accepted'):  # for calculating flux
            self.bgW.accepted = beam.accepted
            self.bgW.acceptedE = beam.acceptedE
            self.bgW.seeded = beam.seeded
            self.bgW.seededI = beam.seededI

        if _DEBUG > 10:
#            print "Jss + Jpp", normOut
            print "expose_wave completed in {0}".format(time.time()-self.t0)
        return oeGlobal, oeLocal, self.bgW

    def _diffraction_integral_conv(self, oe, oeGlobal, nglo, nl, good,
                                   isFraunhofer=False):
        # rows are by image and columns are by inBeam
        if isFraunhofer:   # xglo, yglo, zglo hold aglo, bglo, cglo
            aglo = self.bgW.x[:, np.newaxis]
            bglo = self.bgW.y[:, np.newaxis]
            cglo = self.bgW.z[:, np.newaxis]
            pathAfter = -(aglo*oeGlobal.x[good] +
                          bglo*oeGlobal.y[good] +
                          cglo*oeGlobal.z[good])
            ns = aglo*nglo.a + bglo*nglo.b + cglo*nglo.c
        else:
            aglo = self.bgW.x[:, np.newaxis] - oeGlobal.x[good]
            bglo = self.bgW.y[:, np.newaxis] - oeGlobal.y[good]
            cglo = self.bgW.z[:, np.newaxis] - oeGlobal.z[good]
            pathAfter = (aglo**2 + bglo**2 + cglo**2)**0.5
            ns = (aglo*nglo.a + bglo*nglo.b + cglo*nglo.c) / pathAfter
        k = oeGlobal.E[good] / rm.chbar * 1e7  # [mm^-1]
        U = k * (nl+ns) *\
            np.exp(1j*k*(pathAfter+oeGlobal.path[good])) / pathAfter
        Es = (oeGlobal.Es[good] * U).sum(axis=1)
        Ep = (oeGlobal.Ep[good] * U).sum(axis=1)
        return Es, Ep

    def _diffraction_integral_CL(self, oe, oeGlo, nglo, nl, good,
                                 isFraunhofer=False):
        frontRays = np.int32(len(oeGlo.x[good]))
        imageRays = np.int32(len(self.bgW.x))

        x_mesh = np.float64(self.bgW.x)
        y_mesh = np.float64(self.bgW.y)
        z_mesh = np.float64(self.bgW.z)

        nl_loc = np.float64(nl)
        bOEglo_coord = np.array([oeGlo.x[good],
                                 oeGlo.y[good],
                                 oeGlo.z[good],
                                 0.*oeGlo.z[good]],
                                order='F', dtype=np.float64)
        surface_normal = np.array([nglo.a*np.ones(frontRays),
                                   nglo.b*np.ones(frontRays),
                                   nglo.c*np.ones(frontRays),
                                   0.*nglo.c*np.ones(frontRays)],
                                  order='F',
                                  dtype=np.float64)

        E_loc = np.float64(oeGlo.E[good])
        bOEpath = np.float64(oeGlo.path[good])
        Es_loc = np.complex128(oeGlo.Es[good])
        Ep_loc = np.complex128(oeGlo.Ep[good])
        Es_res = np.zeros(imageRays, dtype=np.complex128)
        Ep_res = np.zeros(imageRays, dtype=np.complex128)
        nl_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                           self.cl_mf.COPY_HOST_PTR,
                           hostbuf=nl_loc)
        xglo_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                             self.cl_mf.COPY_HOST_PTR, hostbuf=x_mesh)
        yglo_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                             self.cl_mf.COPY_HOST_PTR, hostbuf=y_mesh)
        zglo_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                             self.cl_mf.COPY_HOST_PTR, hostbuf=z_mesh)
        Es_loc_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                               self.cl_mf.COPY_HOST_PTR, hostbuf=Es_loc)
        Ep_loc_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                               self.cl_mf.COPY_HOST_PTR, hostbuf=Ep_loc)
        E_loc_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                              self.cl_mf.COPY_HOST_PTR, hostbuf=E_loc)
        beamOEglo_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                                  self.cl_mf.COPY_HOST_PTR,
                                  hostbuf=bOEglo_coord)
        surface_normal_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                                       self.cl_mf.COPY_HOST_PTR,
                                       hostbuf=surface_normal)
        beam_OE_loc_path_buf = cl.Buffer(self.cl_ctx,
                                         self.cl_mf.READ_ONLY |
                                         self.cl_mf.COPY_HOST_PTR,
                                         hostbuf=bOEpath)
        KirchS_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_WRITE |
                               self.cl_mf.COPY_HOST_PTR,
                               hostbuf=Es_res)
        KirchP_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_WRITE |
                               self.cl_mf.COPY_HOST_PTR,
                               hostbuf=Ep_res)

        local_size = None
        global_size = Es_res.shape

        if isFraunhofer:
            code = self.cl_program.integrate_fraunhofer
        else:
            code = self.cl_program.integrate_kirchhoff
        code(self.cl_queue,
             global_size,
             local_size,
             imageRays,
             frontRays,
             np.float64(rm.chbar),
             nl_buf,
             xglo_buf, yglo_buf, zglo_buf,
             Es_loc_buf, Ep_loc_buf,
             E_loc_buf, beamOEglo_buf,
             surface_normal_buf,
             beam_OE_loc_path_buf,
             KirchS_buf,
             KirchP_buf).wait()
        # retrieving the results
        cl.enqueue_read_buffer(self.cl_queue, KirchS_buf, Es_res).wait()
        cl.enqueue_read_buffer(self.cl_queue, KirchP_buf, Ep_res).wait()

        return Es_res, Ep_res


class Screen(MonitorPoints):
    def __init__(self, bl, name, center=[0, 0, 0], x='auto', z='auto',
                 targetOpenCL=None, compressX=None, compressZ=None):
        """
        *bl*: instance of :class:`~xrt.backends.raycing.BeamLine`.

        *name*: str.

        *center*: tuple of 3 floats, is a 3D point in the global system.

        *x, z*: 3-tuples or 'auto'. Normalized 3D vectors in the global system
            which determine the local x and z axes lying in the screen plane.
            If *x* is 'auto', it is horizontal and perpendicular to the beam
            line. If *z* is 'auto', it is vertical.

        *targetOpenCL*:
            pyopencl can accelerate the calculation of diffraction integrals.
            If pyopencl is used, *targetOpenCL* is a tuple (iPlatform, iDevice)
            of indices in the lists cl.get_platforms() and
            platform.get_devices(), see the section :ref:`calculations_on_GPU`.
            None, if pyopencl is not wanted. Ignored if pyopencl is not
            installed.

        *compressX, compressZ* are multiplicative compression coefficients for
            the corresponding axes. Typically are not needed. Can be useful to
            account for the viewing camera magnification or when the camera
            sees the screen at an angle.
        """
        self.name = name
        self.bl = bl
        bl.screens.append(self)
        self.ordinalNum = len(bl.screens)
        self.center = center
        self.set_orientation(x, z)
        self.compressX = compressX
        self.compressZ = compressZ
        self.set_cl(targetOpenCL)
        self.bgW = None
        self.blW = None

    def set_orientation(self, x=None, z=None):
        """Determines the local x, y and z in the global system."""
        if x == 'auto':
            self.x = self.bl.cosAzimuth, -self.bl.sinAzimuth, 0.
        elif x is not None:
            self.x = x
        if z == 'auto':
            self.z = 0., 0., 1.
        elif z is not None:
            self.z = z
        assert np.dot(self.x, self.z) == 0, 'x and z must be orthogonal!'
        self.y = np.cross(self.z, self.x)

    def local_to_global(self, x=0, y=0, z=0):
        xglo, yglo, zglo = \
            (self.center[0] + x*self.x[0] + y*self.y[0] + z*self.z[0],
             self.center[1] + x*self.x[1] + y*self.y[1] + z*self.z[1],
             self.center[2] + x*self.x[2] + y*self.y[2] + z*self.z[2])
        return xglo, yglo, zglo

    def expose(self, beam):
        """Exposes the screen to the beam. *beam* is in global system, the
        returned beam is in local system of the screen and represents the
        desired image."""
        blo = rs.Beam(copyFrom=beam, withNumberOfReflections=True)  # local
        # Converting the beam to the screen local coordinates
        blo.x[:] = beam.x[:] - self.center[0]
        blo.y[:] = beam.y[:] - self.center[1]
        blo.z[:] = beam.z[:] - self.center[2]

        xyz = blo.x, blo.y, blo.z
        # print "screen self.x", self.x
        blo.x[:], blo.y[:], blo.z[:] = \
            sum(c*b for c, b in zip(self.x, xyz)),\
            sum(c*b for c, b in zip(self.y, xyz)),\
            sum(c*b for c, b in zip(self.z, xyz))
        abc = beam.a, beam.b, beam.c
        blo.a[:], blo.b[:], blo.c[:] = \
            sum(c*b for c, b in zip(self.x, abc)),\
            sum(c*b for c, b in zip(self.y, abc)),\
            sum(c*b for c, b in zip(self.z, abc))

        maxa = np.max(abs(blo.a))
        maxb = np.max(abs(blo.b))
        maxc = np.max(abs(blo.c))
        maxMax = max(maxa, maxb, maxc)
        if maxMax == maxa:
            path = -blo.x / blo.a
        elif maxMax == maxb:
            path = -blo.y / blo.b
        else:
            path = -blo.z / blo.c
        blo.path += path

        blo.x[:] += blo.a * path
        blo.z[:] += blo.c * path
        blo.y[:] = 0.

        if hasattr(blo, 'Es'):
            propPhase = np.exp(-1j*(blo.E/rm.chbar*1e7)*path)
            blo.Es *= propPhase
            blo.Ep *= propPhase

        if self.compressX:
            blo.x[:] *= self.compressX
        if self.compressZ:
            blo.z[:] *= self.compressZ
        return blo

    def expose_wave_prepare(self, oe, dim1, dim2, oeArea='auto'):
        """Creates the beam arrays used in wave diffraction calculations. *oe*
        is a diffracting element: a descendant from
        :class:`~xrt.backends.raycing.oes.OE`,
        :class:`~xrt.backends.raycing.apertures.RectangularAperture` or
        :class:`~xrt.backends.raycing.apertures.RoundAperture`. *dim1* and
        *dim2* are *x* and *z* arrays for a flat screen or *phi* and *theta*
        arrays for a hemispheric screen. The two arrays are generally of
        different 1D shapes. They are used to create a 2D mesh by ``meshgrid``.
        """
        d1s, d2s = np.meshgrid(dim1, dim2)
        d1s = d1s.flatten()
        d2s = d2s.flatten()
        self.dS = (dim1[1] - dim1[0]) * (dim2[1] - dim2[0])
        if isinstance(self, HemisphericScreen):
            xlo, ylo, zlo, xglo, yglo, zglo = self.local_to_global(
                phi=d1s, theta=d2s)
            self.dS *= self.R**2
        else:
            xglo, yglo, zglo = self.local_to_global(x=d1s, z=d2s)

        MonitorPoints.expose_wave_prepare(self, oe, xglo, yglo, zglo, oeArea)

        blW = rs.Beam(copyFrom=self.bgW)
        if isinstance(self, HemisphericScreen):
            blW.x = xlo
            blW.y = ylo
            blW.z = zlo
            blW.phi = d1s
            blW.theta = d2s
        else:
            blW.x = d1s
            blW.y = np.zeros_like(d1s)
            blW.z = d2s
        self.blW = blW

    def expose_wave(self, beam):
        """Exposes the screen to the beam by calculating the Kirchhoff
        integral. *beam* is in global system, the returned beams are: a global
        and a local ones in ray reflection regime at the given oe and a global
        and a local ones in wave propagation regime at the screen. The
        calculation points must be given beforehand by invoking
        :meth:`expose_wave_prepare`."""
        if self.bgW is None:
            raise ValueError('Run "expose_wave_prepare" before "expose_wave"!')
        oeGlobal, oeLocal, tmp = MonitorPoints.expose_wave(self, beam)

        if isinstance(self, HemisphericScreen):
            norm = np.abs(np.cos(self.blW.theta)) * self.dS
        else:
            norm = self.dS
        self.blW.Es[:] = self.bgW.Es * norm**0.5
        self.blW.Ep[:] = self.bgW.Ep * norm**0.5
        self.blW.E[:] = self.bgW.E
        self.blW.Jss[:] = self.bgW.Jss * norm
        self.blW.Jpp[:] = self.bgW.Jpp * norm
        self.blW.Jsp[:] = self.bgW.Jsp * norm
        if hasattr(beam, 'accepted'):  # for calculating flux
            self.blW.accepted = beam.accepted
            self.blW.acceptedE = beam.acceptedE
            self.blW.seeded = beam.seeded
            self.blW.seededI = beam.seededI * len(self.bgW.x) / len(beam.x)

        return oeGlobal, oeLocal, self.bgW, self.blW


class HemisphericScreen(Screen):
    def __init__(self, bl, name, center, R, x='auto', z='auto',
                 targetOpenCL=None, phiOffset=0, thetaOffset=0):
        u"""
        *bl*: instance of :class:`~xrt.backends.raycing.BeamLine`.

        *name*: str.

        *center*: tuple of 3 floats, is a 3D point in the global system.

        *x, z*: 3-tuples or 'auto'. Normalized 3D vectors in the global system
            which determine the local x and z axes of the hemispheric screen.
            If *x* (the origin of azimuthal angle φ) is 'auto', it coincides
            with the beamline's *y*; if *z* (the polar axis) is 'auto', it is
            opposite to the beamline's *x*. The equator plane is then vertical.
            The polar angle θ is counted from -π/2 to π/2 with 0 at the
            equator and π/2 at the polar axis direction.

        *R*: radius of the hemisphere in mm.

        *targetOpenCL*:
            pyopencl can accelerate the calculation of diffraction integrals.
            If pyopencl is used, *targetOpenCL* is a tuple (iPlatform, iDevice)
            of indices in the lists cl.get_platforms() and
            platform.get_devices(), see the section :ref:`calculations_on_GPU`.
            None, if pyopencl is not wanted. Ignored if pyopencl is not
            installed.
        """
        self.name = name
        self.bl = bl
        bl.screens.append(self)
        self.ordinalNum = len(bl.screens)
        self.center = center
        self.R = R
        self.phiOffset = phiOffset
        self.thetaOffset = thetaOffset
        self.set_orientation(x, z)
        self.set_cl(targetOpenCL)

    def set_orientation(self, x=None, z=None):
        """Determines the local x, y and z in the global system."""
        if x == 'auto':
            self.x = self.bl.sinAzimuth, self.bl.cosAzimuth, 0.
        elif x is not None:
            self.x = x
        if z == 'auto':
            self.z = self.bl.cosAzimuth, -self.bl.sinAzimuth, 0.
        elif z is not None:
            self.z = z
        assert np.dot(self.x, self.z) == 0, 'x and z must be orthogonal!'
        self.y = np.cross(self.z, self.x)

    def local_to_global(self, phi, theta):
        thetaO = theta + self.thetaOffset
        phiO = phi + self.phiOffset
        z = np.sin(thetaO) * self.R
        y = np.cos(thetaO) * np.sin(phiO) * self.R
        x = np.cos(thetaO) * np.cos(phiO) * self.R
        xglo, yglo, zglo = Screen.local_to_global(self, x, y, z)
        return x, y, z, xglo, yglo, zglo

    def expose(self, beam):
            """Exposes the screen to the beam. *beam* is in global system, the
            returned beam is in local system of the screen and represents the
            desired image."""
            blo = rs.Beam(copyFrom=beam, withNumberOfReflections=True)  # local
            sqb_2 = (beam.a * (beam.x-self.center[0]) +
                     beam.b * (beam.y-self.center[1]) +
                     beam.c * (beam.z-self.center[2]))
            sqc = ((beam.x-self.center[0])**2 +
                   (beam.y-self.center[1])**2 +
                   (beam.z-self.center[2])**2 - self.R**2)
            path = -sqb_2 + (sqb_2**2 - sqc)**0.5
            blo.path += path
            rx = beam.x + beam.a*path - self.center[0]
            ry = beam.y + beam.b*path - self.center[1]
            rz = beam.z + beam.c*path - self.center[2]
            blo.z = rx*self.z[0] + ry*self.z[1] + rz*self.z[2]
            blo.y = rx*self.y[0] + ry*self.y[1] + rz*self.y[2]
            blo.x = rx*self.x[0] + ry*self.x[1] + rz*self.x[2]
            blo.theta = np.arcsin(blo.z / self.R) - self.thetaOffset
            blo.phi = np.arctan2(blo.y, blo.x) - self.phiOffset
            if hasattr(blo, 'Es'):
                propPhase = np.exp(1j * (blo.E / rm.chbar * 1e7) * path)
                blo.Es *= propPhase
                blo.Ep *= propPhase
            return blo
