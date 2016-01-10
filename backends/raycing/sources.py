# -*- coding: utf-8 -*-
r"""
Sources
-------

Module :mod:`~xrt.backends.raycing.sources` defines the container class
:class:`Beam` that holds beam properties as numpy arrays and defines the beam
sources. The sources are geometric sources in continuous and mesh forms and
:ref:`synchrotron sources <own-synchrotron-sources>`.

The intensity and polarization of each ray are determined via coherency matrix.
This way is appropriate for incoherent addition of rays, in which the
components of the coherency matrix of different rays are simply added.

.. autoclass:: Beam
.. autoclass:: GeometricSource()
   :members: __init__
.. autoclass:: MeshSource()
   :members: __init__
.. autoclass:: CollimatedMeshSource
.. autofunction:: make_energy
.. autofunction:: make_polarization
.. autofunction:: rotate_coherency_matrix
.. autofunction:: shrink_source

.. _own-synchrotron-sources:

Synchrotron sources
^^^^^^^^^^^^^^^^^^^

The synchrotron sources have two implementations: based on own fully pythonic
or OpenCL aided calculations and based on external codes Urgent [Urgent]_ and
WS [WS]_. The latter codes have some drawbacks, as demonstrated in the section
:ref:`comparison-synchrotron-sources`, but nonetheless can be used for
comparison purposes. If you are going to use them, the codes are freely
available as parts of XOP package [XOP]_.

.. [Urgent] R. P. Walker, B. Diviacco, URGENT, A computer program for
   calculating undulator radiation spectral, angular, polarization and power
   density properties, ST-M-91-12B, July 1991, Presented at 4th Int. Conf. on
   Synchrotron Radiation Instrumentation, Chester, England, Jul 15-19, 1991

.. [WS] R. J. Dejus, Computer simulations of the wiggler spectrum, Nucl.
   Instrum. Methods Phys. Res., Sect. A, **347** (1994) 56-60

.. [XOP] M. Sanchez del Rio, R. J. Dejus, XOP: A Multiplatform Graphical User
   Interface for Synchrotron Radiation Spectral and Optics Calculations,
   SPIE Proc. **3152** (1997) 148; web page:
   http://www.esrf.eu/Instrumentation/software/data-analysis/xop2.3.

The internal synchrotron sources are based on the following works:
[Kim]_ and [Walker]_.
We use the general formulation for the flux distribution in 3-dimensional
phase space (solid angle and energy) [Kim]_:

    .. math::
        \mathcal{F}(\theta,\psi,E) &= \alpha\frac{\Delta \omega}{\omega}
        \frac{I_e}{e^{-}}(A_{\sigma}^2 + A_{\pi}^2)\\

For the bending magnets the amplitudes can be calculated analytically using the
modified Bessel functions :math:`K_v(y)`:

   .. math::
       \begin{bmatrix}
       A_{\sigma}\\
       A_{\pi}
       \end{bmatrix} &= \frac{\sqrt{3}}{2\pi}\gamma\frac{\omega}{\omega_c}
       (1+\gamma^2\psi^2)\begin{bmatrix}-i K_{2/3}(\eta)\\
       \frac{\gamma^2\psi^2}{\sqrt{1+\gamma^2\psi^2}}
       K_{1/3}(\eta)\end{bmatrix}

where
   .. math::
       \gamma &= \frac{E_e}{m_{e}c^2} = 1957E_e[GeV]\\
       \eta &= \frac{1}{2}\frac{\omega}{\omega_c}(1+\gamma^2\psi^2)^{3/2}\\
       \omega_c &= \frac{3\gamma^{3}c}{2\rho}\\
       \rho &= \frac{m_{e}c\gamma}{e^{-}B}

with :math:`I_e` - the current in the synchrotron ring, :math:`B` - magnetic
field strength, :math:`e^{-}, m_e`- the electron charge and mass,
:math:`c` - the speed of light.

Wiggler radiation relies on the same equation considering each pole as a
bending magnet with variable magnetic field/curvature radius:
:math:`\rho(\theta) = \sin(\arccos(\theta\gamma/K))`, where :math:`K` is
deflection parameter. Total flux is multiplied then by :math:`2N`, where
:math:`N` is the number of wiggler periods.

For the undulator sources the amplitude integrals must be calculated
numerically, starting from the magnetic field.

    .. math::
        \begin{bmatrix}
        A_{\sigma}\\
        A_{\pi}
        \end{bmatrix} &= \frac{\omega}{2\pi}\int\limits_{-\infty}^{+\infty}dt'
        \begin{bmatrix}\theta-\beta_x\\
        \psi-\beta_y \end{bmatrix}e^{i\omega (t' + R(t')/c)}

    .. math::
        B_x &= B_{x0}\sin(2\pi z /\lambda_u + \phi),\\
        B_y &= B_{y0}\sin(2\pi z /\lambda_u)

the corresponding velosity components are

    .. math::
        \beta_x &= \frac{K_y}{\gamma}\cos(\omega_u t),\\
        \beta_y &= -\frac{K_x}{\gamma}\cos(\omega_u t + \phi),

where :math:`\omega_u = 2\pi c /\lambda_u` - undulator frequency,
:math:`\phi` - phase shift between the magnetic field components. In this
simple case one can consider only one period in the integral phase term
replacing the exponential series by a factor of
:math:`\frac{\sin(N\pi\omega_1/\omega)}{\sin(\pi\omega_1/\omega)}`, where
:math:`\omega_1 = \frac{2\gamma^2}{1+K_x^2/2+K_y^2/2+\gamma^2(\theta^2+\psi^2)}
\omega_u`.

In the far-field approximation we consider the undulator as a point source and
replace the vector :math:`\mathbf{R}` by a projection
:math:`-\mathbf{n}\cdot\mathbf{r}`, where :math:`\mathbf{n} =
[\theta,\psi,1-\frac{1}{2}(\theta^2+\psi^2)]` - direction to the observation
point, :math:`\mathbf{r} = [\frac{K_y}{\gamma}\sin(\omega_u t'),
-\frac{K_x}{\gamma}\sin(\omega_u t' + \phi)`,
:math:`\beta_m\omega t'-\frac{1}{8\gamma^2}
(K_y^2\sin(2\omega_u t')+K_x^2\sin(2\omega_u t'+2\phi))]` - electron
trajectory, :math:`\beta_m = 1-\frac{1}{2\gamma^2}-\frac{K_x^2}{4\gamma^2}-
\frac{K_y^2}{4\gamma^2}`.

We directly calculate the integral using the Gauss-Legendre method. The
integration grid is generated around the center of each undulator period. Most
typical cases with reasonably narrow solid angles require around 10 integration
points per period for satisfactory convergence.

Custom field configurations i.e. tapered undulator require integration over
full undulator length, therefore the integration grid is multiplied by the
number of periods. Similar approach is used for the near field calculations,
where the undulator extension is taken into account.

.. [Kim] K.-J. Kim, Characteristics of Synchrotron Radiation, AIP Conference
   Proceedings, **184** (AIP, 1989) 565.

.. [Walker] R. Walker, Insertion devices: undulators and wigglers, CAS - CERN
   Accelerator School: Synchrotron Radiation and Free Electron Lasers,
   Grenoble, France, 22-27 Apr 1996: proceedings (CERN. Geneva, 1998) 129-190.

.. autoclass:: UndulatorUrgent()
   :members: __init__
.. autoclass:: WigglerWS()
   :members: __init__
.. autoclass:: BendingMagnetWS()
   :members: __init__

.. autoclass:: Undulator()
   :members: __init__
.. autoclass:: Wiggler()
   :members: __init__
.. autoclass:: BendingMagnet()
   :members: __init__

.. _comparison-synchrotron-sources:

Comparison of synchrotron source codes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Undulator spectrum across a harmonic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The above six classes result in ray distributions with the density of rays
proportional to intensity. This requires an algorithm of Monte Carlo ray
sampling that needs a 3D (two directions + energy) intensity distribution. The
classes using the external mesh-based codes interpolate the source intensity
pre-calculated over the user specified mesh. The latter three classes (internal
implementation of synchrotron sources) do not require interpolation, which
eliminates two problems: artefacts of interpolation and the need for the mesh
optimization. However, this requires the calculations of intensity *for each*
ray.

For bending magnet and wiggler sources these calculations are not heavy and
are actually faster than 3D interpolation. See the absence of interpolation
artefacts in :ref:`Synchrotron sources <synchrotron-sources>` in the gallery.

For an undulator the calculations are much more demanding and for a wide
angular acceptance the Monte Carlo ray sampling can be extremely inefficient.
To improve the efficiency, a reasonably small acceptance should be considered.

There are several codes that can calculate undulator spectra: Urgent [Urgent]_,
US [US]_, SPECTRA [SPECTRA]_. There is a common problem about them that the
energy spectrum may get *strong distortions* if calculated with a sparse
spatial and energy mesh. SPECTRA code seems to provide the best reference for
undulator spectra, which was used to optimize the meshes of the other codes.
The final comparison of the resulted undulator spectra around a single
harmonic is shown below.

.. [US] R. J. Dejus,
   *US: Program to calculate undulator spectra.* Part of XOP.

.. [SPECTRA] T. Tanaka and H. Kitamura, *SPECTRA - a synchrotron radiation
   calculation code*, J. Synchrotron Radiation **8** (2001) 1221-8.

.. note::

    If you are going to use UndulatorUrgent, you should optimize the spatial
    and energy meshes! The resulted ray distribution is strongly dependent on
    them, especially on the energy mesh. Try different numbers of points and
    various energy ranges.

.. image:: _images/compareUndulators.png
   :scale: 50 %

.. _tapering_comparison:

Undulator spectrum with tapering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The spectrum can be broadened by tapering the magnetic gap. The figure below
shows a comparison of xrt with SPECTRA [SPECTRA]_, YAUP [YAUP]_ and
experimental measurements [exp_taper]_. The gap values and the taper were
slightly varied in all three codes to reach the best match with the
experimental curve. We had to do so because in the other codes the taper is
not clearly defined (where is the gap invariant -- at the center or at one of
the ends?) and also because the nominal experimental gap and taper are not
fully trustful.

.. [YAUP] B. I. Boyanov, G. Bunker, J. M. Lee, and T. I. Morrison, *Numerical
   Modeling of Tapered Undulators*, Nucl. Instr. Meth. **A339** (1994) 596-603.

.. [exp_taper] Measured on 27 Nov 2013 on P06 beamline at Petra 3,
   R. Chernikov and O. Müller, unpublished.

The source code is in ``examples\withRaycing\01_SynchrotronSources``

.. image:: _images/compareTaper.png
   :scale: 50 %

Notice that not only the band width is affected by tapering. Also the
transverse distribution attains inhomogeneity which varies with energy, see the
animation below. Notice also that such a picture is sensitive to emittance; the
one below was calculated for the emittance of Petra III ring.

.. image:: _images/taperingEnergyScan.swf
   :width: 300
   :height: 205
.. image:: _images/zoomIcon.png
   :width: 20
   :target: _images/taperingEnergyScan.swf

Undulator spectrum in transverse plane
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The codes calculating undulator spectra -- Urgent [Urgent]_, US [US]_, SPECTRA
[SPECTRA]_ -- calculate either the spectrum of flux through a given aperture
*or* the transversal distribution at a fixed energy. It is not possible to
simultaneously have two dependencies: on energy *and* on transversal
coordinates.

Whereas xrt gives equal results to other codes in such univariate
distributions as flux through an aperture:

.. image:: _images/compareFlux.png
   :scale: 50 %

... and transversal distribution at a fixed energy:

+----------------+--------------------+-------------+
|                | SPECTRA [SPECTRA]_ |     xrt     |
+================+====================+=============+
| *E* = 4850 eV  |                    |             |
| (3rd harmonic) |  |spectra_lowE|    |  |xrt_lowE| |
+----------------+--------------------+-------------+
| *E* = 11350 eV |                    |             |
| (7th harmonic) |   |spectra_highE|  | |xrt_highE| |
+----------------+--------------------+-------------+

.. |spectra_lowE| image:: _images/undulator-E=04850eV-spectra.*
   :scale: 50 %
   :align: bottom
.. |spectra_highE| image:: _images/undulator-E=11350eV-spectra.*
   :scale: 50 %
   :align: bottom
.. |xrt_lowE| image:: _images/undulator-E=04850eV-xrt.*
   :scale: 50 %
.. |xrt_highE| image:: _images/undulator-E=11350eV-xrt.*
   :scale: 50 %

..., xrt can combine the two distributions in one image and thus be more
informative:

+----------------+-------------------+--------------------+
|                |      zoom in      |      zoom out      |
+================+===================+====================+
| *E* ≈ 4850 eV  |                   |                    |
| (3rd harmonic) |      |xrtLo|      |      |xrtLo5|      |
+----------------+-------------------+--------------------+
| *E* ≈ 11350 eV |                   |                    |
| (7th harmonic) |      |xrtHi|      |      |xrtHi5|      |
+----------------+-------------------+--------------------+

.. |xrtLo| image:: _images/oneHarmonic-E=04850eV-xrt.*
   :scale: 50 %
.. |xrtHi| image:: _images/oneHarmonic-E=11350eV-xrt.*
   :scale: 50 %
.. |xrtLo5| image:: _images/oneHarmonic-E=04850eV-xrt_x5.*
   :scale: 50 %
.. |xrtHi5| image:: _images/oneHarmonic-E=11350eV-xrt_x5.*
   :scale: 50 %

In particular, it is seen that divergence strongly depends on energy, even for
such a narrow energy band within one harmonic. It is also seen that the maximum
flux corresponds to slightly off-axis radiation (greenish color) but not to the
on-axis radiation (bluish color).

.. _near_field_comparison:

Undulator radiation in near field
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Notice that on the following pictures the p-polarized flux is almost 3 orders
of magnitude lower than the full flux.

+------------------------+--------------------+----------------+
|                        | SPECTRA [SPECTRA]_ |       xrt      |
+========================+====================+================+
|   near field at 5 m,   |                    |                |
|   full flux            |  |spectra_n05m0|   |  |xrt_n05m0|   |
+------------------------+--------------------+----------------+
|   near field at 5 m,   |                    |                |
|   p-polarized          |  |spectra_n05mP|   |  |xrt_n05mP|   |
+------------------------+--------------------+----------------+
|   far field at 5 m,    |                    |                |
|   full flux            |  |spectra_f05m0|   |  |xrt_f05m0|   |
+------------------------+--------------------+----------------+
|   far field at 5 m,    |                    |                |
|   p-polarized          |  |spectra_f05mP|   |  |xrt_f05mP|   |
+------------------------+--------------------+----------------+

.. |spectra_n05m0| image:: _images/spectra_near_R0=05m.*
   :scale: 50 %
   :align: bottom
.. |spectra_n05mP| image:: _images/spectra_near_R0=05m_p.*
   :scale: 50 %
   :align: bottom
.. |spectra_f05m0| image:: _images/spectra_far_R0=05m.*
   :scale: 50 %
   :align: bottom
.. |spectra_f05mP| image:: _images/spectra_far_R0=05m_p.*
   :scale: 50 %
   :align: bottom

.. |xrt_n05m0| image:: _images/1u_xrt4-n-near05m-E0-1TotalFlux.*
   :scale: 50 %
   :align: bottom
.. |xrt_n05mP| image:: _images/1u_xrt4-n-near05m-E0-3vertFlux.*
   :scale: 50 %
   :align: bottom
.. |xrt_f05m0| image:: _images/1u_xrt4-n-far05m-E0-1TotalFlux.*
   :scale: 50 %
   :align: bottom
.. |xrt_f05mP| image:: _images/1u_xrt4-n-far05m-E0-3vertFlux.*
   :scale: 50 %
   :align: bottom

+------------------------+--------------------+----------------+
|                        | SPECTRA [SPECTRA]_ |       xrt      |
+========================+====================+================+
|   near field at 25 m   |                    |                |
|   full flux            |  |spectra_n25m0|   |  |xrt_n25m0|   |
+------------------------+--------------------+----------------+
|   near field at 25 m,  |                    |                |
|   p-polarized          |  |spectra_n25mP|   |  |xrt_n25mP|   |
+------------------------+--------------------+----------------+
|   far field at 25 m,   |                    |                |
|   full flux            |  |spectra_f25m0|   |  |xrt_f25m0|   |
+------------------------+--------------------+----------------+
|   far field at 25 m,   |                    |                |
|   p-polarized          |  |spectra_f25mP|   |  |xrt_f25mP|   |
+------------------------+--------------------+----------------+

.. |spectra_n25m0| image:: _images/spectra_near_R0=25m.*
   :scale: 50 %
   :align: bottom
.. |spectra_n25mP| image:: _images/spectra_near_R0=25m_p.*
   :scale: 50 %
   :align: bottom
.. |spectra_f25m0| image:: _images/spectra_far_R0=25m.*
   :scale: 50 %
   :align: bottom
.. |spectra_f25mP| image:: _images/spectra_far_R0=25m_p.*
   :scale: 50 %
   :align: bottom

.. |xrt_n25m0| image:: _images/1u_xrt4-n-near25m-E0-1TotalFlux.*
   :scale: 50 %
   :align: bottom
.. |xrt_n25mP| image:: _images/1u_xrt4-n-near25m-E0-3vertFlux.*
   :scale: 50 %
   :align: bottom
.. |xrt_f25m0| image:: _images/1u_xrt4-n-far25m-E0-1TotalFlux.*
   :scale: 50 %
   :align: bottom
.. |xrt_f25mP| image:: _images/1u_xrt4-n-far25m-E0-3vertFlux.*
   :scale: 50 %
   :align: bottom
"""
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "29 Oct 2014"
import os
import sys
# import copy
# import math
import numpy as np
from scipy import ndimage
from scipy import optimize
from scipy import special
# if os.name == 'nt':
#    import win32console
import pickle
import time
from multiprocessing import Pool, cpu_count
from subprocess import Popen, PIPE

if sys.version_info < (3, 1):
    from string import maketrans
else:
    pass
#    import string
import gzip
from . import run as rr
from .. import raycing

try:
    import pyopencl as cl
    isOpenCL = True
    os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
except ImportError:
    isOpenCL = False

mc2 = 0.510998928  # MeV
c108 = 2.99792458  # 10^8 m/c
defaultEnergy = 9.0e3

E0 = 4.803e-10             # [esu]
C = 2.99792458e10          # [cm/sec]
M0 = 9.10938291e-28        # [g]
HPLANCK = 6.626069573e-27  # [erg*sec]
EV2ERG = 1.602176565e-12   # Energy conversion from [eV] to [erg]
K2B = 2 * np.pi * M0 * C**2 * 0.001 / E0
SIe0 = 1.602176565e-19
SIhplank = 6.626069573e-34
SIme = 9.10938291e-31
SIc = 2.99792458e8
FINE_STR = 1 / 137.03599976
PI = 3.1415926535897932384626433832795
PI2 = 6.283185307179586476925286766559
SQ3 = 1.7320508075688772935274463415059
#E2W = PI2 * SIe0 / SIhplank  # w = E2W * E[eV]
E2W = 1519267514747457.9195337718065469

_DEBUG = 20  # if non-zero, some diagnostics is printed out


class Beam(object):
    """Container for the beam arrays. *x, y, z* give the starting points.
    *a, b, c* give normalized vectors of ray directions (the source must take
    care about the normalization). *E* is energy. *Jss*, *Jpp* and *Jsp* are
    the components  of the coherency matrix. The latter one is complex. *path*
    is the total path length from the source to the last impact point. *theta*
    is the incidence angle. *order* is the order of grating diffraction. If
    multiple reflections are considered: *nRefl* is the number of reflections,
    *elevationD* is the maximum elevation distance between the rays and the
    surface as the ray travels from one impact point to the next one,
    *elevationX*, *elevationY*, *elevationZ* are the coordinates of the
    highest elevation points. If an OE uses a parametric representation,
    *s*, *phi*, *r* arrays store the impact points in the parametric
    coordinates.
    """
    def __init__(self, nrays=raycing.nrays, copyFrom=None, forceState=False,
                 withNumberOfReflections=False, withAmplitudes=False,
                 xyzOnly=False):
        if copyFrom is None:
            # coordinates of starting points
            self.x = np.zeros(nrays)
            self.y = np.zeros(nrays)
            self.z = np.zeros(nrays)
            if not xyzOnly:
                self.sourceSIGMAx = 0.
                self.sourceSIGMAz = 0.
                self.filamentDtheta = 0.
                self.filamentDpsi = 0.
                self.state = np.zeros(nrays, dtype=np.int)
                # components of direction
                self.a = np.zeros(nrays)
                self.b = np.ones(nrays)
                self.c = np.zeros(nrays)
                # total ray path
                self.path = np.zeros(nrays)
                # energy
                self.E = np.ones(nrays) * defaultEnergy
                # components of coherency matrix
                self.Jss = np.ones(nrays)
                self.Jpp = np.zeros(nrays)
                self.Jsp = np.zeros(nrays, dtype=complex)
                if withAmplitudes:
                    self.Es = np.zeros(nrays, dtype=complex)
                    self.Ep = np.zeros(nrays, dtype=complex)
        else:
            self.x = np.copy(copyFrom.x)
            self.y = np.copy(copyFrom.y)
            self.z = np.copy(copyFrom.z)
            self.sourceSIGMAx = copyFrom.sourceSIGMAx
            self.sourceSIGMAz = copyFrom.sourceSIGMAz
            self.filamentDtheta = copyFrom.filamentDtheta
            self.filamentDpsi = copyFrom.filamentDpsi
            self.state = np.copy(copyFrom.state)
            self.a = np.copy(copyFrom.a)
            self.b = np.copy(copyFrom.b)
            self.c = np.copy(copyFrom.c)
            self.path = np.copy(copyFrom.path)
            self.E = np.copy(copyFrom.E)
            self.Jss = np.copy(copyFrom.Jss)
            self.Jpp = np.copy(copyFrom.Jpp)
            self.Jsp = np.copy(copyFrom.Jsp)
            if withNumberOfReflections and hasattr(copyFrom, 'nRefl'):
                self.nRefl = np.copy(copyFrom.nRefl)
            if hasattr(copyFrom, 'elevationD'):
                self.elevationD = np.copy(copyFrom.elevationD)
                self.elevationX = np.copy(copyFrom.elevationX)
                self.elevationY = np.copy(copyFrom.elevationY)
                self.elevationZ = np.copy(copyFrom.elevationZ)
            if hasattr(copyFrom, 's'):
                self.s = np.copy(copyFrom.s)
            if hasattr(copyFrom, 'phi'):
                self.phi = np.copy(copyFrom.phi)
            if hasattr(copyFrom, 'r'):
                self.r = np.copy(copyFrom.r)
            if hasattr(copyFrom, 'theta'):
                self.theta = np.copy(copyFrom.theta)
            if hasattr(copyFrom, 'order'):
                self.order = np.copy(copyFrom.order)
            if hasattr(copyFrom, 'accepted'):  # for calculating flux
                self.accepted = copyFrom.accepted
                self.acceptedE = copyFrom.acceptedE
                self.seeded = copyFrom.seeded
                self.seededI = copyFrom.seededI
            if hasattr(copyFrom, 'Es'):
                self.Es = np.copy(copyFrom.Es)
                self.Ep = np.copy(copyFrom.Ep)
        if type(forceState) == int:
            self.state[:] = forceState

    def concatenate(self, beam):
        """Adds *beam* to *self*. Useful when more than one source is
        presented."""
        self.state = np.concatenate((self.state, beam.state))
        self.x = np.concatenate((self.x, beam.x))
        self.y = np.concatenate((self.y, beam.y))
        self.z = np.concatenate((self.z, beam.z))
        self.a = np.concatenate((self.a, beam.a))
        self.b = np.concatenate((self.b, beam.b))
        self.c = np.concatenate((self.c, beam.c))
        self.path = np.concatenate((self.path, beam.path))
        self.E = np.concatenate((self.E, beam.E))
        self.Jss = np.concatenate((self.Jss, beam.Jss))
        self.Jpp = np.concatenate((self.Jpp, beam.Jpp))
        self.Jsp = np.concatenate((self.Jsp, beam.Jsp))
        if hasattr(self, 'nRefl') and hasattr(beam, 'nRefl'):
            self.nRefl = np.concatenate((self.nRefl, beam.nRefl))
        if hasattr(self, 'elevationD') and hasattr(beam, 'elevationD'):
            self.elevationD = np.concatenate(
                (self.elevationD, beam.elevationD))
            self.elevationX = np.concatenate(
                (self.elevationX, beam.elevationX))
            self.elevationY = np.concatenate(
                (self.elevationY, beam.elevationY))
            self.elevationZ = np.concatenate(
                (self.elevationZ, beam.elevationZ))
        if hasattr(self, 's') and hasattr(beam, 's'):
            self.s = np.concatenate((self.s, beam.s))
        if hasattr(self, 'phi') and hasattr(beam, 'phi'):
            self.phi = np.concatenate((self.phi, beam.phi))
        if hasattr(self, 'r') and hasattr(beam, 'r'):
            self.r = np.concatenate((self.r, beam.r))
        if hasattr(self, 'theta') and hasattr(beam, 'theta'):
            self.theta = np.concatenate((self.theta, beam.theta))
        if hasattr(self, 'order') and hasattr(beam, 'order'):
            self.order = np.concatenate((self.order, beam.order))
        if hasattr(self, 'accepted') and hasattr(beam, 'accepted'):
            seeded = self.seeded + beam.seeded
            self.accepted = (self.accepted / self.seeded +
                             beam.accepted / beam.seeded) * seeded
            self.acceptedE = (self.acceptedE / self.seeded +
                              beam.acceptedE / beam.seeded) * seeded
            self.seeded = seeded
            self.seededI = self.seededI + beam.seededI
        if hasattr(self, 'Es') and hasattr(beam, 'Es'):
            self.Es = np.concatenate((self.Es, beam.Es))
            self.Ep = np.concatenate((self.Ep, beam.Ep))

    def filter_by_index(self, indarr):
        self.state = self.state[indarr]
        self.x = self.x[indarr]
        self.y = self.y[indarr]
        self.z = self.z[indarr]
        self.a = self.a[indarr]
        self.b = self.b[indarr]
        self.c = self.c[indarr]
        self.path = self.path[indarr]
        self.E = self.E[indarr]
        self.Jss = self.Jss[indarr]
        self.Jpp = self.Jpp[indarr]
        self.Jsp = self.Jsp[indarr]
        if hasattr(self, 'nRefl'):
            self.nRefl = self.nRefl[indarr]
        if hasattr(self, 'elevationD'):
            self.elevationD = self.elevationD[indarr]
            self.elevationX = self.elevationX[indarr]
            self.elevationY = self.elevationY[indarr]
            self.elevationZ = self.elevationZ[indarr]
        if hasattr(self, 's'):
            self.s = self.s[indarr]
        if hasattr(self, 'phi'):
            self.phi = self.phi[indarr]
        if hasattr(self, 'r'):
            self.r = self.r[indarr]
        if hasattr(self, 'theta'):
            self.theta = self.theta[indarr]
        if hasattr(self, 'order'):
            self.order = self.order[indarr]
        if hasattr(self, 'Es'):
            self.Es = self.Es[indarr]
            self.Ep = self.Ep[indarr]
        return self

    def replace_by_index(self, indarr, beam):
        self.state[indarr] = beam.state[indarr]
        self.x[indarr] = beam.x[indarr]
        self.y[indarr] = beam.y[indarr]
        self.z[indarr] = beam.z[indarr]
        self.a[indarr] = beam.a[indarr]
        self.b[indarr] = beam.b[indarr]
        self.c[indarr] = beam.c[indarr]
        self.path[indarr] = beam.path[indarr]
        self.E[indarr] = beam.E[indarr]
        self.Jss[indarr] = beam.Jss[indarr]
        self.Jpp[indarr] = beam.Jpp[indarr]
        self.Jsp[indarr] = beam.Jsp[indarr]
        if hasattr(self, 'nRefl') and hasattr(beam, 'nRefl'):
            self.nRefl[indarr] = beam.nRefl[indarr]
        if hasattr(self, 'elevationD') and hasattr(beam, 'elevationD'):
            self.elevationD[indarr] = beam.elevationD[indarr]
            self.elevationX[indarr] = beam.elevationX[indarr]
            self.elevationY[indarr] = beam.elevationY[indarr]
            self.elevationZ[indarr] = beam.elevationZ[indarr]
        if hasattr(self, 's') and hasattr(beam, 's'):
            self.s[indarr] = beam.s[indarr]
        if hasattr(self, 'phi') and hasattr(beam, 'phi'):
            self.phi[indarr] = beam.phi[indarr]
        if hasattr(self, 'r') and hasattr(beam, 'r'):
            self.r[indarr] = beam.r[indarr]
        if hasattr(self, 'theta') and hasattr(beam, 'theta'):
            self.theta[indarr] = beam.theta[indarr]
        if hasattr(self, 'order') and hasattr(beam, 'order'):
            self.order[indarr] = beam.order[indarr]
        if hasattr(self, 'Es') and hasattr(beam, 'Es'):
            self.Es[indarr] = beam.Es[indarr]
        if hasattr(self, 'Ep') and hasattr(beam, 'Ep'):
            self.Ep[indarr] = beam.Ep[indarr]
        return self

    def filter_good(self):
        return self.filter_by_index(self.state == 1)

    def absorb_intensity(self, inBeam):
        self.Jss = inBeam.Jss - self.Jss
        self.Jpp = inBeam.Jpp - self.Jpp
        self.Jsp = inBeam.Jsp - self.Jsp
        self.displayAsAbsorbedPower = True

    def project_energy_to_band(self, EnewMin, EnewMax):
        """Uniformly projects the energy array self.E to a new band determined
        by *EnewMin* and *EnewMax*. This function is useful for simultaneous
        ray tracing of white beam and monochromatic beam parts of a beamline.
        """
        EoldMin = np.min(self.E)
        EoldMax = np.max(self.E)
        if EoldMin >= EoldMax:
            return
        self.E[:] = EnewMin +\
            (self.E-EoldMin) / (EoldMax-EoldMin) * (EnewMax-EnewMin)

    def make_uniform_energy_band(self, EnewMin, EnewMax):
        """Makes a uniform energy distribution. This function is useful for
        simultaneous ray tracing of white beam and monochromatic beam parts of
        a beamline.
        """
        self.E[:] = np.random.uniform(EnewMin, EnewMax, len(self.E))


def copy_beam(
        beamTo, beamFrom, indarr, includeState=False, includeJspEsp=True):
    """Copies arrays of *beamFrom* to arrays of *beamTo*. The slicing of the
    arrays is given by *indarr*."""
    beamTo.x[indarr] = beamFrom.x[indarr]
    beamTo.y[indarr] = beamFrom.y[indarr]
    beamTo.z[indarr] = beamFrom.z[indarr]
    beamTo.a[indarr] = beamFrom.a[indarr]
    beamTo.b[indarr] = beamFrom.b[indarr]
    beamTo.c[indarr] = beamFrom.c[indarr]
    beamTo.path[indarr] = beamFrom.path[indarr]
    beamTo.E[indarr] = beamFrom.E[indarr]
    if includeState:
        beamTo.state[indarr] = beamFrom.state[indarr]
    if hasattr(beamFrom, 'nRefl') and hasattr(beamTo, 'nRefl'):
        beamTo.nRefl[indarr] = beamFrom.nRefl[indarr]
    if hasattr(beamFrom, 'order'):
        beamTo.order = beamFrom.order
    if hasattr(beamFrom, 'elevationD') and hasattr(beamTo, 'elevationD'):
        beamTo.elevationD[indarr] = beamFrom.elevationD[indarr]
        beamTo.elevationX[indarr] = beamFrom.elevationX[indarr]
        beamTo.elevationY[indarr] = beamFrom.elevationY[indarr]
        beamTo.elevationZ[indarr] = beamFrom.elevationZ[indarr]
    if hasattr(beamFrom, 'accepted'):
        beamTo.accepted = beamFrom.accepted
        beamTo.acceptedE = beamFrom.acceptedE
        beamTo.seeded = beamFrom.seeded
        beamTo.seededI = beamFrom.seededI
    if includeJspEsp:
        beamTo.Jss[indarr] = beamFrom.Jss[indarr]
        beamTo.Jpp[indarr] = beamFrom.Jpp[indarr]
        beamTo.Jsp[indarr] = beamFrom.Jsp[indarr]
        if hasattr(beamFrom, 'Es') and hasattr(beamTo, 'Es'):
            beamTo.Es[indarr] = beamFrom.Es[indarr]
            beamTo.Ep[indarr] = beamFrom.Ep[indarr]


def make_energy(distE, energies, nrays, filamentBeam=False):
    """Creates energy distributions with the distribution law given by *distE*.
    *energies* either determine the limits or is a sequence of discrete
    energies.
    """
    locnrays = 1 if filamentBeam else nrays
    if distE == 'normal':
        E = np.random.normal(energies[0], energies[1], locnrays)
    elif distE == 'flat':
        E = np.random.uniform(energies[0], energies[1], locnrays)
    elif distE == 'lines':
        E = np.array(energies)[np.random.randint(len(energies), size=locnrays)]
    return E


def make_polarization(polarization, bo, nrays=raycing.nrays):
    r"""Initializes the coherency matrix. The following polarizations are
    supported:

        1) horizontal (*polarization* is a string started with 'h'):

           .. math::

              J = \left( \begin{array}{ccc}1 & 0 \\ 0 & 0\end{array} \right)

        2) vertical (*polarization* is a string started with 'v'):

           .. math::

              J = \left( \begin{array}{ccc}0 & 0 \\ 0 & 1\end{array} \right)

        3) at +45º (*polarization* = '+45'):

           .. math::

              J = \frac{1}{2}
              \left( \begin{array}{ccc}1 & 1 \\ 1 & 1\end{array} \right)

        4) at -45º (*polarization* = '-45'):

           .. math::

              J = \frac{1}{2}
              \left( \begin{array}{ccc}1 & -1 \\ -1 & 1\end{array} \right)

        5) right (*polarization* is a string started with 'r'):

          .. math::

              J = \frac{1}{2}
              \left( \begin{array}{ccc}1 & i \\ -i & 1\end{array} \right)

        5) left (*polarization* is a string started with 'l'):

           .. math::

              J = \frac{1}{2}
              \left( \begin{array}{ccc}1 & -i \\ i & 1\end{array} \right)

        7) unpolarized (*polarization* is None):

           .. math::

              J = \frac{1}{2}
              \left( \begin{array}{ccc}1 & 0 \\ 0 & 1\end{array} \right)

        8) user-defined (*polarization* is 4-sequence with):

           .. math::

              J = \left( \begin{array}{ccc}
              {\rm polarization[0]} &
              {\rm polarization[2]} + i * {\rm polarization[3]} \\
              {\rm polarization[2]} - i * {\rm polarization[3]} &
              {\rm polarization[1]}\end{array} \right)

        """
    def _fill_beam(Jss, Jpp, Jsp, Es, Ep):
        bo.Jss.fill(Jss)
        bo.Jpp.fill(Jpp)
        bo.Jsp.fill(Jsp)
        if hasattr(bo, 'Es'):
            bo.Es.fill(Es)
            if isinstance(Ep, str):
                bo.Ep[:] = np.random.uniform(size=nrays) * 2**(-0.5)
            else:
                bo.Ep.fill(Ep)

    if (polarization is None) or (polarization.startswith('un')):
        _fill_beam(0.5, 0.5, 0, 2**(-0.5), 'random phase')
    elif isinstance(polarization, tuple):
        if len(polarization) != 4:
            raise ValueError('wrong coherency matrix: must be a 4-tuple!')
        bo.Jss.fill(polarization[0])
        bo.Jpp.fill(polarization[1])
        bo.Jsp.fill(polarization[2] + 1j*polarization[3])
    else:
        if polarization.startswith('h'):
            _fill_beam(1, 0, 0, 1, 0)
        elif polarization.startswith('v'):
            _fill_beam(0, 1, 0, 0, 1)
        elif polarization == '+45':
            _fill_beam(0.5, 0.5, 0.5, 2**(-0.5), 2**(-0.5))
        elif polarization == '-45':
            _fill_beam(0.5, 0.5, -0.5, 2**(-0.5), -2**(-0.5))
        elif polarization.startswith('r'):
            _fill_beam(0.5, 0.5, 0.5j, 2**(-0.5), -1j * 2**(-0.5))
        elif polarization.startswith('l'):
            _fill_beam(0.5, 0.5, -0.5j, 2**(-0.5), 1j * 2**(-0.5))
        else:
            raise ValueError('wrong polarization!')


def rotate_coherency_matrix(beam, indarr, roll):
    r"""Rotates the coherency matrix :math:`J`:

    .. math::

        J = \left( \begin{array}{ccc}
        J_{ss} & J_{sp} \\
        J^*_{sp} & J_{pp}\end{array} \right)

    by angle :math:`\phi` around the beam direction as :math:`J' = R_{\phi}
    J R^{-1}_{\phi}` with the rotation matrix :math:`R_{\phi}` defined as:

    .. math::

        R_{\phi} = \left( \begin{array}{ccc}
        \cos{\phi} & \sin{\phi} \\
        -\sin{\phi} & \cos{\phi}\end{array} \right)
    """
#    if (roll == 0).all():
#        return beam.Jss[indarr], beam.Jpp[indarr], beam.Jsp[indarr]
    c = np.cos(roll)
    s = np.sin(roll)
    c2 = c**2
    s2 = s**2
    cs = c * s
    JssN = beam.Jss[indarr]*c2 + beam.Jpp[indarr]*s2 +\
        2*beam.Jsp[indarr].real*cs
    JppN = beam.Jss[indarr]*s2 + beam.Jpp[indarr]*c2 -\
        2*beam.Jsp[indarr].real*cs
    JspN = (beam.Jpp[indarr]-beam.Jss[indarr])*cs +\
        beam.Jsp[indarr].real*(c2-s2) + beam.Jsp[indarr].imag*1j
    return JssN, JppN, JspN


class GeometricSource(object):
    """Implements a geometric source - a source with the ray origin,
    divergence and energy sampled with the given distribution laws."""
    def __init__(
        self, bl, name, center=(0, 0, 0), nrays=raycing.nrays, distx='normal',
        dx=0.32, disty=None, dy=0, distz='normal', dz=0.018,
        distxprime='normal', dxprime=1e-3, distzprime='normal', dzprime=1e-4,
        distE='lines', energies=(defaultEnergy,),
            polarization='horizontal', filamentBeam=False):
        """
        *bl*: instance of :class:`~xrt.backends.raycing.BeamLine`

        *name*: str

        *center*: tuple of 3 floats
            3D point in global system.

        *nrays*: int

        *distx, disty, distz, distxprime, distzprime*:
            'normal', 'flat', 'annulus' or None.
            If is None, the corresponding arrays remain with the values got at
            the instantiation of :class:`Beam`.
            'annulus' sets a uniform distribution for (x and z) or for (xprime
            and zprime) pairs. You can assign 'annulus' to only one member in
            the pair.

        *dx, dy, dz, dxprime, dzprime*: float
            for normal distribution is sigma, for flat is full width or tuple
            (min, max), for annulus is tuple (rMin, rMax), otherwise is
            ignored.

        *distE*: 'normal', 'flat', 'lines', None.

        *energies*, all in eV: (centerE, sigmaE) for *distE* = 'normal',
            (minE, maxE) for *distE* = 'flat', a sequence of E values for
            *distE* = 'lines'

        *polarization*:
            'h[orizontal]', 'v[ertical]', '+45', '-45', 'r[ight]', 'l[eft]',
            None, custom. In the latter case the polarization is given by a
            tuple of 4 components of the coherency matrix:
            (Jss, Jpp, Re(Jsp), Im(Jsp)).
        *filamentBeam*: if True the source generates coherent monochromatic
            wavefronts. Required for the wave propagation calculations.
        """
        self.bl = bl
        bl.sources.append(self)
        self.ordinalNum = len(bl.sources)
        self.name = name
        self.center = center  # 3D point in global system
        self.nrays = nrays

        self.distx = distx
        self.dx = dx
        self.disty = disty
        self.dy = dy
        self.distz = distz
        self.dz = dz
        self.distxprime = distxprime
        self.dxprime = dxprime
        self.distzprime = distzprime
        self.dzprime = dzprime
        self.distE = distE
        if self.distE == 'lines':
            self.energies = np.array(energies)
        else:
            self.energies = energies
        self.polarization = polarization
        self.filamentBeam = filamentBeam

    def _apply_distribution(self, axis, distaxis, daxis):
        if (distaxis == 'normal') and (daxis > 0):
            axis[:] = np.random.normal(0, daxis, self.nrays)
        elif (distaxis == 'flat'):
            if raycing.is_sequence(daxis):
                aMin, aMax = daxis[0], daxis[1]
            else:
                if daxis <= 0:
                    return
                aMin, aMax = -daxis*0.5, daxis*0.5
            axis[:] = np.random.uniform(aMin, aMax, self.nrays)
#        else:
#            axis[:] = 0

    def _set_annulus(self, axis1, axis2, rMin, rMax, phiMin, phiMax):
        if rMax > rMin:
            A = 2. / (rMax**2 - rMin**2)
            r = np.sqrt(2*np.random.uniform(0, 1, self.nrays)/A + rMin**2)
        else:
            r = rMax
        phi = np.random.uniform(phiMin, phiMax, self.nrays)
        axis1[:] = r * np.cos(phi)
        axis2[:] = r * np.sin(phi)

    def shine(self, toGlobal=True, withAmplitudes=False, accuBeam=None):
        """Returns the source beam. If *toGlobal* is True, the output is in
        the global system. If *withAmplitudes* is True, the resulted beam
        contains arrays Es and Ep with the *s* and *p* components of the
        electric field.
        """
        bo = Beam(self.nrays, withAmplitudes=withAmplitudes)  # beam-out
        bo.state[:] = 1
# =0: ignored, =1: good,
# =2: reflected outside of working area, =3: transmitted without intersection
# =-NN: lost (absorbed) at OE#NN (OE numbering starts from 1!) If NN>1000 then
# the slit with ordinal number NN-1000 is meant.

# in local coordinate system:
        self._apply_distribution(bo.y, self.disty, self.dy)

        isAnnulus = False
        if (self.distx == 'annulus') or (self.distz == 'annulus'):
            isAnnulus = True
            if raycing.is_sequence(self.dx):
                rMin, rMax = self.dx
            else:
                isAnnulus = False
            if raycing.is_sequence(self.dz):
                phiMin, phiMax = self.dz
            else:
                phiMin, phiMax = 0, 2*np.pi
        if isAnnulus:
            self._set_annulus(bo.x, bo.z, rMin, rMax, phiMin, phiMax)
        else:
            self._apply_distribution(bo.x, self.distx, self.dx)
            self._apply_distribution(bo.z, self.distz, self.dz)

        isAnnulus = False
        if (self.distxprime == 'annulus') or (self.distzprime == 'annulus'):
            isAnnulus = True
            if raycing.is_sequence(self.dxprime):
                rMin, rMax = self.dxprime
            else:
                isAnnulus = False
            if raycing.is_sequence(self.dzprime):
                phiMin, phiMax = self.dzprime
            else:
                phiMin, phiMax = 0, 2*np.pi
        if isAnnulus:
            self._set_annulus(bo.a, bo.c, rMin, rMax, phiMin, phiMax)
        else:
            self._apply_distribution(bo.a, self.distxprime, self.dxprime)
            self._apply_distribution(bo.c, self.distzprime, self.dzprime)

# normalize (a,b,c):
        ac = bo.a**2 + bo.c**2
        if sum(ac > 1) > 0:
            bo.b[:] = (ac + 1)**0.5
            bo.a[:] /= bo.b
            bo.c[:] /= bo.b
            bo.b[:] = 1.0 / bo.b
        else:
            bo.b[:] = (1 - ac)**0.5
        if self.distE is not None:
            if accuBeam is None:
                bo.E[:] = make_energy(self.distE, self.energies, self.nrays,
                                      self.filamentBeam)
            else:
                bo.E[:] = accuBeam.E[:]
        make_polarization(self.polarization, bo, self.nrays)
        if toGlobal:  # in global coordinate system:
            raycing.virgin_local_to_global(self.bl, bo, self.center)
        return bo


class MeshSource(object):
    """Implements a point source representing a rectangular angular mesh of
    rays. Primarily, it is meant for internal usage for matching the maximum
    divergence to the optical sizes of optical elements."""
    def __init__(
        self, bl, name, center=(0, 0, 0), minxprime=-1e-4, maxxprime=1e-4,
        minzprime=-1e-4, maxzprime=1e-4, nx=11, nz=11,
        distE='lines', energies=(defaultEnergy,), polarization='horizontal',
            withCentralRay=True, autoAppendToBL=False):
        """
        *bl*: instance of :class:`~xrt.backends.raycing.BeamLine`

        *name*: str

        *center*: tuple of 3 floats
            3D point in global system.

        *minxprime, maxxprime, minzprime, maxzprime*: float
            Limits for the ungular distributions.

        *nx, nz*: int
            Numbers of points in x and z dircetions.

        *distE*: 'normal', 'flat', 'lines', None

        *energies*, all in eV: (centerE, sigmaE) for *distE* = 'normal',
            (minE, maxE) for *distE* = 'flat', a sequence of E values for
            *distE* = 'lines'

        *polarization*: 'h[orizontal]', 'v[ertical]', '+45', '-45', 'r[ight]',
            'l[eft]', None, custom. In the latter case the polarization is
            given by a tuple of 4 components of the coherency matrix:
            (Jss, Jpp, Re(Jsp), Im(Jsp))

        *withCentralRay*: bool
            If True, the 1st ray in the beam is along the nominal beamline
            direction.

        *autoAppendToBL*: bool
            If True, the source is added to the list of beamline sources.
            Otherwise the user must manually start it with :meth:`shine`.
        """
        self.bl = bl
        if autoAppendToBL:
            bl.sources.append(self)
            self.ordinalNum = len(bl.sources)
        self.withCentralRay = withCentralRay
        self.name = name
        self.center = center  # 3D point in global system
        self.minxprime = minxprime
        self.maxxprime = maxxprime
        self.minzprime = minzprime
        self.maxzprime = maxzprime
        self.nx = nx
        self.nz = nz
        self.nrays = self.nx * self.nz + int(withCentralRay)
        self.distE = distE
        if self.distE == 'lines':
            self.energies = np.array(energies)
        else:
            self.energies = energies
        self.polarization = polarization

    def shine(self, toGlobal=True):
        """Returns the source. If *toGlobal* is True, the output is in
        the global system."""
        self.dxprime = (self.maxxprime-self.minxprime) / (self.nx-1)
        self.dzprime = (self.maxzprime-self.minzprime) / (self.nz-1)
        bo = Beam(self.nrays)  # beam-out
        bo.state[:] = 1
# in local coordinate system:
        xx, zz = np.meshgrid(
            np.linspace(self.minxprime, self.maxxprime, self.nx),
            np.linspace(self.minzprime, self.maxzprime, self.nz))
        zz = np.flipud(zz)
        bo.a[int(self.withCentralRay):] = xx.flatten()
        bo.c[int(self.withCentralRay):] = zz.flatten()
# normalize (a,b,c):
        bo.b[:] = (bo.a**2 + 1.0 + bo.c**2)**0.5
        bo.a[:] /= bo.b
        bo.c[:] /= bo.b
        bo.b[:] = 1.0 / bo.b
        if self.distE is not None:
            bo.E[:] = make_energy(self.distE, self.energies, self.nrays)
        make_polarization(self.polarization, bo, self.nrays)
        if toGlobal:  # in global coordinate system:
            raycing.virgin_local_to_global(self.bl, bo, self.center)
        return bo


class NESWSource(MeshSource):
    """Implements a point source with 4 rays: N(ord), E(ast), S(outh), W(est).
    Used internally for matching the maximum divergence to the optical sizes of
    optical elements.
    """
    def shine(self, toGlobal=True):
        """Returns the source. If *toGlobal* is True, the output is in
        the global system."""
        bo = Beam(4)  # beam-out
        bo.state[:] = 1
# in local coordinate system:
        bo.a[0] = 0
        bo.a[1] = self.maxxprime
        bo.a[2] = 0
        bo.a[3] = self.minxprime
        bo.c[0] = self.maxzprime
        bo.c[1] = 0
        bo.c[2] = self.minzprime
        bo.c[3] = 0
# normalize (a,b,c):
        bo.b[:] = (bo.a**2 + 1.0 + bo.c**2)**0.5
        bo.a[:] /= bo.b
        bo.c[:] /= bo.b
        bo.b[:] = 1.0 / bo.b
        bo.z[:] += 0.05

        if toGlobal:  # in global coordinate system:
            raycing.virgin_local_to_global(self.bl, bo, self.center)
        return bo


class CollimatedMeshSource(object):
    """Implements a source representing a mesh of collimated rays. Is similar
    to :class:`MeshSource`.
    """
    def __init__(
        self, bl, name, center=(0, 0, 0), dx=1., dz=1., nx=11, nz=11,
        distE='lines', energies=(defaultEnergy,), polarization='horizontal',
            withCentralRay=True, autoAppendToBL=False):
        self.bl = bl
        if autoAppendToBL:
            bl.sources.append(self)
            self.ordinalNum = len(bl.sources)
        self.withCentralRay = withCentralRay
        self.name = name
        self.center = center  # 3D point in global system
        self.dx = dx
        self.dz = dz
        self.nx = nx
        self.nz = nz
        self.nrays = self.nx * self.nz + int(withCentralRay)
        self.distE = distE
        if self.distE == 'lines':
            self.energies = np.array(energies)
        else:
            self.energies = energies
        self.polarization = polarization

    def shine(self, toGlobal=True):
        """Returns the source beam. If *toGlobal* is True, the output
        is in the global system."""
        bo = Beam(self.nrays)  # beam-out
        bo.state[:] = 1
# in local coordinate system:
        xx, zz = np.meshgrid(
            np.linspace(-self.dx/2., self.dx/2., self.nx),
            np.linspace(-self.dz/2., self.dz/2., self.nz))
        zz = np.flipud(zz)
        bo.x[int(self.withCentralRay):] = xx.flatten()
        bo.z[int(self.withCentralRay):] = zz.flatten()
        if self.distE is not None:
            bo.E[:] = make_energy(self.distE, self.energies, self.nrays)
        make_polarization(self.polarization, bo, self.nrays)
        if toGlobal:  # in global coordinate system:
            raycing.virgin_local_to_global(self.bl, bo, self.center)
        return bo


def shrink_source(beamLine, beams, minxprime, maxxprime, minzprime, maxzprime,
                  nx, nz):
    """Utility function that does ray tracing with a mesh source and shrinks
    its divergence until the footprint beams match to the optical surface.
    Parameters:

        *beamline*: instance of :class:`~xrt.backends.raycing.BeamLine`

        *beams*: tuple of str

            Dictionary keys in the result of
            :func:`~xrt.backends.raycing.run.run_process()` corresponding to
            the wanted footprints.

        *minxprime, maxxprime, minzprime, maxzprime*: float

            Determines the size of the mesh source. This size can only be
            shrunk, not expanded. Therefore, you should provide it sufficiently
            big for your needs. Typically, min values are negative and max
            values are positive.

        *nx, nz*: int

            Sizes of the 2D mesh grid in *x* and *z* direction.

    Returns an instance of :class:`MeshSource` which can be used then for
    getting the divergence values.
    """
    if not isinstance(beams, tuple):
        beams = (beams,)
    storeSource = beamLine.sources[0]  # store the current 1st source
    for ibeam in beams:
        # discover which side of the footprint corresponds to which side of
        # divergence
        neswSource = NESWSource(
            beamLine, 'NESW', storeSource.center, minxprime*0.1, maxxprime*0.1,
            minzprime*0.1, maxzprime*0.1)
        beamLine.sources[0] = neswSource
        raycing_output = rr.run_process(beamLine, shineOnly1stSource=True)
        beam = raycing_output[ibeam]
        badNum = beam.state != 1
        if badNum.sum() > 0:
            print "cannot shrink the source!"
            raise
        sideDict = {'left': np.argmin(beam.x), 'right': np.argmax(beam.x),
                    'bottom': np.argmin(beam.y), 'top': np.argmax(beam.y)}
#        print sideDict
        checkSides = set(i for key, i in sideDict.iteritems())
        if len(checkSides) != 4:
            print "cannot shrink the source!"
            raise
        sideList = ['', '', '', '']
        for k, v in sideDict.iteritems():
            sideList[v] = k
#        print sideList
# end of discover which side of the footprint ...
        meshSource = MeshSource(
            beamLine, 'mesh', storeSource.center, minxprime, maxxprime,
            minzprime, maxzprime, nx, nz)
        beamLine.sources[0] = meshSource
        raycing_output = rr.run_process(beamLine, shineOnly1stSource=True)
        beam = raycing_output[ibeam]
        rectState = beam.state[1:].reshape((meshSource.nz, meshSource.nx))
#        badNum = (rectState < 0) | (rectState > 1)
        badNum = rectState != 1
        nxLeft, nxRight, nzBottom, nzTop = 0, 0, 0, 0
        while badNum.sum() > 0:
            badNumRow = badNum.sum(axis=1)
            badNumCol = badNum.sum(axis=0)
            badNumRowMax = 2*badNumRow.max() - badNum.shape[1]
            badNumColMax = 2*badNumCol.max() - badNum.shape[0]
            if badNumRowMax >= badNumColMax:
                izDel, = np.where(badNumRow == badNumRow.max())
                izDel = max(izDel)
                if izDel < meshSource.nz/2:
                    nzTop += 1
                else:
                    nzBottom += 1
                badNum = np.delete(badNum, izDel, axis=0)
            else:
                ixDel, = np.where(badNumCol == badNumCol.max())
                ixDel = max(ixDel)
                if ixDel < meshSource.nx/2:
                    nxLeft += 1
                else:
                    nxRight += 1
                badNum = np.delete(badNum, ixDel, axis=1)
#        print 'n', nzTop, nxRight, nzBottom, nxLeft
        if nxLeft > 1:
            nxLeft += 1
        if nxRight > 1:
            nxRight += 1
        if nzBottom > 1:
            nzBottom += 1
        if nzTop > 1:
            nzTop += 1
        cutDict = {
            'left': nxLeft, 'right': nxRight, 'bottom': nzBottom, 'top': nzTop}
        maxzprime -= cutDict[sideList[0]] * meshSource.dzprime
        maxxprime -= cutDict[sideList[1]] * meshSource.dxprime
        minzprime += cutDict[sideList[2]] * meshSource.dzprime
        minxprime += cutDict[sideList[3]] * meshSource.dxprime
        meshSource.maxzprime = maxzprime
        meshSource.maxxprime = maxxprime
        meshSource.minzprime = minzprime
        meshSource.minxprime = minxprime
#    print 'shrink_source: ', meshSource.minxprime, meshSource.maxxprime,\
#      meshSource.minzprime, meshSource.maxzprime
    beamLine.sources[0] = storeSource  # restore the 1st source
    beamLine.alarms = []
    return meshSource

# You should better modify these paths to XOP here, otherwise you have to give
# the path as a parameter of UndulatorUrgent, WigglerWS or BendingMagnetWS.
if os.name == 'posix':
    xopBinDir = r'/home/konkle/xop2.3/bin.linux'
else:
    xopBinDir = r'c:\Program Files\xop2.3\bin.x86'


def run_one(path, tmpwd, infile, msg=None):
    if _DEBUG:
        if msg is not None:
            if os.name == 'posix':
                sys.stdout.write("\r\x1b[K "+msg)
            else:
                sys.stdout.write("\r" + "    ")
                print msg, ' ',
            sys.stdout.flush()
    with open(os.devnull, 'w') as fn:
        cproc = Popen(path, stdin=PIPE, stdout=fn, cwd=tmpwd)
        cproc.communicate(infile)


def gzip_output(tmpwd, outName, msg=None):
    if _DEBUG:
        if msg is not None:
            if os.name == 'posix':
                sys.stdout.write("\r\x1b[K "+msg)
            else:
                sys.stdout.write("\r" + "    ")
                print msg, ' ',
            sys.stdout.flush()
    fname = os.path.join(tmpwd, outName)
# for Python 2.7+:
#    with open(fname, 'rb') as txtFile:
#        with gzip.open(fname + '.gz', 'wb') as zippedFile:
#            zippedFile.writelines(txtFile)
# for Python 2.7-:
    txtFile = open(fname, 'rb')
    zippedFile = gzip.open(fname + '.gz', 'wb')
    try:
        zippedFile.writelines(txtFile)
    finally:
        txtFile.close()
        zippedFile.close()
    os.remove(fname)


if sys.version_info < (3, 1):
    transFortranD = maketrans('dD', 'ee')
else:
    transFortranD = ''.maketrans('dD', 'ee')


def read_output(tmpwd, outName, skiprows, usecols, comments, useZip, msg=None):
    if _DEBUG:
        if msg is not None:
            if os.name == 'posix':
                sys.stdout.write("\r\x1b[K "+msg)
            else:
                sys.stdout.write("\r" + "    ")
                print msg, ' ',
            sys.stdout.flush()
    try:
        return np.loadtxt(
            os.path.join(tmpwd, outName+('.gz' if useZip else '')),
            skiprows=skiprows, unpack=True, usecols=usecols, comments=comments,
            converters={2: lambda s: float(s.translate(transFortranD))})
    except:
        pass


class UndulatorUrgent(object):
    def __init__(
        self, bl, name='UrgentU', center=(0, 0, 0), nrays=raycing.nrays,
        period=32., K=2.668, Kx=0., Ky=0., n=12, eE=6., eI=0.1,
        eSigmaX=134.2, eSigmaZ=6.325, eEpsilonX=1., eEpsilonZ=0.01,
        uniformRayDensity=False,
        eMin=1500, eMax=81500, eN=1000, eMinRays=None, eMaxRays=None,
        xPrimeMax=0.25, zPrimeMax=0.1, nx=25, nz=25, path=None,
            mode=4, icalc=1, useZip=True, order=3, processes='auto'):
        """
        The 1st instantiation of this class runs the Urgent code and saves
        its output into a ".pickle" file. The temporary directory "tmp_urgent"
        can then be deleted. If any of the Urgent parameters has changed since
        the previous run, the Urgent code is forced to redo the calculations.

        *period*: float
            magnet period (mm).
        *K* or *Ky*: float
            magnet deflection parameter (Ky) in the vertical field.
        *Kx*: float
            magnet deflection parameter in the horizontal field.
        *n*: int
            number of magnet periods.
        *eE*, *eI*: float
            electron beam energy (GeV) and current (A).
        *eSigmaX*, *eSigmaZ*: float
            rms horizontal and vertical electron beam sizes (µm).
        *eEpsilonX*, *eEpsilonZ*:
            horizontal and vertical electron beam emittance (nm rad).
        *eMin*, *eMax*: float
            minimum and maximum photon energy (eV) used by Urgent.
        *eN*: int
            number of photon energy intervals used by Urgent
        *eMinRays*, *eMaxRays*: float
            the range of energies for rays. If None, are set equal to *eMin*
            and *eMax*. These two parameters are useful for playing with the
            energy axis without having to force Urgent to redo the
            calculations each time.
        *xPrimeMax*, *zPrimeMax*:
            Half of horizontal and vertical acceptance (mrad).
        *nx*, *nz*:
            number of intervals in the horizontal and vertical directions from
            zero to maximum.
        *path*: str
            full path to Urgent executable. If None, it is set automatically
            from the module variable *xopBinDir*.
        *mode*: 1, 2 or 4
            the MODE parameter of Urgent. If =1, UndulatorUrgent scans energy
            and reads the xz distribution from Urgent. If =2 or 4,
            UndulatorUrgent scans x and z and reads energy spectrum (angular
            density for 2 or flux through a window for 4) from Urgent. The
            meshes for x, z, and E are restricted in Urgent: nx,nz<50 and
            nE<5000. You may overcome these restrictions if you scan the
            corresponding quantities outside of Urgent, i.e. inside of this
            class UndulatorUrgent. *mode* = 4 is by far most preferable.
        *icalc*:
            the ICALC parameter of Urgent.
        *useZip*: boolean
            use gzip module to compress the output files of Urgent. If True,
            the temporary storage takes much less space but a slightly bit
            more time.
        *order*: int, 1 or 3
            the order of the spline interpolation. 3 is recommended.
        *processes*: int or any other value for 'auto'
            the number of worker processes to use. If the type is not int then
            the number returned by multiprocessing.cpu_count() is used.
        """
        self.bl = bl
        bl.sources.append(self)
        self.ordinalNum = len(bl.sources)
        self.name = name
        self.center = center  # 3D point in global system
        self.nrays = nrays

        self.period = period
        self.K = K if Ky == 0 else Ky
        self.Kx = Kx
        self.n = n
        self.eE = eE
        self.gamma = eE / mc2 * 1e3
        self.eI = eI
        self.eSigmaX = eSigmaX
        self.eSigmaZ = eSigmaZ
        self.eEpsilonX = eEpsilonX
        self.eEpsilonZ = eEpsilonZ
        self.eMin = eMin
        self.eMax = eMax
        self.eN = eN
        if eMinRays is None:
            self.eMinRays = eMin
        else:
            self.eMinRays = eMinRays
        if eMaxRays is None:
            self.eMaxRays = eMax
        else:
            self.eMaxRays = eMaxRays
        self.logeMinRays = np.log(self.eMinRays)
        self.logeMaxRays = np.log(self.eMaxRays)
        self.xPrimeMax = xPrimeMax
        self.zPrimeMax = zPrimeMax
        self.nx = nx
        self.nz = nz
        self.path = path
        self.mode = mode
        self.icalc = icalc
        self.useZip = useZip
        if isinstance(processes, int):
            self.processes = processes
            pp = processes
        else:
            self.processes = None
            pp = cpu_count()
        if _DEBUG:
            print '{0} process{1} will be requested'.format(
                pp, ('' if pp == 1 else 'es'))
        self.xpads = len(str(self.nx))
        self.zpads = len(str(self.nz))
        self.Epads = len(str(self.eN))
# extra rows and columns to the negative part (reflect from the 1st quadrant)
# in order to have good spline coefficients. Otherwise the spline may have
# discontinuity at the planes x=0 and z=0.
        self.extraRows = 0
        self.order = order
        self.prefilter = self.order == 1
        self.run_and_save(pp)
        self.xzE = 4e3 * self.xPrimeMax * self.zPrimeMax *\
            (self.logeMaxRays-self.logeMinRays)  # =2[-Max to +Max]*2*(0.1%)
        self.fluxConst = self.Imax * self.xzE
        self.uniformRayDensity = uniformRayDensity

    def run_and_save(self, pp):
        tstart = time.time()
        self.run()
        if self.needRecalculate:
            if _DEBUG:
                print '. Finished after {0} seconds'.format(
                    time.time() - tstart)
        tstart = time.time()
        self.splines, self.Imax = self.make_spline_arrays(
            skiprows=32, cols1=(2, 3, 4, 5), cols2=(0, 2, 6, 7, 8))
        if _DEBUG:
            print '. Finished after {0} seconds'.format(time.time() - tstart)

    def code_name(self):
        return 'urgent'

    def comment_strings(self):
        return [" MAXIMUM", " TOTAL"]

    def prefix_save_name(self):
        if self.Kx > 0:
            return '4-elu-{0}'.format(self.code_name())
        else:
            return '1-und-{0}'.format(self.code_name())

    def make_input(self, x, z, E):
        # 1) ITYPE, PERIOD, KX, KY, PHASE, N
        # 2) EMIN, EMAX, NE
        # 3) ENERGY, CUR, SIGX, SIGY, SIGX1, SIGY1
        # 4) D, XPC, YPC, XPS, YPS, NXP, NYP
        # 5) MODE, ICALC, IHARM
        # 6) NPHI, NSIG, NALPHA, DALPHA, NOMEGA, DOMEGA
        infile = ''
        infile += '1 {0} {1} {2} 0. {3}\n'.format(
            self.period*1e-3, self.Kx, self.K, self.n)
        infile += '{0} {1} {2}\n'.format(E, self.eMax, self.eN)
        infile += '{0} {1} {2} {3} {4:.7f} {5:.7f}\n'.format(
            self.eE, self.eI, self.eSigmaX*1e-3, self.eSigmaZ*1e-3,
            self.eEpsilonX/self.eSigmaX, self.eEpsilonZ/self.eSigmaZ)
        if self.mode == 1:
            infile += '0. {0} {1} {2} {3} {4} {5}\n'.format(
                0, 0, 2*self.xPrimeMax, 2*self.zPrimeMax, self.nx, self.nz)
        elif self.mode == 2:
            infile += '0. {0} {1} {2} {3} {4} {5}\n'.format(
                x, z, self.xPrimeMax, self.zPrimeMax, self.nx, self.nz)
        elif self.mode == 4:
            infile += '0. {0} {1} {2} {3} {4} {5}\n'.format(
                x, z, self.xPrimeMax/self.nx, self.zPrimeMax/self.nz, 11, 11)
# ICALC=1 non-zero emittance, finite N
# ICALC=2 non-zero emittance, infinite N
# ICALC=3 zero emittance, finite N
        infile += '{0} {1} -1\n'.format(self.mode, self.icalc)
        infile += '80 7 4 0 0 0\n'
        return infile

    def tmp_wd_xz(self, cwd, ix, iz):
        return os.path.join(cwd, 'tmp_'+self.code_name(), 'x{0}z{1}'.format(
            (str(ix)).zfill(self.xpads), (str(iz)).zfill(self.zpads)))

    def tmp_wd_E(self, cwd, iE):
        return os.path.join(cwd, 'tmp_'+self.code_name(), 'E{0}'.format(
            (str(iE)).zfill(self.Epads)))

    def msg_xz(self, ix, iz):
        return '{0} of {1}, {2} of {3}'.format(
            (str(ix+1)).zfill(self.xpads),
            (str(len(self.xs))).zfill(self.xpads),
            (str(iz+1)).zfill(self.zpads),
            (str(len(self.zs))).zfill(self.zpads))

    def msg_E(self, iE):
        return '{0} of {1}'.format(
            (str(iE+1)).zfill(self.Epads),
            (str(len(self.Es))).zfill(self.Epads))

    def run(self, forceRecalculate=False, iniFileForEachDirectory=False):
        self.xs = np.linspace(0, self.xPrimeMax, self.nx+1)
        self.zs = np.linspace(0, self.zPrimeMax, self.nz+1)
        self.Es = np.linspace(self.eMin, self.eMax, self.eN+1)
        cwd = os.getcwd()
        inpName = os.path.join(cwd, self.prefix_save_name()+'.inp')
        infile = self.make_input(0, 0, self.eMin)
        self.needRecalculate = True
        if os.path.exists(inpName):
            saved = ""
            with open(inpName, 'r') as f:
                for line in f:
                    saved += line
            self.needRecalculate = saved != infile
        if self.needRecalculate:
            with open(inpName, 'w') as f:
                f.write(infile)
        cwd = os.getcwd()
        pickleName = os.path.join(cwd, self.prefix_save_name()+'.pickle')
        if not os.path.exists(pickleName):
            if self.mode == 1:
                for iE, E in enumerate(self.Es):
                    tmpwd = self.tmp_wd_E(cwd, iE)
                    outName = os.path.join(tmpwd, self.code_name() + '.out' +
                                           ('.gz' if self.useZip else ''))
                    if not os.path.exists(outName):
                        self.needRecalculate = True
                        break
            elif self.mode in (2, 4):
                for iz, z in enumerate(self.zs):
                    for ix, x in enumerate(self.xs):
                        tmpwd = self.tmp_wd_xz(cwd, ix, iz)
                        outName = os.path.join(
                            tmpwd, self.code_name() + '.out' +
                            ('.gz' if self.useZip else ''))
                        if not os.path.exists(outName):
                            self.needRecalculate = True
                            break
            else:
                raise ValueError("mode must be 1, 2 or 4!")
        if (not self.needRecalculate) and (not forceRecalculate):
            return

        if self.path is None:
            self.path = os.path.join(
                xopBinDir, self.code_name() +
                ('.exe' if os.name == 'nt' else ''))
        if not os.path.exists(self.path):
            raise ImportError("The file {0} does not exist!".format(self.path))
        pool = Pool(self.processes)
        if _DEBUG:
            print 'calculating with {0} ... '.format(self.code_name())
        if self.mode == 1:
            for iE, E in enumerate(self.Es):
                tmpwd = self.tmp_wd_E(cwd, iE)
                if not os.path.exists(tmpwd):
                    os.makedirs(tmpwd)
                infile = self.make_input(0, 0, E)
                if iniFileForEachDirectory:
                    inpName = os.path.join(tmpwd, self.code_name()+'.inp')
                    with open(inpName, 'w') as f:
                        f.write(infile)
                msg = self.msg_E(iE) if iE % 10 == 0 else None
                pool.apply_async(run_one, (self.path, tmpwd, infile, msg))
        elif self.mode in (2, 4):
            for iz, z in enumerate(self.zs):
                for ix, x in enumerate(self.xs):
                    tmpwd = self.tmp_wd_xz(cwd, ix, iz)
                    if not os.path.exists(tmpwd):
                        os.makedirs(tmpwd)
                    infile = self.make_input(x, z, self.eMin)
                    if iniFileForEachDirectory:
                        inpName = os.path.join(tmpwd, self.code_name()+'.inp')
                        with open(inpName, 'w') as f:
                            f.write(infile)
                    msg = self.msg_xz(ix, iz) if ix % 10 == 0 else None
                    pool.apply_async(run_one, (self.path, tmpwd, infile, msg))
        else:
            raise ValueError("mode must be 1, 2 or 4!")
        pool.close()
        pool.join()
        if _DEBUG:
            print ""
        if self.useZip:
            if _DEBUG:
                print 'zipping ... '
            poolz = Pool(self.processes)
            if self.mode == 1:
                for iE, E in enumerate(self.Es):
                    tmpwd = self.tmp_wd_E(cwd, iE)
                    msg = self.msg_E(iE) if iE % 10 == 0 else None
                    poolz.apply_async(gzip_output, (
                        tmpwd, self.code_name() + '.out', msg))
            elif self.mode in (2, 4):
                for iz, z in enumerate(self.zs):
                    for ix, x in enumerate(self.xs):
                        tmpwd = self.tmp_wd_xz(cwd, ix, iz)
                        msg = self.msg_xz(ix, iz) if ix % 10 == 0 else None
                        poolz.apply_async(gzip_output, (
                            tmpwd, self.code_name() + '.out', msg))
            poolz.close()
            poolz.join()

    def make_spline_arrays(self, skiprows, cols1, cols2):
        cwd = os.getcwd()
        pickleName = os.path.join(cwd, self.prefix_save_name()+'.pickle')
        if self.needRecalculate or (not os.path.exists(pickleName)):
            if _DEBUG:
                print 'reading ... '
            if self.mode == 1:
                I = np.zeros(
                    (self.Es.shape[0], self.xs.shape[0], self.zs.shape[0]))
                l1 = np.zeros_like(I)
                l2 = np.zeros_like(I)
                l3 = np.zeros_like(I)
                for iE, E in enumerate(self.Es):
                    tmpwd = self.tmp_wd_E(cwd, iE)
                    msg = self.msg_E(iE) if iE % 10 == 0 else None
                    res = read_output(
                        tmpwd, self.code_name()+'.out', skiprows,
                        cols1, self.comment_strings()[0], self.useZip, msg)
                    if res is not None:
                        It, l1t, l2t, l3t = res
                    else:
                        pass
#                        raise ValueError('Error in the calculation at ' +\
#                          'i={0}, E={1}'.format(iE, E))
                    if res is not None:
                        try:
                            I[iE, :, :] = \
                                np.reshape(It, (self.nx+1, self.nz+1))
                            l1[iE, :, :] = \
                                np.reshape(l1t, (self.nx+1, self.nz+1))
                            l2[iE, :, :] = \
                                np.reshape(l2t, (self.nx+1, self.nz+1))
                            l3[iE, :, :] = \
                                np.reshape(l3t, (self.nx+1, self.nz+1))
                        except:
                            pass
            elif self.mode in (2, 4):
                I = None
                for iz, z in enumerate(self.zs):
                    for ix, x in enumerate(self.xs):
                        tmpwd = self.tmp_wd_xz(cwd, ix, iz)
                        msg = self.msg_xz(ix, iz) if ix % 10 == 0 else None
                        res = read_output(
                            tmpwd, self.code_name()+'.out', skiprows,
                            cols2, self.comment_strings()[1], self.useZip, msg)
                        if res is not None:
                            self.Es, It, l1t, l2t, l3t = res
                            if self.mode == 4:
                                It /= self.xPrimeMax / (self.nx+0.5) *\
                                    self.zPrimeMax / (self.nz+0.5)
                        else:
                            pass
                        if I is None:
                            I = np.zeros((self.Es.shape[0], self.xs.shape[0],
                                          self.zs.shape[0]))
                            l1 = np.zeros_like(I)
                            l2 = np.zeros_like(I)
                            l3 = np.zeros_like(I)
                            I[:, ix, iz] = It
                            l1[:, ix, iz] = l1t
                            l2[:, ix, iz] = l2t
                            l3[:, ix, iz] = l3t
                        else:
                            if res is not None:
                                I[:, ix, iz], l1[:, ix, iz], l2[:, ix, iz],\
                                    l3[:, ix, iz] = It, l1t, l2t, l3t
            splines, Imax = self.save_spline_arrays(
                pickleName, (I, l1, l2, l3))
        else:
            if _DEBUG:
                print 'restoring arrays ... '
        splines, Imax = self.restore_spline_arrays(pickleName)
        if _DEBUG:
            print 'shape={0}, max={1}'.format(splines[0].shape, Imax)
        return splines, Imax

    def save_spline_arrays(self, pickleName, what):
        if _DEBUG:
            print '. Pickling splines to\n{0}'.format(pickleName)
        splines = []
        for ia, a in enumerate(what):
            a = np.concatenate((a[:, self.extraRows:0:-1, :], a), axis=1)
            a = np.concatenate((a[:, :, self.extraRows:0:-1], a), axis=2)
            if self.order == 3:
                spline = ndimage.spline_filter(a)
            else:
                spline = a
            splines.append(spline)
        Imax = np.max(what[0])
        with open(pickleName, 'wb') as f:
            pickle.dump((Imax, splines), f, -1)
        return splines, Imax

    def restore_spline_arrays(
            self, pickleName, findNewImax=True, IminCutOff=1e-50):
        with open(pickleName, 'rb') as f:
            Imax, savedSplines = pickle.load(f)
        try:
            if findNewImax:
                ind = [i for i in range(self.Es.shape[0]) if
                       self.eMinRays <= self.Es[i] <= self.eMaxRays]
                if len(ind) == 0:
                    fact = self.eN / (self.eMax-self.eMin)
                    ind = [(self.eMinRays-self.eMin) * fact,
                           (self.eMaxRays-self.eMin) * fact]
                elif len(ind) == 1:
                    ind = [ind[0], ind[0]]
                coords = np.mgrid[ind[0]:ind[-1],
                                  self.extraRows:self.nx+1+self.extraRows,
                                  self.extraRows:self.nz+1+self.extraRows]

                I = ndimage.map_coordinates(
                    savedSplines[0], coords, order=self.order,
                    prefilter=self.prefilter)
                Imax = I.max()
                _eMinRays = self.Es[min(np.nonzero(I > Imax*IminCutOff)[0])]
                if _eMinRays > self.eMinRays:
                    self.eMinRays = _eMinRays
                    print 'eMinRays has been corrected up to', _eMinRays
                _eMaxRays = self.Es[max(np.nonzero(I > Imax*IminCutOff)[0])]
                if _eMaxRays < self.eMaxRays:
                    self.eMaxRays = _eMaxRays
                    print 'eMaxRays has been corrected down to', _eMaxRays
        except ValueError:
            pass
        return savedSplines, Imax

    def intensities_on_mesh(self):
        Is = []
        coords = np.mgrid[0:self.Es.shape[0],
                          self.extraRows:self.nx+1+self.extraRows,
                          self.extraRows:self.nz+1+self.extraRows]
        for a in self.splines:
            aM = ndimage.map_coordinates(a, coords, order=self.order,
                                         prefilter=self.prefilter)
            Is.append(aM)
        return Is

    def find_electron_path(self, vec, K, npassed):
        anorm = vec * self.gamma / K
        phase = np.empty_like(anorm)
        a1 = np.where(abs(anorm) <= 1)[0]
        phase[a1] = np.arcsin(anorm[a1])
        a1 = np.where(abs(anorm) > 1)[0]
        phase[a1] = np.sign(
            anorm[a1]) * np.random.normal(np.pi/2, np.pi/2/K, len(anorm[a1]))
        phase[::2] = np.sign(phase[::2]) * np.pi - phase[::2]
        phase -= np.sign(phase) * np.pi *\
            np.random.random_integers(-self.n+1, self.n, npassed)
        y = self.period / 2 / np.pi * phase
        x = K * self.period / 2 / np.pi / self.gamma * np.cos(phase)
        a = K / self.gamma * np.sin(phase)
        return y, x, a

    def shine(self, toGlobal=True):
        """Returns the source beam. If *toGlobal* is True, the output
        is in the global system."""
        bo = None
        length = 0
        seeded = long(0)
        seededI = 0.
        while length < self.nrays:
            bot = Beam(self.nrays)  # beam-out
            seeded += self.nrays
            bot.state[:] = 1  # good
            bot.E = np.exp(np.random.uniform(self.logeMinRays,
                                             self.logeMaxRays, self.nrays))
#            bot.E = np.random.uniform(
#                self.eMinRays, self.eMaxRays, self.nrays)
# mrad:
            bot.a = np.tan(
                np.random.uniform(-1, 1, self.nrays)*self.xPrimeMax * 1e-3)
            bot.c = np.tan(
                np.random.uniform(-1, 1, self.nrays)*self.zPrimeMax * 1e-3)
            coords = np.array(
                [(bot.E - self.eMin)/(self.eMax - self.eMin) * self.eN,
                 np.abs(bot.a)/(self.xPrimeMax*1e-3)*self.nx + self.extraRows,
                 np.abs(bot.c)/(self.zPrimeMax*1e-3)*self.nz + self.extraRows])
# coords.shape = (3, self.nrays)
            Icalc = ndimage.map_coordinates(
                self.splines[0], coords, order=self.order,
                prefilter=self.prefilter)
            seededI += Icalc.sum() * self.xzE
            if self.uniformRayDensity:
                npassed = self.nrays
                Icalc[Icalc < 0] = 0
                I0 = Icalc * 4 * self.xPrimeMax * self.zPrimeMax
            else:
                I = np.random.uniform(0, 1, self.nrays)
                passed = np.where(I * self.Imax < Icalc)[0]
                npassed = len(passed)
                if npassed == 0:
                    print 'No good rays in this seed!', length, 'of',\
                        self.nrays, 'rays in total so far...'
                    continue
                I0 = 1.
                coords = coords[:, passed]
                bot.filter_by_index(passed)
#            print npassed, length

            l1 = ndimage.map_coordinates(self.splines[1], coords,
                                         order=self.order,
                                         prefilter=self.prefilter)
            l2 = ndimage.map_coordinates(self.splines[2], coords,
                                         order=self.order,
                                         prefilter=self.prefilter)
            l3 = ndimage.map_coordinates(self.splines[3], coords,
                                         order=self.order,
                                         prefilter=self.prefilter)
            if self.Kx == 0:
                l3[bot.c < 0] *= -1.
            if self.order == 3:
                l1[l1 < -1] = -1.
                l1[l1 > 1] = 1.
                l2[l2 < -1] = -1.
                l2[l2 > 1] = 1.
                l3[l3 < -1] = -1.
                l3[l3 > 1] = 1.
            bot.Jss[:] = (1 + l1) / 2. * I0
            bot.Jpp[:] = (1 - l1) / 2. * I0
            sign = 1 if isinstance(self, WigglerWS) else -1
            bot.Jsp[:] = sign * (l2 + 1j*l3) / 2. * I0
# origin coordinates:
            if isinstance(self, BendingMagnetWS):
                bot.y[:] = -bot.a * self.rho
                bot.x[:] = bot.a**2 * self.rho / 2
            elif isinstance(self, WigglerWS):
                if self.Kx > 0:
                    bot.y[:], bot.z[:], bot.c[:] = \
                        self.find_electron_path(bot.c, self.Kx, npassed)
                if self.K > 0:
                    bot.y[:], bot.x[:], bot.a[:] = \
                        self.find_electron_path(bot.a, self.K, npassed)
            else:
                pass

# as by Walker and by Ellaume; SPECTRA's value is two times smaller:
            sigma_r2 = 2 * (HPLANCK * C / EV2ERG / bot.E * 10 *
                            self.period*self.n) / (2 * np.pi)**2
            bot.sourceSIGMAx = ((self.eSigmaX*1e-3)**2 + sigma_r2)**0.5
            bot.sourceSIGMAz = ((self.eSigmaZ*1e-3)**2 + sigma_r2)**0.5
            bot.x[:] += np.random.normal(0, bot.sourceSIGMAx, npassed)
            bot.z[:] += np.random.normal(0, bot.sourceSIGMAz, npassed)

            if bo is None:
                bo = bot
            else:
                bo.concatenate(bot)
            length = len(bo.a)
        if length >= self.nrays:
            bo.accepted = length * self.fluxConst
            bo.acceptedE = bo.E.sum() * self.fluxConst * raycing.elec_e
            bo.seeded = seeded
            bo.seededI = seededI
        if length > self.nrays:
            bo.filter_by_index(slice(0, self.nrays))
# normalize (a,b,c):
        norm = (bo.a**2 + 1.0 + bo.c**2)**0.5
        bo.a /= norm
        bo.b /= norm
        bo.c /= norm
        if toGlobal:  # in global coordinate system:
            raycing.virgin_local_to_global(self.bl, bo, self.center)
        return bo


class WigglerWS(UndulatorUrgent):
    def __init__(self, *args, **kwargs):
        """Uses WS code. All the parameters are the same as in
        UndulatorUrgent."""
        kwargs['name'] = kwargs.pop('name', 'WSwiggler')
        kwargs['mode'] = kwargs.pop('mode', 1)
        UndulatorUrgent.__init__(self, *args, **kwargs)

    def run_and_save(self, pp):
        tstart = time.time()
        self.run(iniFileForEachDirectory=True)
        if self.needRecalculate:
            if _DEBUG:
                print '. Finished after {0} seconds'.format(
                    time.time() - tstart)
        tstart = time.time()
        self.splines, self.Imax = self.make_spline_arrays(
            skiprows=18, cols1=(2, 3, 4, 5), cols2=(0, 1, 2, 3, 4))
        if _DEBUG:
            print '. Finished after {0} seconds'.format(time.time() - tstart)

    def code_name(self):
        return 'ws'

    def comment_strings(self):
        return ["#", "#"]

    def prefix_save_name(self):
        return '2-wig-{0}'.format(self.code_name())

    def make_input(self, x, z, E, isBM=False):
        # 1) Name
        # 2) RING-ENERGY CURRENT
        # 3) PERIOD N KX KY
        # 4) EMIN EMAX NE
        # 5) D XPC YPC XPS YPS NXP XYP
        # 6) MODE
        if isBM:
            xxx = self.xPrimeMax / 50.
        else:
            if self.mode == 1:
                xxx = 2 * self.xPrimeMax
            elif self.mode == 2:
                xxx = self.xPrimeMax / self.nx
        infile = ''
        infile += self.name+'\n'
        infile += '{0} {1}\n'.format(self.eE, self.eI*1e3)
        infile += '{0} {1} 0. {2}\n'.format(self.period/10., self.n, self.K)
        infile += '{0} {1} {2}\n'.format(E, self.eMax, self.eN)
        if self.mode == 1:
            infile += '0. {0} {1} {2} {3} {4} {5}\n'.format(
                0, 0, xxx, 2*self.zPrimeMax, self.nx, self.nz)
        elif self.mode == 2:
            infile += '0. {0} {1} {2} {3} {4} {5}\n'.format(
                x, z, xxx, self.zPrimeMax/self.nz, self.nx, self.nz)
        infile += '{0}\n'.format(self.mode)
        return infile


class BendingMagnetWS(WigglerWS):
    def __init__(self, *args, **kwargs):
        """Uses WS code. B0 in Tesla. *K*, *n*, *period* and *nx* are set
        internally. The other parameters are the same as in UndulatorUrgent.
        """
        kwargs['K'] = 50.
        kwargs['n'] = 0.5
        kwargs['name'] = kwargs.pop('name', 'WSmagnet')
        kwargs['mode'] = kwargs.pop('mode', 1)
        self.B0 = kwargs.pop('B0')
        # kwargs['period'] = kwargs['K'] / (93.36 * self.B0)
        kwargs['period'] = K2B * kwargs['K'] / self.B0
        kwargs['nx'] = 1
        UndulatorUrgent.__init__(self, *args, **kwargs)
        self.rho = 10. / c108 * self.eE / self.B0 * 1e3  # mm

    def make_input(self, x, z, E):
        return WigglerWS.make_input(self, 0, z, E, True)

    def prefix_save_name(self):
        return '3-BM-{0}'.format(self.code_name())


class BendingMagnet(object):
    def __init__(self, bl, name='BM', center=(0, 0, 0), nrays=raycing.nrays,
                 eE=6.0, eI=0.1, eEspread=0., eSigmaX=None, eSigmaZ=None,
                 eEpsilonX=1., eEpsilonZ=0.01, betaX=20., betaZ=5.,
                 B0=1., rho=None, filamentBeam=False, uniformRayDensity=False,
                 eMin=5000., eMax=15000., eN=51, distE='eV',
                 xPrimeMax=0.5, zPrimeMax=0.5, nx=25, nz=25):
        """
        *bl*: instance of :class:`~xrt.backends.raycing.BeamLine`

        *name*: str

        *center*: tuple of 3 floats
            3D point in global system.
        *nrays*: int
            The number of rays sampled in one iteration.
        *eE*, *eI*:
            beam energy (GeV) and current (A)
        *eEspread*:
            energy spread relative to the beam energy.
        *eSigmaX*, *eSigmaZ*: float
            rms horizontal and vertical electron beam sizes (µm).
            Alternatively, betatron functions can be specified instead of the
            electron beam sizes.
        *eEpsilonX*, *eEpsilonZ*:
            horizontal and vertical electron beam emittance (nm rad).
        *betaX*, *betaZ*:
            betatron function (m). Alternatively, beam size can be specified.
        *B0*:
            magnetic field (T). Alternatively, specify *rho*.
        *rho*:
            curvature radius (m). Alternatively, specify *B0*.
        *eMin*, *eMax*:
            minimum and maximum photon energy (eV).
        *eN*: int
            number of photon energy intervals, used only in the test suit,
            not required in ray tracing.
        *distE*: 'eV' or 'BW'
            The resulted flux density is per 1 eV or 0.1% bandwidth. For ray
            tracing 'eV' is used.
        *xPrimeMax*, *zPrimeMax*:
            horizontal and vertical acceptance (mrad)
        *nx*, *nz*:
            number of intervals in the horizontal and vertical directions,
            used only in the test suit, not required in ray tracing.
        *filamentBeam*: if True the source generates coherent monochromatic
            wavefronts. Required for the wave propagation calculations.
        """
        if not isinstance(self, Wiggler):
            self.Np = 0.5
            self.B = B0
            self.ro = rho
            self.Ee = eE
            self.gamma = self.Ee * 1e9 * EV2ERG / (M0 * C**2)
            if self.ro:
                if not self.B:
                    self.B = M0 * C**2 * self.gamma / self.ro / E0 / 1e6
            elif self.B:
                self.ro = M0 * C**2 * self.gamma / self.B / E0 / 1e6
            self.isMPW = False

        self.bl = bl
        bl.sources.append(self)
        self.ordinalNum = len(bl.sources)
        self.name = name
        self.center = center  # 3D point in global system
        self.nrays = nrays
        self.dx = eSigmaX * 1e-3 if eSigmaX else None
        self.dz = eSigmaZ * 1e-3 if eSigmaZ else None
        self.eEpsilonX = eEpsilonX
        self.eEpsilonZ = eEpsilonZ
        self.I0 = eI
        self.eEspread = eEspread
        self.energies = (eMin, eMax)
        self.eMin = eMin
        self.eMax = eMax
        self.xPrimeMax = xPrimeMax * 1e-3 if xPrimeMax else None
        self.zPrimeMax = zPrimeMax * 1e-3 if zPrimeMax else None
        self.betaX = betaX
        self.betaZ = betaZ
        self.eN = eN + 1
        self.nx = 2*nx + 1
        self.nz = 2*nz + 1
        self.distE = distE
        self.xs = np.linspace(-xPrimeMax, xPrimeMax, self.nx)
        self.zs = np.linspace(-zPrimeMax, zPrimeMax, self.nz)
        self.Es = np.linspace(eMin, eMax, self.eN)
        self.mode = 1
        self.uniformRayDensity = uniformRayDensity
        self.filamentBeam = filamentBeam

        if (self.dx is None) and (self.betaX is not None):
            self.dx = np.sqrt(self.eEpsilonX * self.betaX * 0.001)
        elif (self.dx is None) and (self.betaX is None):
            print "Set either eSigmaX or betaX!"
        if (self.dz is None) and (self.betaZ is not None):
            self.dz = np.sqrt(self.eEpsilonZ * self.betaZ * 0.001)
        elif (self.dz is None) and (self.betaZ is None):
            print "Set either eSigmaZ or betaZ!"

        dxprime, dzprime = None, None
        if dxprime:
            self.dxprime = dxprime
        else:
            self.dxprime = 1e-6 * self.eEpsilonX /\
                self.dx if self.dx > 0 else 0.  # [rad]
        if dzprime:
            self.dzprime = dzprime
        else:
            self.dzprime = 1e-6 * self.eEpsilonZ /\
                self.dz if self.dx > 0 else 0.  # [rad]

        self.gamma2 = self.gamma**2
        """" K2B: Conversion of Deflection parameter to magnetic field [T]
                        for the period in [mm]"""
        #self.c_E = 0.0075 * HPLANCK * C * self.gamma**3 / PI / EV2ERG
        #self.c_3 = 40. * PI * E0 * EV2ERG * self.I0 /\
        #    (np.sqrt(3) * HPLANCK * HPLANCK * C * self.gamma2) * \
        #    200. * EV2ERG / (np.sqrt(3) * HPLANCK * C * self.gamma2)

        mE = self.eN
        mTheta = self.nx
        mPsi = self.nz

        if self.isMPW:  # xPrimeMaxAutoReduce
            xPrimeMaxTmp = self.K / self.gamma
            if self.xPrimeMax > xPrimeMaxTmp:
                print "Reducing xPrimeMax from {0} down to {1} mrad".format(
                    self.xPrimeMax * 1e3, xPrimeMaxTmp * 1e3)
                self.xPrimeMax = xPrimeMaxTmp

        self.Theta_min = float(-self.xPrimeMax)
        self.Psi_min = float(-self.zPrimeMax)
        self.Theta_max = float(self.xPrimeMax)
        self.Psi_max = float(self.zPrimeMax)
        self.E_min = float(np.min(self.energies))
        self.E_max = float(np.max(self.energies))

        self.dE = (self.E_max-self.E_min) / float(mE-1)
        self.dTheta = (self.Theta_max-self.Theta_min) / float(mTheta-1)
        self.dPsi = (self.Psi_max-self.Psi_min) / float(mPsi-1)

        """Trying to find real maximum of the flux density"""
        E0fit = 0.5 * (self.E_max+self.E_min)

        precalc = True
        rMax = int(self.nrays)
        if precalc:
            rE = np.random.uniform(self.E_min, self.E_max, rMax)
            rTheta = np.random.uniform(0., self.Theta_max, rMax)
            rPsi = np.random.uniform(0., self.Psi_max, rMax)
            DistI = self.build_I_map(rE, rTheta, rPsi)[0]
            f_max = np.amax(DistI)
            a_max = np.argmax(DistI)
            NZ = np.ceil(np.max(rPsi[np.where(DistI > 0)[0]]) / self.dPsi) *\
                self.dPsi
            self.zPrimeMax = min(self.zPrimeMax, NZ)
            self.Psi_max = float(self.zPrimeMax)
            initial_x = [(rE[a_max]-E0fit) * 1e-5,
                         rTheta[a_max] * 1e3, rPsi[a_max] * 1e3]
        else:
            xE, xTheta, xPsi = np.mgrid[
                self.E_min:self.E_max + 0.5*self.dE:self.dE,
                self.Theta_min:self.Theta_max + 0.5*self.dTheta:self.dTheta,
                self.Psi_min:self.Psi_max + 0.5*self.dPsi:self.dPsi]
            DistI = self.build_I_map(xE, xTheta, xPsi)[0]
            f_max = np.amax(DistI)
            initial_x = [
                (self.E_min + 0.6 * mE * self.dE - E0fit) * 1e-5,
                (self.Theta_min + 0.6 * mTheta * self.dTheta) * 1e3,
                (self.Psi_min + 0.6 * self.dPsi * mPsi) * 1e3]

        bounds_x = [
            ((self.E_min - E0fit) * 1e-5, (self.E_max - E0fit) * 1e-5),
            (0, self.Theta_max * 1e3),
            (0, self.Psi_max * 1e3)]

        def int_fun(x):
            return -1. * (self.build_I_map(x[0] * 1e5 + E0fit,
                                           x[1] * 1e-3,
                                           x[2] * 1e-3)[0]) / f_max
        res = optimize.fmin_slsqp(int_fun, initial_x,
                                  bounds=bounds_x,
                                  acc=1e-12,
                                  iter=1000,
                                  epsilon=1.e-8,
                                  full_output=1,
                                  iprint=0)
        self.Imax = max(-1 * int_fun(res[0]) * f_max, f_max)

        if self.filamentBeam:
            self.nrepmax = np.floor(rMax / len(np.where(
                self.Imax * np.random.rand(rMax) < DistI)[0]))

        """Preparing to calculate the total flux integral"""
        self.xzE = 4 * (self.E_max-self.E_min) * self.Theta_max * self.Psi_max
        self.fluxConst = self.Imax * self.xzE

    def prefix_save_name(self):
        return '3-BM-xrt'

    def build_I_map(self, dde, ddtheta, ddpsi):
        np.seterr(invalid='ignore')
        np.seterr(divide='ignore')
        gamma = self.gamma
        if self.eEspread > 0:
            if np.array(dde).shape:
                if dde.shape[0] > 1:
                    gamma += np.random.normal(0, gamma*self.eEspread,
                                              dde.shape)
            gamma2 = gamma**2
        else:
            gamma2 = self.gamma2

        w_cr = 1.5 * gamma2 * self.B * SIe0 / SIme
        if self.isMPW:
            w_cr *= np.sin(np.arccos(ddtheta * gamma / self.K))
        w_cr = np.where(np.isfinite(w_cr), w_cr, 0.)

        gammapsi = gamma * ddpsi
        gamma2psi2p1 = gammapsi**2 + 1
        eta = 0.5 * dde * E2W / w_cr * gamma2psi2p1**1.5

        ampSP = -0.5j * SQ3 / PI * gamma * dde * E2W / w_cr * gamma2psi2p1
        ampS = ampSP * special.kv(2./3., eta)
        ampP = 1j * gammapsi * ampSP * special.kv(1./3., eta) /\
            np.sqrt(gamma2psi2p1)

        ampS = np.where(np.isfinite(ampS), ampS, 0.)
        ampP = np.where(np.isfinite(ampP), ampP, 0.)

        bwFact = 0.001 if self.distE == 'BW' else 1./dde
        Amp2Flux = FINE_STR * bwFact * self.I0 / SIe0 * 2 * self.Np

        np.seterr(invalid='warn')
        np.seterr(divide='warn')

        return (Amp2Flux * (np.abs(ampS)**2 + np.abs(ampP)**2),
                np.sqrt(Amp2Flux) * ampS,
                np.sqrt(Amp2Flux) * ampP)

    def intensities_on_mesh(self):
        xE, xTheta, xPsi = np.mgrid[
            self.E_min:self.E_max + 0.5*self.dE:self.dE,
            self.Theta_min:self.Theta_max + 0.5*self.dTheta:self.dTheta,
            self.Psi_min:self.Psi_max + 0.5*self.dPsi:self.dPsi]
        self.Itotal, ampS, ampP =\
            self.build_I_map(xE, xTheta, xPsi)
        self.Is = (ampS * np.conj(ampS)).real
        self.Ip = (ampP * np.conj(ampP)).real
        self.Isp = ampS * np.conj(ampP)
        s0 = self.Is + self.Ip
        Pol1 = np.where(s0, (self.Is - self.Ip) / s0, s0)
        Pol3 = np.where(s0, 2. * self.Isp / s0, s0)

        return (self.Itotal*1e-6, Pol1, self.Is*0., Pol3)

    def shine(self, toGlobal=True, withAmplitudes=False, accuBeam=None):
        """Returns the source beam. If *toGlobal* is True, the output is in
        the global system. If *withAmplitudes* is True, the resulted beam
        contains arrays Es and Ep with the *s* and *p* components of the
        electric field.
        """
        if self.uniformRayDensity:
            withAmplitudes = True

        bo = None
        length = 0
        seeded = long(0)
        seededI = 0.
        np.seterr(invalid='warn')
        np.seterr(divide='warn')
        mcRays = self.nrays * 1.2 if not self.uniformRayDensity else self.nrays
        if self.filamentBeam:
            if accuBeam is None:
                rE = np.random.random_sample() *\
                    float(self.E_max - self.E_min) + self.E_min
                if self.isMPW:
                    sigma_r2 = 2 * (HPLANCK*C/EV2ERG/rE*10*self.L0*self.Np) /\
                        (2 * np.pi)**2
                    sourceSIGMAx = (self.dx**2 + sigma_r2)**0.5
                    sourceSIGMAz = (self.dz**2 + sigma_r2)**0.5
                    rTheta0 = np.random.random_sample() *\
                        (self.Theta_max - self.Theta_min) + self.Theta_min
                    ryNp = 0.5 * self.L0 *\
                        (np.arccos(rTheta0 * self.gamma / self.K) / PI) +\
                        0.5 * self.L0 *\
                        np.random.random_integers(0, int(2*self.Np - 1))
                    rY = ryNp - 0.5*self.L0*self.Np
                    if (ryNp - 0.25*self.L0 <= 0):
                        rY += self.L0*self.Np
                    rX = self.X0 * np.sin(PI2 * rY / self.L0) +\
                        sourceSIGMAx * np.random.standard_normal()
                    rY -= 0.25 * self.L0
                    rZ = sourceSIGMAz * np.random.standard_normal()
                else:
                    rZ = self.dz * np.random.standard_normal()
                    rTheta0 = np.random.random_sample() *\
                        (self.Theta_max - self.Theta_min) + self.Theta_min
                    R1 = self.dx * np.random.standard_normal() +\
                        self.ro * 1000.
                    rX = -R1 * np.cos(rTheta0) + self.ro*1000.
                    rY = R1 * np.sin(rTheta0)
                dtheta = self.dxprime * np.random.standard_normal()
                dpsi = self.dzprime * np.random.standard_normal()
            else:
                rE = accuBeam.E[0]
                rX = accuBeam.x[0]
                rY = accuBeam.y[0]
                rZ = accuBeam.z[0]
                dtheta = accuBeam.filamentDtheta
                dpsi = accuBeam.filamentDpsi

        nrep = 0
        rep_condition = True
#        while length < self.nrays:
        while rep_condition:
            """Preparing 4 columns of random numbers
            0: Energy
            1: Theta / horizontal
            2: Psi / vertical
            3: Monte-Carlo discriminator"""
            rnd_r = np.random.rand(mcRays, 4)
            seeded += mcRays
            if self.filamentBeam:
                rThetaMin = np.max(self.Theta_min, rTheta0 - 1 / self.gamma)
                rThetaMax = np.min(self.Theta_max, rTheta0 + 1 / self.gamma)
                rTheta = (rnd_r[:, 1]) * (rThetaMax - rThetaMin) +\
                    rThetaMin
                if False:  # self.mono:
                    rE = np.ones(mcRays) * 0.5 *\
                        float(self.E_max - self.E_min) + self.E_min
                else:
                    rE *= np.ones(mcRays)

            else:
                rE = rnd_r[:, 0] * float(self.E_max - self.E_min) +\
                    self.E_min
                rTheta = (rnd_r[:, 1]) * (self.Theta_max - self.Theta_min) +\
                    self.Theta_min
            rPsi = rnd_r[:, 2] * (self.Psi_max - self.Psi_min) +\
                self.Psi_min
            Intensity, mJss, mJpp = self.build_I_map(rE, rTheta, rPsi)

            if self.uniformRayDensity:
                seededI += self.nrays * self.xzE
            else:
                seededI += Intensity.sum() * self.xzE

            if self.uniformRayDensity:
                I_pass = slice(None)
                npassed = mcRays
            else:
                I_pass =\
                    np.where(self.Imax * rnd_r[:, 3] < Intensity)[0]
                npassed = len(I_pass)
            if npassed == 0:
                print 'No good rays in this seed!', length, 'of',\
                    self.nrays, 'rays in total so far...'
                continue

            bot = Beam(npassed, withAmplitudes=withAmplitudes)
            bot.state[:] = 1  # good

            bot.E[:] = rE[I_pass]

            Theta0 = rTheta[I_pass]
            Psi0 = rPsi[I_pass]

            if not self.filamentBeam:
                if self.dxprime > 0:
                    dtheta = np.random.normal(0, self.dxprime, npassed)
                else:
                    dtheta = 0

                if self.dzprime > 0:
                    dpsi = np.random.normal(0, self.dzprime, npassed)
                else:
                    dpsi = 0

            bot.a[:] = -np.tan(Theta0 + dtheta)
            bot.c[:] = -np.tan(Psi0 + dpsi)

            intensS = (mJss[I_pass] * np.conj(mJss[I_pass])).real
            intensP = (mJpp[I_pass] * np.conj(mJpp[I_pass])).real
            if self.uniformRayDensity:
                sSP = 1.
            else:
                sSP = intensS + intensP
            # as by Walker and by Ellaume; SPECTRA's value is two times
            # smaller:

            if self.isMPW:
                sigma_r2 = 2 * (HPLANCK*C/EV2ERG/bot.E*10 *
                                self.L0*self.Np) / (2 * np.pi)**2
                bot.sourceSIGMAx = np.sqrt(self.dx**2 + sigma_r2)
                bot.sourceSIGMAz = np.sqrt(self.dz**2 + sigma_r2)
                if self.filamentBeam:
                    bot.z[:] = rZ
                    bot.x[:] = rX
                    bot.y[:] = rY
                else:
                    yNp = 0.5 * self.L0 *\
                        (np.arccos(Theta0 * self.gamma / self.K) / PI) +\
                        0.5 * self.L0 *\
                        np.random.random_integers(0,
                                                  int(2*self.Np - 1),
                                                  npassed)
                    bot.y[:] = np.where(
                        yNp - 0.25*self.L0 > 0,
                        yNp, self.L0*self.Np + yNp) - 0.5*self.L0*self.Np
                    bot.x[:] = self.X0 * np.sin(PI2 * bot.y / self.L0) +\
                        np.random.normal(0., bot.sourceSIGMAx, npassed)
                    bot.y[:] -= 0.25 * self.L0
                    bot.z[:] = np.random.normal(0., bot.sourceSIGMAz, npassed)
                bot.Jsp[:] = np.zeros(npassed)
            else:
                if self.filamentBeam:
                    bot.z[:] = rZ
                    bot.x[:] = rX
                    bot.y[:] = rY
                else:
                    bot.z[:] = np.random.normal(0., self.dz, npassed)
                    R1 = np.random.normal(self.ro * 1000., self.dx, npassed)
                    bot.x[:] = -R1 * np.cos(Theta0) + self.ro*1000.
                    bot.y[:] = R1 * np.sin(Theta0)

                bot.Jsp[:] = np.array(
                    np.where(sSP,
                             mJss[I_pass] * np.conj(mJpp[I_pass]) / sSP,
                             sSP), dtype=complex)

            bot.Jss[:] = np.where(sSP, intensS / sSP, sSP)
            bot.Jpp[:] = np.where(sSP, intensP / sSP, sSP)

            if withAmplitudes:
                bot.Es[:] = mJss[I_pass]
                bot.Ep[:] = mJpp[I_pass]

            if bo is None:
                bo = bot
            else:
                bo.concatenate(bot)
            length = len(bo.a)
            if _DEBUG > 20:
                print "%i rays of %i" % (length, self.nrays)
            if self.filamentBeam:
                nrep += 1
                rep_condition = nrep < self.nrepmax
            else:
                rep_condition = length < self.nrays
            if self.uniformRayDensity:
                rep_condition = False

        if length >= self.nrays:
            bo.accepted = length * self.fluxConst
            bo.acceptedE = bo.E.sum() * self.fluxConst * raycing.elec_e
            bo.seeded = seeded
            bo.seededI = seededI
        if length > self.nrays and not self.filamentBeam:
            bo.filter_by_index(slice(0, self.nrays))
        if self.filamentBeam:
            bo.filamentDtheta = dtheta
            bo.filamentDpsi = dpsi
        norm = np.sqrt(bo.a**2 + 1.0 + bo.c**2)
        bo.a /= norm
        bo.b /= norm
        bo.c /= norm
        if toGlobal:  # in global coordinate system:
            raycing.virgin_local_to_global(self.bl, bo, self.center)

        return bo


class Wiggler(BendingMagnet):
    def __init__(self, *args, **kwargs):
        """Parameters are the same as in BendingMagnet except *B0* and *rho*
        which are not required and additionally:
        *K*, *period* [mm], *n* (number of periods)."""
        self.isMPW = True
        self.Ee = kwargs['eE']
        self.gamma = self.Ee * 1e9 * EV2ERG / (M0 * C**2)
        self.K = kwargs.pop('K')
        self.L0 = kwargs.pop('period')
        self.Np = kwargs.pop('n')
        name = kwargs.pop('name', 'wiggler')
        kwargs['name'] = name
        self.B = K2B * self.K / self.L0
        self.ro = M0 * C**2 * self.gamma / self.B / E0 / 1e6
        super(Wiggler, self).__init__(*args, **kwargs)
        self.X0 = 0.5 * self.K * self.L0 / self.gamma / PI

    def prefix_save_name(self):
        return '2-Wiggler-xrt'


class Undulator(object):
    def __init__(self, bl, name='und', center=(0, 0, 0), nrays=raycing.nrays,
                 eE=6.0, eI=0.1, eEspread=0., eSigmaX=None, eSigmaZ=None,
                 eEpsilonX=1., eEpsilonZ=0.01, betaX=20., betaZ=5.,
                 period=50, n=50, K=10., Kx=0, Ky=0., phaseDeg=0,
                 taper=None, R0=None, targetE=None, filamentBeam=False,
                 eMin=5000., eMax=15000., eN=51, distE='eV',
                 xPrimeMax=0.5, zPrimeMax=0.5, nx=25, nz=25,
                 xPrimeMaxAutoReduce=True, zPrimeMaxAutoReduce=True,
                 gp=1e-6, gIntervals=1,
                 targetOpenCL=None, uniformRayDensity=False,
                 precisionOpenCL='float32'):
        """
        *bl*: instance of :class:`~xrt.backends.raycing.BeamLine`

        *name*: str

        *center*: tuple of 3 floats
            3D point in global system.
        *nrays*: int
            The number of rays sampled in one iteration.
        *eE*, *eI*:
            beam energy (GeV) and current (A)
        *eEspread*:
            energy spread relative to the beam energy, rms.
        *eSigmaX*, *eSigmaZ*: float
            rms horizontal and vertical electron beam sizes (µm).
            Alternatively, betatron functions can be specified instead of the
            electron beam sizes.
        *eEpsilonX*, *eEpsilonZ*:
            horizontal and vertical electron beam emittance (nm rad).
        *betaX*, *betaZ*:
            betatron function (m). Alternatively, beam size can be specified.
        *period*, *n*:
            magnetic period (mm) length and number of periods.
        *K*, *Kx*, *Ky*:
            deflection parameter for the vertical field or for an elliptical
            undulator.
        *phaseDeg*:
            phase difference between horizontal and vertical magnetic arrays.
            Used in the elliptical case where it should be equal to 90 or -90.
        *taper*: tuple(dgap(mm), gap(mm))
            linear variation in undulator gap, maximum minus minimum and the
            central gap value. None if tapering is not used. Tapering should be
            used only with pyopencl.
        *R0*: distance center-to-screen for the near field calculations (mm).
            If None, the far field approximation (i.e. "usual" calculations) is
            used. Near field calculations should be used only with pyopencl.
            Here, a GPU can be much faster than a CPU.
        *targetE*: a tuple (Energy, harmonic) for automatic calculation of the
            deflection parameter.
        *eMin*, *eMax*:
            minimum and maximum photon energy (eV).
        *eN*: int
            number of photon energy intervals, used only in the test suit, not
            required in ray tracing.
        *distE*: 'eV' or 'BW'
            The resulted flux density is per 1 eV or 0.1% bandwidth. For ray
            tracing 'eV' is used.
        *xPrimeMax*, *zPrimeMax*:
            horizontal and vertical acceptance (mrad)

        .. note::

            The Monte Carlo sampling of the rays having their density
            proportional to the beam intensity can be extremely inefficient
            for sharply peaked distributions, like the undulator angular
            density distribution. It is therefore very important to restrict
            the sampled angular acceptance down to very small angles. Use this
            source only with reasonably small *xPrimeMax* and *zPrimeMax*!

        *nx*, *nz*:
            number of intervals in the horizontal and vertical directions,
            used only in the test suit, not required in ray tracing.
        *xPrimeMaxAutoReduce*, *zPrimeMaxAutoReduce*: bool
            whether to reduce too large angular ranges down to the feasible
            values in order to improve efficiency. It is highly recommended to
            keep them True.
        *gp*:
            defines the precision of the Gauss integration.
        *gIntervals*: int
            integral of motion is divided by gIntervals to reduce the order of
            Gauss-Legendre quadrature. Default value of 1 is usually enough for
            a conventional undulator. For extreme cases (wigglers, near field,
            wide angles) this value can be set to the order of few hundreds to
            achieve the convergence of the integral. Large values can
            significantly increase the calculation time and RAM consumption
            especially if OpenCL is not used.
        *targetOpenCL*:
            If pyopencl is used: a tuple (iPlatform, iDevice) of indices in the
            lists cl.get_platforms() and platform.get_devices(), see the
            section :ref:`calculations_on_GPU`. None, if pyopencl is not
            wanted. Ignored if pyopencl is not installed.

            .. warning::
                A good graphics card is highly recommended! Special cases as
                wigglers by the undulator code, near field, wide angles and
                tapering are hardly doable on CPU.

        *precisionOpenCL*: 'float32' or 'float64', only for GPU.
            Single precision (float32) should be enough. So far, we do not see
            any example where double precision is required. The calculations
            with doube precision are much slower. Double precision may be
            unavailable on your system.
        *filamentBeam*: if True the source generates coherent monochromatic
            wavefronts. Required for the wave propagation calculations.
        """
        self.bl = bl
        bl.sources.append(self)
        self.ordinalNum = len(bl.sources)
        self.name = name
        self.center = center  # 3D point in global system
        self.nrays = nrays
        self.gp = gp
        self.dx = eSigmaX * 1e-3 if eSigmaX else None
        self.dz = eSigmaZ * 1e-3 if eSigmaZ else None
        self.eEpsilonX = eEpsilonX
        self.eEpsilonZ = eEpsilonZ
        self.Ee = float(eE)
        self.eEspread = eEspread
        self.I0 = float(eI)
        self.energies = (float(eMin), float(eMax))
        self.eMin = float(eMin)
        self.eMax = float(eMax)
        self.xPrimeMax = xPrimeMax * 1e-3  # if xPrimeMax else None
        self.zPrimeMax = zPrimeMax * 1e-3  # if zPrimeMax else None
        self.betaX = betaX
        self.betaZ = betaZ
        self.eN = eN + 1
        self.nx = 2*nx + 1
        self.nz = 2*nz + 1
        self.distE = distE
        self.uniformRayDensity = uniformRayDensity
        self.filamentBeam = filamentBeam
        self.gIntervals = gIntervals
        self.L0 = period
        self.R0 = R0 if R0 is None else R0 + self.L0*0.25

        if (self.R0 is not None):
            precisionOpenCL = 'float64'
        if (targetOpenCL is not None) and not isOpenCL:
            print "pyopencl is not available!"
        if (targetOpenCL is not None) and isOpenCL:
            iPlatform, iDevice = targetOpenCL
            platform = cl.get_platforms()[iPlatform]
            device = platform.get_devices()[iDevice]
            print('    Device - Name:  ' + device.name)
            self.cl_ctx = cl.Context(devices=[device])
#            self.cl_ctx = cl.create_some_context()
            self.cl_queue = cl.CommandQueue(self.cl_ctx)
            cl_file = os.path.join(os.path.dirname(__file__), r'undulator.cl')
            with open(cl_file, 'r') as f:
                kernelsource = f.read()
            if precisionOpenCL == 'float64':
                self.cl_precisionF = np.float64
                self.cl_precisionC = np.complex128
                kernelsource = kernelsource.replace('float', 'double')
            else:
                self.cl_precisionF = np.float32
                self.cl_precisionC = np.complex64
            self.cl_program = cl.Program(self.cl_ctx, kernelsource).build()
            self.cl_mf = cl.mem_flags
        else:
            self.cl_ctx = None
        self.xs = np.linspace(-xPrimeMax, xPrimeMax, self.nx)
        self.zs = np.linspace(-zPrimeMax, zPrimeMax, self.nz)
        self.Es = np.linspace(eMin, eMax, self.eN)
        self.mode = 1

        if (self.dx is None) and (self.betaX is not None):
            self.dx = np.sqrt(self.eEpsilonX*self.betaX*0.001)
        elif (self.dx is None) and (self.betaX is None):
            print "Set either dx or mean_betaX!"
        if (self.dz is None) and (self.betaZ is not None):
            self.dz = np.sqrt(self.eEpsilonZ*self.betaZ*0.001)
        elif (self.dz is None) and (self.betaZ is None):
            print "Set either dz or mean_betaZ!"
        dxprime, dzprime = None, None
        if dxprime:
            self.dxprime = dxprime
        else:
            self.dxprime = 1e-6 * self.eEpsilonX / self.dx if self.dx > 0\
                else 0.  # [rad]
        if dzprime:
            self.dzprime = dzprime
        else:
            self.dzprime = 1e-6 * self.eEpsilonZ / self.dz if self.dz > 0\
                else 0.  # [rad]
        if _DEBUG:
            print 'dxprime = {0} rad'.format(self.dxprime)
            print 'dzprime = {0} rad'.format(self.dzprime)
        self.gamma = self.Ee * 1e9 * EV2ERG / (M0 * C**2)
        self.gamma2 = self.gamma**2

        if targetE is not None:
            K = np.sqrt(targetE[1] * 8 * PI * C * 10 * self.gamma2 /
                        period / targetE[0] / E2W - 2)
            if _DEBUG:
                print "K =", K

        self.Kx = Kx
        self.Ky = Ky
        self.K = K
        self.phase = np.radians(phaseDeg)

        self.Np = n

        if taper is not None:
            self.taper = taper[0] / self.Np / self.L0 / taper[1]
            self.gap = taper[1]
        else:
            self.taper = None
        if self.Kx == 0 and self.Ky == 0:
            self.Ky = self.K

        if xPrimeMaxAutoReduce:
            xPrimeMaxTmp = self.Ky / self.gamma
            if self.xPrimeMax > xPrimeMaxTmp:
                print "Reducing xPrimeMax from {0} down to {1} mrad".format(
                    self.xPrimeMax * 1e3, xPrimeMaxTmp * 1e3)
                self.xPrimeMax = xPrimeMaxTmp
        if zPrimeMaxAutoReduce:
            K0 = self.Kx if self.Kx > 0 else 1.
            zPrimeMaxTmp = K0 / self.gamma
            if self.zPrimeMax > zPrimeMaxTmp:
                print "Reducing zPrimeMax from {0} down to {1} mrad".format(
                    self.zPrimeMax * 1e3, zPrimeMaxTmp * 1e3)
                self.zPrimeMax = zPrimeMaxTmp

        self.reset()

    def reset(self):
        self.wu = PI * (0.01 * C) / self.L0 / 1e-3 / self.gamma2 * \
            (2*self.gamma2 - 1 - 0.5*self.Kx**2 - 0.5*self.Ky**2) / E2W
        #wnu = 2 * PI * (0.01 * C) / self.L0 / 1e-3 / E2W
        if _DEBUG:
            print "E1: ", 2 * self.wu * self.gamma2 / (1 + 0.5 * self.Ky**2)
            print "E3: ", 6 * self.wu * self.gamma2 / (1 + 0.5 * self.Ky**2)
            print "B0:", self.Ky / 0.09336 / self.L0
            if self.taper is not None:
                print "dB/dx/B:", -PI * self.gap * self.taper / self.L0 * 1e3
        mE = self.eN
        mTheta = self.nx
        mPsi = self.nz

        if not self.xPrimeMax:
            print "No Theta range specified, using default 1 mrad"
            self.xPrimeMax = 1.

        self.Theta_min = float(-self.xPrimeMax)
        self.Theta_max = float(self.xPrimeMax)
        self.Psi_min = float(-self.zPrimeMax)
        self.Psi_max = float(self.zPrimeMax)
        self.E_min = float(np.min(self.energies))
        self.E_max = float(np.max(self.energies))

        self.dE = (self.E_max - self.E_min) / float(mE - 1)
        self.dTheta = (self.Theta_max - self.Theta_min) / float(mTheta - 1)
        self.dPsi = (self.Psi_max - self.Psi_min) / float(mPsi - 1)

        """Adjusting the number of points for Gauss integration"""
#        self.gp = 1
        gau_int_error = self.gp * 10.
        ii = 2
        self.gau = int(ii*2 + 1)
        #self.gau = int(2**ii + 1)
        while gau_int_error >= self.gp:
            ii += self.gIntervals
            self.gau = int(ii*2)
            I1 = self.build_I_map(self.E_max, self.Theta_max, self.Psi_max)[0]
            self.gau = int(ii*2 + 1)
            I2 = self.build_I_map(self.E_max, self.Theta_max, self.Psi_max)[0]
            gau_int_error = np.abs((I2 - I1)/I1)
            if _DEBUG:
                print "G =", self.gau, gau_int_error, np.abs(I2)
#            I0 = self.build_I_map(36762.74, -0.000101588, 0.0001495)[0]
#            print I0
#        raise

        if self.filamentBeam:
            rMax = self.nrays
            rE = np.random.uniform(self.E_min, self.E_max, rMax)
            rTheta = np.random.uniform(self.Theta_min, self.Theta_max, rMax)
            rPsi = np.random.uniform(self.Psi_min, self.Psi_max, rMax)
            tmpEspread = self.eEspread
            self.eEspread = 0
            DistI = self.build_I_map(rE, rTheta, rPsi)[0]
            self.Imax = np.max(DistI) * 1.2
            self.nrepmax = np.floor(rMax / len(np.where(
                self.Imax * np.random.rand(rMax) < DistI)[0]))
            self.eEspread = tmpEspread
        else:
            self.Imax = 0.
        """Preparing to calculate the total flux integral"""
#        if _DEBUG:
#            print self.Imax
        self.xzE = (self.E_max - self.E_min) *\
            (self.Theta_max - self.Theta_min) *\
            (self.Psi_max - self.Psi_min)
        self.fluxConst = self.Imax * self.xzE

    def prefix_save_name(self):
        if self.Kx > 0:
            return '4-elu-xrt'
        else:
            return '1-und-xrt'

    def intensities_on_mesh(self):
        xE, xTheta, xPsi = np.mgrid[
            self.E_min:self.E_max + 0.5*self.dE:self.dE,
            self.Theta_min:self.Theta_max + 0.5*self.dTheta:self.dTheta,
            self.Psi_min:self.Psi_max + self.dPsi:self.dPsi]
# linear arrays for OpenCL:
        sh = xE.shape
        res = self.build_I_map(xE.flatten(), xTheta.flatten(), xPsi.flatten())
# restore the shape:
        self.Itotal = res[0].reshape(sh)
        self.Is = res[1].reshape(sh)
        self.Ip = res[2].reshape(sh)

        s0 = (self.Is*np.conj(self.Is) + self.Ip*np.conj(self.Ip)).real
        s1 = (self.Is*np.conj(self.Is) - self.Ip*np.conj(self.Ip)).real
        s2 = 2. * np.real(self.Is * np.conj(self.Ip))
        s3 = -2. * np.imag(self.Is * np.conj(self.Ip))

        return (self.Itotal * 1e-6,
                np.where(s0, s1 / s0, s0),
                np.where(s0, s2 / s0, s0),
                np.where(s0, s3 / s0, s0))

    def _sp(self, dim, x, ww1, w, wu, gamma, ddphi, ddpsi):
        gS = gamma
        if dim == 0:
            ww1S = ww1
            wS, wuS = w, wu
            ddphiS = ddphi
            ddpsiS = ddpsi
        elif dim == 1:
            ww1S = ww1[:, np.newaxis]
            wS = w[:, np.newaxis]
            wuS = wu[:, np.newaxis]
            ddphiS = ddphi[:, np.newaxis]
            ddpsiS = ddpsi[:, np.newaxis]
            if self.eEspread > 0:
                gS = gamma[:, np.newaxis]
        elif dim == 3:
            ww1S = ww1[:, :, :, np.newaxis]
            wS, wuS = w[:, :, :, np.newaxis], wu[:, :, :, np.newaxis]
            ddphiS = ddphi[:, :, :, np.newaxis]
            ddpsiS = ddpsi[:, :, :, np.newaxis]
            if self.eEspread > 0:
                gS = gamma[:, :, :, np.newaxis]
        taperC = 1
        if self.taper is not None:
            print "tapering"
            alphaS = self.taper * C * 10 / E2W
            taperC = 1 - alphaS * x / wuS
            sinx = np.sin(x)
            sin2x = np.sin(2 * x)
            cosx = np.cos(x)
            ucos = ww1S * x +\
                wS / gS / wuS *\
                (-self.Ky * ddphiS * (sinx + alphaS / wuS *
                                      (1 - cosx - x * sinx)) +
                 self.Kx * ddpsiS * np.sin(x + self.phase) +
                 0.125 / gS *
                 (self.Kx**2 * np.sin(2 * (x + self.phase)) +
                 self.Ky**2 * (sin2x - 2 * alphaS / wuS *
                               (x**2 + cosx**2 + x * sin2x))))
        elif self.R0 is not None:
            betam = 1 - (1 + 0.5 * self.Kx**2 + 0.5 * self.Ky**2) / 2. / gS**2
            WR0 = self.R0 / 10 / C * E2W
            ddphiS = -ddphiS
            drx = WR0 * np.tan(ddphiS) - self.Ky / wuS / gS * np.sin(x)
            dry = WR0 * np.tan(ddpsiS) + self.Kx / wuS / gS * np.sin(
                x + self.phase)
            drz = WR0 * np.cos(np.sqrt(ddphiS**2+ddpsiS**2)) -\
                betam * x / wuS + 0.125 / wuS / gS**2 *\
                (self.Ky**2 * np.sin(2 * x) +
                 self.Kx**2 * np.sin(2 * (x + self.phase)))
            ucos = wS * (x / wuS + np.sqrt(drx**2 + dry**2 + drz**2))
        else:
            ucos = ww1S * x + wS / gS / wuS *\
                (-self.Ky * ddphiS * np.sin(x) +
                 self.Kx * ddpsiS * np.sin(x + self.phase) +
                 0.125 / gS * (self.Ky**2 * np.sin(2. * x) +
                 self.Kx**2 * np.sin(2. * (x + self.phase))))
#            print ucos
        eucos = np.exp(1j * ucos)
        return ((ddphiS - taperC * self.Ky / gS * np.cos(x)) * eucos,
                (ddpsiS + self.Kx / gS * np.cos(x + self.phase)) * eucos)

    def build_I_map(self, w, ddtheta, ddpsi):
        if self.cl_ctx is None:
            return self._build_I_map_conv(w, ddtheta, ddpsi)
        else:
            return self._build_I_map_CL(w, ddtheta, ddpsi)

    def _build_I_map_conv(self, w, ddtheta, ddpsi):
#        np.seterr(invalid='ignore')
#        np.seterr(divide='ignore')
        gamma = self.gamma
        if self.eEspread > 0:
            if np.array(w).shape:
                if w.shape[0] > 1:
                    if self.filamentBeam:
                        gamma += gamma * self.eEspread * np.ones_like(w) *\
                            np.random.standard_normal()
                    else:
                        gamma += np.random.normal(0,
                                                  gamma*self.eEspread,
                                                  w.shape)
            gamma2 = gamma**2
            wu = PI * C * 10 / self.L0 / gamma2 * \
                (2*gamma2 - 1 - 0.5*self.Kx**2 - 0.5*self.Ky**2) / E2W
        else:
            gamma = self.gamma
            gamma2 = self.gamma2
            wu = self.wu * np.ones_like(w)

        ww1 = w * ((1. + 0.5*self.Kx**2 + 0.5*self.Ky**2) +
                   gamma2 * (ddtheta**2 + ddpsi**2)) / (2. * gamma2 * wu)
        tg_n, ag_n = np.polynomial.legendre.leggauss(self.gau)

        if (self.taper is not None) or (self.R0 is not None):
            AB = w / PI2 / wu
            dstep = 2 * PI / float(self.gIntervals)
            dI = np.arange(0.5 * dstep - PI * self.Np, PI * self.Np, dstep)
        else:
            AB = w / PI2 / wu * np.sin(PI * self.Np * ww1) / np.sin(PI * ww1)
            dstep = 2 * PI / float(self.gIntervals)
            dI = np.arange(-PI + 0.5 * dstep, PI, dstep)

        tg = (dI[:, None] + 0.5*dstep*tg_n).flatten() + PI/2
        ag = (dI[:, None]*0 + ag_n).flatten()
        #Bsr = np.zeros_like(w, dtype='complex')
        #Bpr = np.zeros_like(w, dtype='complex')
        dim = len(np.array(w).shape)
        sp3res = self._sp(dim, tg, ww1, w, wu, gamma, ddtheta, ddpsi)
        Bsr = np.sum(ag * sp3res[0], axis=dim)
        Bpr = np.sum(ag * sp3res[1], axis=dim)

        bwFact = 0.001 if self.distE == 'BW' else 1./w
        Amp2Flux = FINE_STR * bwFact * self.I0 / SIe0

#        np.seterr(invalid='warn')
#        np.seterr(divide='warn')
        return (Amp2Flux * AB**2 * 0.25 * dstep**2 *
                (np.abs(Bsr)**2 + np.abs(Bpr)**2),
                np.sqrt(Amp2Flux) * AB * Bsr,
                np.sqrt(Amp2Flux) * AB * Bpr)

    def _build_I_map_CL(self, w, ddtheta, ddpsi):
#        np.seterr(invalid='ignore')
#        np.seterr(divide='ignore')
        gamma = self.gamma
        if self.eEspread > 0:
            if np.array(w).shape:
                if w.shape[0] > 1:
                    if self.filamentBeam:
                        gamma += gamma * self.eEspread * \
                            np.ones_like(w, dtype=self.cl_precisionF) *\
                            np.random.standard_normal()
                    else:
                        gamma += np.random.normal(0,
                                                  gamma*self.eEspread,
                                                  w.shape)
            gamma2 = gamma**2
            wu = PI * C * 10 / self.L0 / gamma2 * \
                (2*gamma2 - 1 - 0.5*self.Kx**2 - 0.5*self.Ky**2) / E2W
        else:
            gamma = self.gamma * np.ones_like(w, dtype=self.cl_precisionF)
            gamma2 = self.gamma2 * np.ones_like(w, dtype=self.cl_precisionF)
            wu = self.wu * np.ones_like(w, dtype=self.cl_precisionF)
        NRAYS = 1 if len(np.array(w).shape) == 0 else len(w)

        ww1 = w * ((1. + 0.5 * self.Kx**2 + 0.5 * self.Ky**2) +
                   gamma2 * (ddtheta * ddtheta + ddpsi * ddpsi)) /\
            (2. * gamma2 * wu)

        if self.R0 is not None:
            R0 = self.cl_precisionF(self.R0)
        taper = self.cl_precisionF(self.taper) if self.taper is not None else\
            self.cl_precisionF(0.)
        gamma = self.cl_precisionF(gamma)
        Kx = self.cl_precisionF(self.Kx)
        Ky = self.cl_precisionF(self.Ky)
        wu = self.cl_precisionF(wu)
        phase = self.cl_precisionF(self.phase)
        Np = np.int32(self.Np)

        tg_n, ag_n = np.polynomial.legendre.leggauss(self.gau)

        if (self.taper is not None) or (self.R0 is not None):
            ab = w / PI2 / wu
            dstep = 2 * PI / float(self.gIntervals)
            dI = np.arange(0.5 * dstep - PI * Np, PI * Np, dstep)
        else:
            ab = w / PI2 / wu * np.sin(PI * Np * ww1) / np.sin(PI * ww1)
            dstep = 2 * PI / float(self.gIntervals)
            dI = np.arange(-PI + 0.5*dstep, PI, dstep)

        ww1 = self.cl_precisionF(ww1)

        tg = self.cl_precisionF((dI[:, None]+0.5*dstep*tg_n).flatten()) + PI/2
        ag = self.cl_precisionF((dI[:, None]*0+ag_n).flatten())
        jend = np.int32(len(tg))
        Is_local = np.zeros(NRAYS, dtype=self.cl_precisionC)
        Ip_local = np.zeros(NRAYS, dtype=self.cl_precisionC)
        Itotal_local = np.zeros(NRAYS, dtype=self.cl_precisionF)

        rE = self.cl_precisionF(w)
        rTheta = self.cl_precisionF(ddtheta)
        rPsi = self.cl_precisionF(ddpsi)
#        time1 = time.time()

        gamma_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                              self.cl_mf.COPY_HOST_PTR, hostbuf=gamma)
        wu_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                           self.cl_mf.COPY_HOST_PTR, hostbuf=wu)
        w_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                          self.cl_mf.COPY_HOST_PTR, hostbuf=rE)
        ww1_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                            self.cl_mf.COPY_HOST_PTR, hostbuf=ww1)
        ddtheta_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                                self.cl_mf.COPY_HOST_PTR, hostbuf=rTheta)
        ddpsi_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                              self.cl_mf.COPY_HOST_PTR, hostbuf=rPsi)
        tg_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                           self.cl_mf.COPY_HOST_PTR, hostbuf=tg)
        ag_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_ONLY |
                           self.cl_mf.COPY_HOST_PTR, hostbuf=ag)
        Is_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_WRITE |
                           self.cl_mf.COPY_HOST_PTR, hostbuf=Is_local)
        Ip_buf = cl.Buffer(self.cl_ctx, self.cl_mf.READ_WRITE |
                           self.cl_mf.COPY_HOST_PTR, hostbuf=Ip_local)

        global_size = Itotal_local.shape
        local_size = None

        if self.taper is not None:
            self.cl_program.undulator_taper(self.cl_queue,
                                            global_size,
                                            local_size,
                                            taper, Kx, Ky, phase,
                                            jend, gamma_buf,
                                            wu_buf, w_buf, ww1_buf,
                                            ddtheta_buf, ddpsi_buf,
                                            tg_buf, ag_buf,
                                            Is_buf, Ip_buf).wait()
        elif self.R0 is not None:
            self.cl_program.undulator_nf(self.cl_queue,
                                         global_size,
                                         local_size,
                                         R0, Kx, Ky, phase,
                                         jend, gamma_buf,
                                         wu_buf, w_buf,
                                         ddtheta_buf, ddpsi_buf,
                                         tg_buf, ag_buf,
                                         Is_buf, Ip_buf).wait()
        else:
            self.cl_program.undulator(self.cl_queue,
                                      global_size,
                                      local_size,
                                      taper, Kx, Ky, phase,
                                      # Np, dstep,
                                      jend, gamma_buf,
                                      wu_buf, w_buf, ww1_buf,
                                      ddtheta_buf, ddpsi_buf,
                                      tg_buf, ag_buf,
                                      Is_buf, Ip_buf).wait()

        cl.enqueue_read_buffer(self.cl_queue,
                               Is_buf,
                               Is_local).wait()
        cl.enqueue_read_buffer(self.cl_queue,
                               Ip_buf,
                               Ip_local).wait()

        bwFact = 0.001 if self.distE == 'BW' else 1./w

#        np.seterr(invalid='warn')
#        np.seterr(divide='warn')
        Amp2Flux = FINE_STR * bwFact * self.I0 / SIe0

        return (Amp2Flux * ab**2 * 0.25 * dstep**2 *
                (np.abs(Is_local)**2 + np.abs(Ip_local)**2),
                np.sqrt(Amp2Flux) * Is_local * ab * 0.5 * dstep,
                np.sqrt(Amp2Flux) * Ip_local * ab * 0.5 * dstep)

#    def _reportNaN(self, x, strName):
#        nanSum = np.isnan(x).sum()
#        if nanSum > 0:
#            print "{0} NaN rays in {1}!".format(nanSum, strName)

    def shine(self, toGlobal=True, withAmplitudes=True, accuBeam=None,
              fixedEnergy=False):
        """Returns the source beam. If *toGlobal* is True, the output is in
        the global system. If *withAmplitudes* is True, the resulted beam
        contains arrays Es and Ep with the *s* and *p* components of the
        electric field.
        """
        if self.uniformRayDensity:
            withAmplitudes = True
        if not self.uniformRayDensity:
            if _DEBUG > 10:
                print "Rays generation"
        bo = None
        length = 0
        seeded = long(0)
        seededI = 0.
        np.seterr(invalid='warn')
        np.seterr(divide='warn')
        mcRays = self.nrays * 1.2 if not self.uniformRayDensity else self.nrays
        if self.filamentBeam:
            if accuBeam is None:
                if fixedEnergy:
                    rsE = fixedEnergy
                else:
                    rsE = np.random.random_sample() * \
                        float(self.E_max - self.E_min) + self.E_min
                sigma_r2 = 2 * (HPLANCK*C/EV2ERG/rsE*10 * self.L0*self.Np) /\
                    (2 * np.pi)**2
                sourceSIGMAx = (self.dx**2 + sigma_r2)**0.5
                sourceSIGMAz = (self.dz**2 + sigma_r2)**0.5
                rX = sourceSIGMAx * np.random.standard_normal()
                rZ = sourceSIGMAz * np.random.standard_normal()
                dtheta = self.dxprime * np.random.standard_normal()
                dpsi = self.dzprime * np.random.standard_normal()
            else:
                rsE = accuBeam.E[0]
                rX = accuBeam.x[0]
                rZ = accuBeam.z[0]
                dtheta = accuBeam.filamentDtheta
                dpsi = accuBeam.filamentDpsi
                seeded = accuBeam.seeded
                seededI = accuBeam.seededI
        nrep = 0

        rep_condition = True
#        while length < self.nrays:
        while rep_condition:
            """Preparing 4 columns of random numbers
            0: Energy
            1: Theta / horizontal
            2: Psi / vertical
            3: Monte-Carlo discriminator"""
            #print "rep", nrep+1
            rnd_gen = np.random.rand(mcRays, 4)
            seeded += mcRays
            # start_time = time.time()
            if self.filamentBeam:
                if False:  # self.mono:
                    rE = np.ones(mcRays) * 0.5 *\
                        float(self.E_max - self.E_min) + self.E_min
                else:
                    rE = rsE*np.ones(mcRays)
            else:
                rE = rnd_gen[:, 0] * float(self.E_max - self.E_min) +\
                    self.E_min
            rTheta = rnd_gen[:, 1] * float(self.Theta_max - self.Theta_min) +\
                self.Theta_min
            rPsi = rnd_gen[:, 2] * float(self.Psi_max - self.Psi_min) +\
                self.Psi_min
            Intensity, mJs, mJp = self.build_I_map(rE, rTheta, rPsi)

            if self.uniformRayDensity:
                seededI += mcRays * self.xzE
            else:
                seededI += Intensity.sum() * self.xzE
            # run_time = time.time() - start_time
            # print "Time elapsed:", run_time,"s"
            tmp_max = np.max(Intensity)
            if tmp_max > self.Imax:
                self.Imax = tmp_max
                self.fluxConst = self.Imax * self.xzE
                if _DEBUG:
                    imax = np.argmax(Intensity)
                    print self.Imax, imax, rE[imax], rTheta[imax], rPsi[imax]
            if self.uniformRayDensity:
                I_pass = slice(None)
                npassed = mcRays
            else:
                I_pass = np.where(self.Imax * rnd_gen[:, 3] < Intensity)[0]
                npassed = len(I_pass)
            if npassed == 0:
                print 'No good rays in this seed!', length, 'of',\
                    self.nrays, 'rays in total so far...'
                continue

            bot = Beam(npassed, withAmplitudes=withAmplitudes)
            bot.state[:] = 1  # good

            bot.E[:] = rE[I_pass]

            Theta0 = rTheta[I_pass]
            Psi0 = rPsi[I_pass]

            if not self.filamentBeam:
                if self.dxprime > 0:
                    dtheta = np.random.normal(0, self.dxprime, npassed)
                else:
                    dtheta = 0

                if self.dzprime > 0:
                    dpsi = np.random.normal(0, self.dzprime, npassed)
                else:
                    dpsi = 0

            bot.a[:] = -np.tan(Theta0 + dtheta)
            bot.c[:] = -np.tan(Psi0 + dpsi)

# as by Walker and by Ellaume; SPECTRA's value is two times smaller:
            sigma_r2 = 2 * (HPLANCK*C/EV2ERG/bot.E*10 * self.L0*self.Np)\
                / (2 * np.pi)**2
            bot.sourceSIGMAx = (self.dx**2 + sigma_r2)**0.5
            bot.sourceSIGMAz = (self.dz**2 + sigma_r2)**0.5

            if self.filamentBeam:
                bot.x[:] = rX
                bot.z[:] = rZ
            else:
                bot.x[:] = np.random.normal(0, bot.sourceSIGMAx, npassed)
                bot.z[:] = np.random.normal(0, bot.sourceSIGMAz, npassed)

            mJs = mJs[I_pass]
            mJp = mJp[I_pass]
            mJs2 = (mJs * np.conj(mJs)).real
            mJp2 = (mJp * np.conj(mJp)).real

            if self.uniformRayDensity:
                sSP = 1.
            else:
                sSP = mJs2 + mJp2

            bot.Jsp[:] = np.where(sSP, mJs * np.conj(mJp) / sSP, 0)
            bot.Jss[:] = np.where(sSP, mJs2 / sSP, 0)
            bot.Jpp[:] = np.where(sSP, mJp2 / sSP, 0)

            if withAmplitudes:
                bot.Es[:] = mJs
                bot.Ep[:] = mJp

            if bo is None:
                bo = bot
            else:
                bo.concatenate(bot)
            length = len(bo.a)
            if not self.uniformRayDensity:
                if _DEBUG > 10:
                    print "{0} rays of {1}".format(length, self.nrays)
            if self.filamentBeam:
                nrep += 1
                rep_condition = nrep < self.nrepmax
            else:
                rep_condition = length < self.nrays
            if self.uniformRayDensity:
                rep_condition = False

            bo.accepted = length * self.fluxConst
            bo.acceptedE = bo.E.sum() * self.fluxConst * raycing.elec_e
            #print "seeded", seeded
            #print "seededI", seededI
            #print "xzE", self.xzE
            bo.seeded = seeded
            bo.seededI = seededI

        if length > self.nrays and not self.filamentBeam:
            bo.filter_by_index(slice(0, self.nrays))
        if self.filamentBeam:
            bo.filamentDtheta = dtheta
            bo.filamentDpsi = dpsi
        norm = np.sqrt(bo.a**2 + 1 + bo.c**2)
        bo.a /= norm
        bo.b /= norm
        bo.c /= norm
        if toGlobal:  # in global coordinate system:
            raycing.virgin_local_to_global(self.bl, bo, self.center)

#        if _DEBUG:
#            self._reportNaN(bo.Jss, 'Jss')
#            self._reportNaN(bo.Jpp, 'Jpp')
#            self._reportNaN(bo.Jsp, 'Jsp')
#            self._reportNaN(bo.E, 'E')
#            self._reportNaN(bo.x, 'x')
#            self._reportNaN(bo.y, 'y')
#            self._reportNaN(bo.z, 'z')
#            self._reportNaN(bo.a, 'a')
#            self._reportNaN(bo.b, 'b')
#            self._reportNaN(bo.c, 'c')
        return bo
