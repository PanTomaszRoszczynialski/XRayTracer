# -*- coding: utf-8 -*-
r"""
.. _slitDiffraction:

Slit diffraction with undulator source
--------------------------------------

This example shows wave diffraction of an undulator beam on a square slit. The
calculations were done for (1) a monochromatic beam with zero electron beam
emittance and (2) a finite energy band beam with the real electron emittance of
Petra III ring. The resulted images are compared with experimentally measured
ones [Zozulya_Sprung]_.

Notice that vertical stripes are less pronounced in the complete calculations
because the horizontal emittance is much larger than the vertical one.

.. [Zozulya_Sprung] A. Zozulya and M. Sprung, measured at P10 beamline,
    DESY Photon Science (2014) unpublished.

+-----------+----------------+----------------+----------------+--------------+
| slit size | ray tracing,   | diffraction,   | diffraction,   | experimental |
|   (µm²)   | zero emittance | zero emittance | real emittance |              |
+===========+================+================+================+==============+
| 100×100   | |slit_r0_100|  | |slit_w0_100|  | to do          | |slit_e_100| |
+-----------+----------------+----------------+----------------+--------------+
| 150×150   | |slit_r0_150|  | |slit_w0_150|  | to do          | |slit_e_150| |
+-----------+----------------+----------------+----------------+--------------+
| 200×200   | |slit_r0_200|  | |slit_w0_200|  | |slit_wr_200|  | |slit_e_200| |
+-----------+----------------+----------------+----------------+--------------+
| 300×300   | |slit_r0_300|  | |slit_w0_300|  | |slit_wr_300|  | |slit_e_300| |
+-----------+----------------+----------------+----------------+--------------+

.. |slit_r0_100| image:: _images/slit_r0_100.*
   :scale: 50 %
.. |slit_r0_150| image:: _images/slit_r0_150.*
   :scale: 50 %
.. |slit_r0_200| image:: _images/slit_r0_200.*
   :scale: 50 %
.. |slit_r0_300| image:: _images/slit_r0_300.*
   :scale: 50 %

.. |slit_w0_100| image:: _images/slit_w0_100.*
   :scale: 50 %
.. |slit_w0_150| image:: _images/slit_w0_150.*
   :scale: 50 %
.. |slit_w0_200| image:: _images/slit_w0_200.*
   :scale: 50 %
.. |slit_w0_300| image:: _images/slit_w0_300.*
   :scale: 50 %

.. |slit_wr_100| image:: _images/slit_wr_100.*
   :scale: 50 %
.. |slit_wr_150| image:: _images/slit_wr_150.*
   :scale: 50 %
.. |slit_wr_200| image:: _images/slit_wr_200.*
   :scale: 50 %
.. |slit_wr_300| image:: _images/slit_wr_300.*
   :scale: 50 %

.. |slit_e_100| image:: _images/slit_e_100.*
   :scale: 50 %
.. |slit_e_150| image:: _images/slit_e_150.*
   :scale: 50 %
.. |slit_e_200| image:: _images/slit_e_200.*
   :scale: 50 %
.. |slit_e_300| image:: _images/slit_e_300.*
   :scale: 50 %
"""
__author__ = "Roman Chernikov", "Konstantin Klementiev"
__date__ = "10 Apr 2015"

import sys
# import numpy as np
sys.path.append(r"c:\Ray-tracing")
# sys.path.append(r"/media/sf_Ray-tracing")
# import time
#import matplotlib as mpl
#mpl.use('Agg')
import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.screens as rsc
import xrt.backends.raycing.apertures as ra
import xrt.backends.raycing.run as rr
import xrt.plotter as xrtp
import xrt.runner as xrtr
import numpy as np

R0 = 44000
mynrays = 5e6

slitDx = 0.2
slitDz = 0.2

SCRx = 1
SCRz = 1
dE = 0.5

#nrep = 160
nrep = 1000
tOCL = (0, 0)

E0 = 7900
eMinRays = E0 - dE
eMaxRays = E0 + dE

kwargs = dict(
    period=29., n=172,
    eE=6.08, eI=0.1,  # eEspread=0.001,
    #eEpsilonX=0., eEpsilonZ=0.,
    eEpsilonX=1, eEpsilonZ=0.01,
    betaX=1.20, betaZ=3.95,
    filamentBeam=True,
    uniformRayDensity=True,
    xPrimeMax=np.arctan(slitDx/R0)*2e3, zPrimeMax=np.arctan(slitDz/R0)*2e3,
    targetE=[E0, 3],
    eMin=eMinRays,
    eMax=eMaxRays,
    targetOpenCL=tOCL)

prefix = 'far{0:02.0f}m-E0{1:4.0f}-'.format(R0*1e-3, E0)
suffix = "_realEmittance"

if False:  # zero source size:
    kwargs['eSigmaX'] = 1e-3
    kwargs['eSigmaZ'] = 1e-3
    kwargs['eEpsilonX'] = 0
    kwargs['eEpsilonZ'] = 0

imcnst = 164

imSizeX = imcnst
imSizeZ = imcnst
xBins = imcnst
zBins = imcnst
xppb = 1
zppb = 1
imSize = imcnst
eBins = 16
eppb = 10
xfactor = 1.  # e3
zfactor = 1.  # e3
isScreenHemispheric = False
if isScreenHemispheric:
    screenName = '-hemis'
    xlimits = [-4*slitDx/R0*1e6, 4*slitDx/R0*1e6]
    zlimits = [-4*slitDz/R0*1e6, 4*slitDz/R0*1e6]
    xName = '$\\theta$'
    zName = '$\\phi$'
    unit = u'µrad'
else:
    screenName = '-plane'
    xlimits = [-0.175, 0.175]
    zlimits = [-0.175, 0.175]
    xName = '$x$'
    zName = '$z$'
    unit = "mm"

dx = (xlimits[1] - xlimits[0]) / float(xBins)
xmesh = np.linspace((xlimits[0] + dx/2) / xfactor,
                    (xlimits[1] - dx/2) / xfactor, xBins)
dz = (zlimits[1] - zlimits[0]) / float(zBins)
zmesh = np.linspace((zlimits[0] + dz/2) / zfactor,
                    (zlimits[1] - dz/2) / zfactor, zBins)


def build_beamline(nrays=mynrays):
    beamLine = raycing.BeamLine()
    rs.Undulator(beamLine, nrays=nrays, **kwargs)
    beamLine.fsm0 = rsc.Screen(beamLine, 'FSM0', (0, R0-1, 0))
    #beamLine.slit = ra.DoubleSlit(
    #    beamLine, 'squareSlit', [0, R0, 0],
#        ('left', 'right', 'bottom', 'top'),
    #    [-slitDx, slitDx, -slitDz, slitDz], ShadeFraction=0.1)
    beamLine.slit = ra.RectangularAperture(
        beamLine, 'squareSlit', [0, R0, 0], ('left', 'right', 'bottom', 'top'),
        [-slitDx, slitDx, -slitDz, slitDz])

    if isScreenHemispheric:
        beamLine.fsm1 = rsc.HemisphericScreen(
            beamLine, 'FSM1-hs', (0, 4*R0, 0), R=R0, targetOpenCL=tOCL)
    else:
        beamLine.fsm1 = rsc.Screen(beamLine, 'FSM1', [0, R0, 0],
                                   targetOpenCL=tOCL)
    return beamLine


def run_process(beamLine):
    area = ((beamLine.slit.opening[3] - beamLine.slit.opening[2]) *
            (beamLine.slit.opening[1] - beamLine.slit.opening[0]))
    beamLine.fsm1.expose_wave_prepare(beamLine.slit, xmesh, zmesh, area)
    beamSource = None
    repeats = 10
    for repeat in range(repeats):
        beamSource = beamLine.sources[0].shine(accuBeam=beamSource)
#                                               fixedEnergy=E0)
        beamFSM0 = beamLine.fsm0.expose(beamSource)
        beamFSM1, beamFSM1local, beamFSM1wave, beamFSM1waveLocal =\
            beamLine.fsm1.expose_wave(beamSource)
        if beamLine.fsm1.expose_wave_repeats == 0:
            break
        if repeats > 1:
            print 'wave repeats: {0} of {1} done'.format(repeat+1, repeats)

    outDict = {'beamSource': beamSource,
               'beamFSM0': beamFSM0,
               'beamFSM1': beamFSM1,
               'beamFSM1waveLocal': beamFSM1waveLocal
               }
    return outDict
rr.run_process = run_process


def main():
    beamLine = build_beamline()
    plots = []

    plot = xrtp.XYCPlot(
        'beamFSM0', aspect='auto',
        xaxis=xrtp.XYCAxis(xName, unit, bins=xBins, ppb=xppb),
        yaxis=xrtp.XYCAxis(zName, unit, bins=zBins, ppb=zppb),
        caxis=xrtp.XYCAxis('energy', 'eV', bins=eBins, ppb=eppb),
        title='1-BeamSource')
    plot.baseName = plot.title + suffix
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSM1', aspect='auto',
        xaxis=xrtp.XYCAxis(xName, unit, bins=xBins, ppb=xppb),
        yaxis=xrtp.XYCAxis(zName, unit, bins=zBins, ppb=zppb),
        caxis=xrtp.XYCAxis('energy', 'eV', bins=eBins, ppb=eppb),
        title='2-Screen Rays')
    plot.baseName = plot.title + suffix
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSM1waveLocal', aspect='auto',
        xaxis=xrtp.XYCAxis(xName, unit, bins=xBins, ppb=xppb),
        yaxis=xrtp.XYCAxis(zName, unit, bins=zBins, ppb=zppb),
        caxis=xrtp.XYCAxis('energy', 'eV', bins=eBins, ppb=eppb),
        title='3-Screen Wave')
    plot.baseName = plot.title + suffix
    plots.append(plot)

    def plot_generator():
        for dS in np.linspace(slitDx/2, slitDz/2, 1):
            beamLine.slit.opening[0] = -dS
            beamLine.slit.opening[1] = dS
            beamLine.slit.opening[2] = -dS
            beamLine.slit.opening[3] = dS

            dX = 3.7
            beamLine.fsm1.center[1] = R0+dX*1000
            for plot in plots:
                plot.ax2dHist.locator_params(nbins=4)
                plot.xaxis.fwhmFormatStr = '%.2f'
                plot.yaxis.fwhmFormatStr = '%.2f'
                plot.xaxis.limits = xlimits
                plot.yaxis.limits = zlimits
                plot.caxis.limits = [eMinRays, eMaxRays]
                plot.caxis.offset = (eMinRays + eMaxRays) / 2
                plot.fluxFormatStr = '%.2p'
                plot.saveName =\
                    plot.baseName +\
                    ", slit {0:.2f} mm at {1} m, screen at {2} m.png".\
                    format(2*dS, R0/1000, dX)
                plot.persistentName = plot.saveName + '.pickle'
            yield
    xrtr.run_ray_tracing(plots, repeats=nrep, beamLine=beamLine, processes=1,
                         generator=plot_generator)

# this is necessary to use multiprocessing in Windows, otherwise the new Python
# contexts cannot be initialized:
if __name__ == '__main__':
    main()
