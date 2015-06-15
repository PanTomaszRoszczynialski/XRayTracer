# -*- coding: utf-8 -*-
r"""
.. _mirrorDiffraction:

Diffraction from mirror surface
-------------------------------

This example shows wave diffraction from a geometric source onto a flat or
hemispheric screen. The source is rectangular with no divergence. The mirror
has no material properties (reflectivity is 1) for simplicity. Notice the
difference in the calculated flux between the rays and the waves.

+------------+------------+
|    rays    |    wave    |
+============+============+
| |mirrorWR| | |mirrorWW| |
+------------+------------+

.. |mirrorWR| image:: _images/mirror-256-01flat-01-beamFSMrays_f.*
   :scale: 50 %
.. |mirrorWW| image:: _images/mirror-256-01flat-02-beamFSMwave_f.*
   :scale: 50 %

The flux losses are not due to the integration errors, as was proven by
variously dense meshes. The losses are solely caused by cutting the tails, as
proven by a wider image shown below.

.. image:: _images/mirror-256wide-01flat-02-beamFSMwave_f.*
   :scale: 50 %
"""
__author__ = "Konstantin Klementiev"
__date__ = "20 Jan 2014"
import sys
sys.path.append(r"c:\Ray-tracing")
# sys.path.append(r"/media/c/Ray-tracing")
import numpy as np

import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.apertures as ra
#import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.screens as rsc

#coating = rm.Material('Au', rho=19.3)
#coating = rm.Material('Ni', rho=8.902)
coating = None

tCL = (0, 0)
#tCL = None
E0 = 150.
dE = 1.
dx = 1.
pitch = 4e-3
p = 20000.
q = 20000.
dPrime = 2e-5

cosTheta = np.cos(pitch)
sinTheta = np.sin(pitch)

#xBins, xppb = 128, 2
#yBins, yppb = 128, 2
xBins, xppb = 256, 1
yBins, yppb = 256, 1
eBins, eppb = 32, 8
xName = 'yaw'
yName = 'pitch'
unit, ufactor = u'µrad', 1e6

oe = 'mirror'
#oe = 'aperture'

prefix = oe
nrays = 1e5

sourceType = 'flat'
#sourceType = 'annulus'
#sourceType = 'divergent'
if sourceType == 'flat':
    kw = {'distx': 'flat', 'dx': dx, 'distz': 'flat', 'dz': dx,
          'distxprime': None, 'distzprime': None}
    prefix += '-01' + sourceType + '-'
elif sourceType == 'annulus':
    kw = {'distx': 'annulus', 'dx': (0, dx/2),
          'distxprime': None, 'distzprime': None}
    prefix += '-02' + sourceType + '-'
elif sourceType == 'divergent':
    kw = {'distx': None, 'distz': None,
          'distxprime': 'flat', 'distzprime': 'flat',
          'dxprime': dPrime, 'dzprime': dPrime}
    prefix += '-03' + sourceType + '-'

kw['distE'] = 'lines'
kw['energies'] = [E0]
#kw['distE'] = 'flat'
#kw['energies'] = [E0-dE/2, E0+dE/2]


def build_beamline():
    beamLine = raycing.BeamLine()
    beamLine.plotW = None
    rs.GeometricSource(
        beamLine, 'source', nrays=nrays, polarization='horizontal', **kw)

    if oe == 'mirror':
        beamLine.diffoe = roe.OE(
            beamLine, 'PlaneMirror', (0, p, 0), pitch=pitch, material=coating)
#            targetOpenCL=tCL)
        phiOffset = 2 * pitch
        zFlatScreen = q*np.tan(2*pitch)
    elif oe == 'aperture':
        beamLine.diffoe = ra.RectangularAperture(
            beamLine, 'ra', (0, p, 0), ('left', 'right', 'bottom', 'top'))
        phiOffset = 0
        zFlatScreen = 0

    beamLine.fsmF = rsc.Screen(beamLine, 'FSM', [0, p+q, zFlatScreen],
                               targetOpenCL=tCL)
    beamLine.fsmH = rsc.HemisphericScreen(
        beamLine, 'FSM', (0, p, 0), R=q, phiOffset=phiOffset,
        targetOpenCL=tCL)
    return beamLine


def run_process(beamLine):
    beamLine.fsmF.expose_wave_prepare(
        beamLine.diffoe, beamLine.xMeshF, beamLine.zMeshF)
    beamLine.fsmH.expose_wave_prepare(
        beamLine.diffoe, beamLine.xMeshH, beamLine.zMeshH)

    repeats = 100
    for repeat in range(repeats):
        beamSource = beamLine.sources[0].shine(withAmplitudes=True)
        if repeats > 1:
            print 'wave repeats: {0} of {1} done'.format(repeat+1, repeats)

        oeGlobalF, oeLocalF, bgWF, blWF = beamLine.fsmF.expose_wave(beamSource)
        beamFSMraysF = beamLine.fsmF.expose(oeGlobalF)

        oeGlobalH, oeLocalH, bgWH, blWH = beamLine.fsmH.expose_wave(beamSource)
        beamFSMraysH = beamLine.fsmH.expose(oeGlobalH)

    outDict = {'beamSource': beamSource,
               'beamFSMwave_f': blWF,
               'beamFSMrays_f': beamFSMraysF,
               'beamFSMwave_h': blWH,
               'beamFSMrays_h': beamFSMraysH}
    return outDict
rr.run_process = run_process


def nodes(dmin, dmax, dbins):
    dd = (dmax - dmin) / dbins
    centers = np.linspace(dmin + dd/2, dmax - dd/2, dbins)
    return centers


def main():
    beamLine = build_beamline()
    plots = []

    if beamLine.sources[0].distxprime == 'flat':
        mf = dPrime/2 * (p+q) * 4
        ms = mf / q * ufactor
        ma = dPrime/2 * p
    else:
        mf = 0.55 * dx * 2
        ms = mf / q * ufactor
        ma = dx/2
    if oe == 'aperture':
        beamLine.diffoe.opening = [-ma, ma, -ma, ma]

    dataPhi = raycing.get_phi
    dataTheta = raycing.get_theta

    beamLine.xMeshF = nodes(-mf, mf, xBins)
    beamLine.zMeshF = nodes(-mf, mf, yBins)
    beamLine.xMeshH = nodes(-ms, ms, xBins) / ufactor
    beamLine.zMeshH = nodes(-ms, ms, yBins) / ufactor

    plot = xrtp.XYCPlot(
        'beamFSMrays_f', aspect='auto',
        xaxis=xrtp.XYCAxis('x', 'mm', bins=xBins, ppb=xppb, limits=[-mf, mf]),
        yaxis=xrtp.XYCAxis('z', 'mm', bins=yBins, ppb=yppb, limits=[-mf, mf]),
        caxis=xrtp.XYCAxis('energy', 'eV', bins=eBins, ppb=eppb))
    plot.baseName = prefix + '01-beamFSMrays_f'
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSMwave_f', aspect='auto',
        xaxis=xrtp.XYCAxis('x', 'mm', bins=xBins, ppb=xppb, limits=[-mf, mf]),
        yaxis=xrtp.XYCAxis('z', 'mm', bins=yBins, ppb=yppb, limits=[-mf, mf]),
        caxis=xrtp.XYCAxis('energy', 'eV', bins=eBins, ppb=eppb))
    plot.baseName = prefix + '02-beamFSMwave_f'
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSMrays_h', aspect='auto',
        xaxis=xrtp.XYCAxis(xName, unit, bins=xBins, ppb=xppb, data=dataTheta,
                           limits=[-ms, ms]),
        yaxis=xrtp.XYCAxis(yName, unit, bins=yBins, ppb=yppb, data=dataPhi,
                           limits=[-ms, ms]),
        caxis=xrtp.XYCAxis('energy', 'eV', bins=eBins, ppb=eppb))
    plot.baseName = prefix + '01-beamFSMrays_h'
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSMwave_h', aspect='auto',
        xaxis=xrtp.XYCAxis(xName, unit, bins=xBins, ppb=xppb, data=dataTheta,
                           limits=[-ms, ms]),
        yaxis=xrtp.XYCAxis(yName, unit, bins=yBins, ppb=yppb, data=dataPhi,
                           limits=[-ms, ms]),
        caxis=xrtp.XYCAxis('energy', 'eV', bins=eBins, ppb=eppb))
    plot.baseName = prefix + '02-beamFSMwave_h'
    plots.append(plot)
    beamLine.cohPlotS = plot

    for plot in plots:
        plot.xaxis.fwhmFormatStr = '%.2f'
        plot.yaxis.fwhmFormatStr = '%.2f'
        plot.caxis.limits = [E0-dE/2, E0+dE/2]
        plot.caxis.offset = E0
        plot.fluxFormatStr = '%.2p'
        if hasattr(plot, 'baseName'):
            plot.saveName = plot.baseName + '.png'
#            plot.persistentName = plot.baseName + '.pickle'

    xrtr.run_ray_tracing(plots, repeats=1, beamLine=beamLine)

# this is necessary to use multiprocessing in Windows, otherwise the new Python
# contexts cannot be initialized:
if __name__ == '__main__':
    main()
