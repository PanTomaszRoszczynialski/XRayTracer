# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "20 Apr 2013"

import sys
#sys.path.append(r"/media/sf_Ray-tracing")
sys.path.append(r"c:\Ray-tracing")
import numpy as np
import BalderBL
import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.run as rr


def run_process(beamLine):
    beamSource = beamLine.sources[0].shine()

    beamLine.feFixedMask.propagate(beamSource)
    beamFilter1global, beamFilter1local1, beamFilter1local2 = \
        beamLine.filter1.double_refract(beamSource)
    beamFilter1local2A = rs.Beam(copyFrom=beamFilter1local2)
    beamFilter1local2A.absorb_intensity(beamSource)

    outDict = {'beamSource': beamSource,
               'beamFilter1global': beamFilter1global,
               'beamFilter1local1': beamFilter1local1,
               'beamFilter1local2': beamFilter1local2,
               'beamFilter1local2A': beamFilter1local2A}
    beamLine.beams = outDict
    return outDict

rr.run_process = run_process


def main():
    beamLine = BalderBL.build_beamline()
    BalderBL.align_beamline(beamLine)
    plots = []

    plot = xrtp.XYCPlot(
        'beamFilter1local1', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'), yaxis=xrtp.XYCAxis(r'$y$', 'mm'),
        caxis=xrtp.XYCAxis('energy', 'keV'),
        title='Footprint1_I')
    plot.fluxFormatStr = '%.2p'
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFilter1local1', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'), yaxis=xrtp.XYCAxis(r'$y$', 'mm'),
        caxis=xrtp.XYCAxis('energy', 'keV'),
        fluxKind='power', title='Footprint1_P',
        contourLevels=[0.85, 0.95], contourColors=['b', 'r'],
        contourFmt=r'%.1f W/mm$^2$')
    plot.fluxFormatStr = '%.0f'
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFilter1local2', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'), yaxis=xrtp.XYCAxis(r'$y$', 'mm'),
        caxis=xrtp.XYCAxis('energy', 'keV'),
        title='Footprint2_I')
    plot.fluxFormatStr = '%.2p'
    plot.textPanel = plot.fig.text(
        0.88, 0.8, '', transform=plot.fig.transFigure,
        size=14, color='r', ha='center')
    plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFilter1local2A', (1,),
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'), yaxis=xrtp.XYCAxis(r'$y$', 'mm'),
        caxis=xrtp.XYCAxis('energy', 'keV'),
        fluxKind='power', title='Footprint2_PA',
        contourLevels=[0.85, 0.95], contourColors=['b', 'r'],
        contourFmt=r'%.1f W/mm$^2$')
    plot.fluxFormatStr = '%.0f'
    plot.textPanel = plot.fig.text(
        0.88, 0.8, '', transform=plot.fig.transFigure, size=14, color='r',
        ha='center')
    plots.append(plot)

    for plot in plots:
        plot.xaxis.limits = [-7, 7]
        plot.yaxis.limits = [-7, 7]
        plot.caxis.limits = [0, beamLine.sources[0].eMax*1e-3]
        plot.caxis.fwhmFormatStr = None

    thicknesses = np.linspace(0, 0.4, 21)

    def plot_generator():
        for t in thicknesses:
            beamLine.filter1.t = t
            for plot in plots:
                baseName = 'filter{0}{1:03.0f}mum'.format(plot.title, t*1e3)
                plot.saveName = baseName + '.png'
#                plot.persistentName = baseName + '.pickle'
                if hasattr(plot, 'textPanel'):
                    plot.textPanel.set_text(u'$t$ = {0:.0f} µm'.format(t*1e3))
            yield

    xrtr.run_ray_tracing(
        plots, repeats=160, generator=plot_generator,
        beamLine=beamLine, globalNorm=True, processes='all')

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()
