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
#import xrt.backends.raycing.sources as rs
#import xrt.backends.raycing.run as rr

stripe = 'Si'
E0 = 9000
dE = 2


def main():
    beamLine = BalderBL.build_beamline(
        stripe=stripe, eMinRays=E0-dE, eMaxRays=E0+dE)
    plots = []

    plot = xrtp.XYCPlot(
        'beamFSMSample', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'mm'),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
        caxis=xrtp.XYCAxis('energy', 'eV'), title='Sample', ePos=0)
    plot.xaxis.limits = [-10, 10]
    plot.yaxis.limits = [42.79-10, 42.79+10]
#    plot.xaxis.fwhmFormatStr = '%.0f'
#    plot.yaxis.fwhmFormatStr = '%.2f'
    plot.fluxFormatStr = '%.1p'
    plot.textPanel = plot.ax2dHist.text(
        0.5, 0.9, '', transform=plot.ax2dHist.transAxes, size=14, color='r',
        ha='center')
    plots.append(plot)

    for plot in plots:
        plot.caxis.limits = [E0 - dE, E0 + dE]
        plot.caxis.offset = E0

    pitches = np.linspace(1., 4., 31)

    def plot_generator():
        for pitch in pitches:
            BalderBL.align_beamline(beamLine, energy=E0, pitch=pitch*1e-3)
            for plot in plots:
                baseName = 'pitch-{0}{1:.1f}mrad'.format(plot.title, pitch)
                plot.saveName = baseName + '.png'
#                plot.persistentName = baseName + '.pickle'
                if hasattr(plot, 'textPanel'):
                    plot.textPanel.set_text(
                        r'$\theta$ = {0:.1f} mrad'.format(pitch))
            yield

    xrtr.run_ray_tracing(
        plots, repeats=0, generator=plot_generator,
        beamLine=beamLine, globalNorm=True, processes='all')

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()
