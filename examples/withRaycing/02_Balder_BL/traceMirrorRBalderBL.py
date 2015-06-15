# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev"
__date__ = "20 Apr 2013"

import sys
#sys.path.append(r"/media/sf_Ray-tracing")
sys.path.append(r"c:\Ray-tracing")

import numpy as np
from matplotlib.ticker import FixedLocator

import BalderBL
import xrt.plotter as xrtp
import xrt.runner as xrtr
#import xrt.backends.raycing.sources as rs
#import xrt.backends.raycing.run as rr

stripe = 'Si'
E0 = 9000
dE = 2

mirror = 'vfm'
mirrorText = 'collimating' if mirror == 'vcm' else 'focusing'


def main():
    beamLine = BalderBL.build_beamline(
        stripe=stripe, eMinRays=E0-dE, eMaxRays=E0+dE)
    BalderBL.align_beamline(beamLine, energy=E0)
    plots = []

    if mirror == 'vcm':
        plot = xrtp.XYCPlot(
            'beamFSMDCM', (1,),
            xaxis=xrtp.XYCAxis(r'$x$', 'mm'), yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
            caxis=xrtp.XYCAxis('energy', 'eV'), title='DCM')
        plot.xaxis.limits = [-7., 7.]
        plot.yaxis.limits = [38.1-7., 38.1+7.]
        plot.fluxFormatStr = '%.2p'
        plot.textPanel = plot.fig.text(
            0.88, 0.8, '', transform=plot.fig.transFigure, size=14, color='r',
            ha='center')
        plots.append(plot)

    plot = xrtp.XYCPlot(
        'beamFSMSample', (1,), aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', u'µm', bins=xrtp.defaultBins/2),
        yaxis=xrtp.XYCAxis(r'$z$', 'mm'),
        caxis=xrtp.XYCAxis('energy', 'eV'), title='Sample')
    plot.xaxis.limits = [-300, 300]
    plot.yaxis.limits = [42.8-0.6, 42.8+0.6]
    plot.ax2dHist.xaxis.set_major_locator(FixedLocator([-200, 0, 200]))
    plot.xaxis.fwhmFormatStr = '%.0f'
    plot.yaxis.fwhmFormatStr = '%.2f'
    plot.fluxFormatStr = '%.2p'
    plot.textPanel = plot.fig.text(
        0.88, 0.8, '', transform=plot.fig.transFigure, size=14, color='r',
        ha='center')
    plots.append(plot)

    for plot in plots:
        plot.caxis.limits = [E0 - dE, E0 + dE]
        plot.caxis.offset = E0

    Rs = np.linspace(0.6, 1.4, 21)
    if mirror == 'vcm':
        Rs *= beamLine.vcm.R
    else:
        Rs *= beamLine.vfm.R

    def plot_generator():
        for R in Rs:
            if mirror == 'vcm':
                beamLine.vcm.R = R
            else:
                beamLine.vfm.R = R
            for plot in plots:
                baseName = '{0}R-{1}{2:.1f}km'.format(
                    mirror, plot.title, R*1e-6)
                plot.saveName = baseName + '.png'
#                plot.persistentName = baseName + '.pickle'
                if hasattr(plot, 'textPanel'):
                    plot.textPanel.set_text(
                        '{0}\nmirror\n$R$ = {1:.1f} km'.format(
                            mirrorText, R*1e-6))
            yield

    xrtr.run_ray_tracing(
        plots, repeats=160, generator=plot_generator,
        beamLine=beamLine, globalNorm=True, processes='half')

#this is necessary to use multiprocessing in Windows, otherwise the new Python
#contexts cannot be initialized:
if __name__ == '__main__':
    main()
