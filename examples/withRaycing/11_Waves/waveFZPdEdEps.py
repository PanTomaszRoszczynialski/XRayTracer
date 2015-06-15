# -*- coding: utf-8 -*-
"""

Diffraction from FZP at variable band width and divergence
----------------------------------------------------------

This examples extends the previous one and adds into the incoming beam a finite
energy band and a finite divergence. The latter is expressed in terms of a
phase space volume. The resulted flux and focal size are plotted as colored
bubbles in the 2D graph below. The sigma values are the rms focus sizes and
delta is FWHM. The bubble size is proportional to sigma.

.. image:: _images/1-LE-FZP-focus2D.*
   :scale: 50 %
"""
__author__ = "Konstantin Klementiev"
__date__ = "20 Jan 2014"
import os
import sys
sys.path.append(r"c:\Ray-tracing")
#sys.path.append(r"/media/c/Ray-tracing")
import numpy as np
import pickle
#import matplotlib as mpl
#mpl.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import xrt.plotter as xrtp
import xrt.runner as xrtr
import xrt.backends.raycing as raycing
import xrt.backends.raycing.sources as rs
import xrt.backends.raycing.oes as roe
import xrt.backends.raycing.run as rr
import xrt.backends.raycing.materials as rm
import xrt.backends.raycing.screens as rsc

cwd = os.getcwd()
tCL = (0, 0)
#tCL = None

E_FZP = 120
eBins, eppb = 16, 16
mGold = rm.Material('Au', rho=19.3, kind='FZP')

p = 5000
f = 2.
thinnestZone = 50e-6  # in mm
#cmap = cm.get_cmap('gnuplot')
cmap = cm.get_cmap('jet')

prefix = '1-LE-FZP-focus'
pickleName = os.path.join(cwd, prefix + '.pickle')
Nr = 256
rbins = 1
maxOrder = 1
nrays = 1e5

energyBands = np.logspace(-4, -2, 5)
emittances = np.logspace(-7, -5, 5)


def build_beamline():  # for test=True
    beamLine = raycing.BeamLine()
    gsource = rs.GeometricSource(
        beamLine, 'GeometricSource', (0, 0, 0), nrays=nrays,
        distx='annulus', distxprime='annulus',
        distE='normal', polarization='h', filamentBeam=True)

    beamLine.fzp = roe.NormalFZP(
        beamLine, 'FZP', [0, p, 0], pitch=np.pi/2, material=mGold, f=f,
        E=E_FZP, thinnestZone=thinnestZone, isCentralZoneBlack=True)
    gsource.dx = (0, beamLine.fzp.rn[-1])
    beamLine.fzp.surfaceArea = np.pi * beamLine.fzp.rn[-1]**2
    print 'FZP radius = {0} mm'.format(beamLine.fzp.rn[-1])
    print 'FZP N = {0}'.format(len(beamLine.fzp.rn))

    beamLine.r0max = beamLine.fzp.rn[-1] * 5
    beamLine.rNmax = thinnestZone * 40
    beamLine.yglo = np.repeat(
        [10000 if i == 0 else f/i for i in range(maxOrder+1)], Nr)
    beamLine.dr = np.array(
        [beamLine.r0max/(Nr-1) if i == 0 else beamLine.rNmax/(Nr-1)
         for i in range(maxOrder+1)])
    beamLine.zglo = (np.arange(Nr) * beamLine.dr[:, np.newaxis]).flatten()
    beamLine.xglo = np.zeros_like(beamLine.yglo)

    beamLine.fsm = rsc.MonitorPoints(targetOpenCL=tCL)
    return beamLine


def run_process(beamLine):
    beamLine.fsm.expose_wave_prepare(
        beamLine.fzp, beamLine.xglo, beamLine.yglo+p, beamLine.zglo)

    beamSource = None
#    repeats = 1
    repeats = beamLine.repeats
    for repeat in range(repeats):
        beamSource = beamLine.sources[0].shine(
            withAmplitudes=True, accuBeam=beamSource)
        outDict = {'beamSource': beamSource}
        beamLine.fluxIn = (beamSource.Jss + beamSource.Jpp).sum()
        oeGlobal, oeLocal, bgW = beamLine.fsm.expose_wave(
            beamSource)
        if repeats > 1:
            print 'wave repeats: {0} of {1} done'.format(repeat, repeats)

    beamLine.intensityDiffr = bgW.Jss + bgW.Jpp
    flux = (beamLine.intensityDiffr * beamLine.zglo).reshape(maxOrder+1, Nr) *\
        beamLine.dr[:, None] * 2*np.pi

    blo1I = rs.Beam(nrays=Nr, forceState=1)
    blo1I.E[:] = beamSource.E[0]
    blo1I.x[:] = 0.
    blo1I.z[:] = beamLine.zglo.reshape(maxOrder+1, Nr)[1, :]
    blo1I.Jss[:] = beamLine.intensityDiffr.reshape(maxOrder+1, Nr)[1, :]
    blo1I.Jpp[:] = 0
    blo1F = rs.Beam(copyFrom=blo1I)
    blo1F.Jss[:] = flux[1, :]

    outDict['blo1I'] = blo1I
    outDict['blo1F'] = blo1F
    return outDict
rr.run_process = run_process


def get_focus():
    beamLine = build_beamline()
    plots = []

##0th order:
#    plot0I = xrtp.XYCPlot(
#        'blo0I', aspect='auto',
#        xaxis=xrtp.XYCAxis(r'$x$', 'nm', bins=1, ppb=20, limits=[-1, 1]),
#        xPos=0,
#        yaxis=xrtp.XYCAxis(r'$z$', 'mm', bins=Nr, ppb=rbins,
#                           limits=[0, beamLine.r0max]),
#        caxis=xrtp.XYCAxis('energy', 'eV', bins=eBins, ppb=eppb))
#    plots.append(plot0I)
#
#    plot0F = xrtp.XYCPlot(
#        'blo0F', aspect='auto',
#        xaxis=xrtp.XYCAxis(r'$x$', 'nm', bins=1, ppb=20, limits=[-1, 1]),
#        xPos=0,
#        yaxis=xrtp.XYCAxis(r'$z$', 'mm', bins=Nr, ppb=rbins,
#                           limits=[0, beamLine.r0max]),
#        caxis=xrtp.XYCAxis('energy', 'eV', bins=eBins, ppb=eppb))
#    plots.append(plot0F)

#1st order:
    plot1I = xrtp.XYCPlot(
        'blo1I', aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'nm', bins=1, ppb=20, limits=[-1, 1]),
        xPos=0,
        yaxis=xrtp.XYCAxis(r'$z$', 'nm', bins=Nr, ppb=rbins,
                           limits=[0, beamLine.rNmax*1e6]),
        caxis=xrtp.XYCAxis('energy', 'eV', bins=eBins, ppb=eppb))
    plots.append(plot1I)

    plot1F = xrtp.XYCPlot(
        'blo1F', aspect='auto',
        xaxis=xrtp.XYCAxis(r'$x$', 'nm', bins=1, ppb=20, limits=[-1, 1]),
        xPos=0,
        yaxis=xrtp.XYCAxis(r'$z$', 'nm', bins=Nr, ppb=rbins,
                           limits=[0, beamLine.rNmax*1e6]),
        caxis=xrtp.XYCAxis('energy', 'eV', bins=eBins, ppb=eppb))
    plots.append(plot1F)

    for plot in plots:
        plot.caxis.offset = E_FZP
        plot.caxis.fwhmFormatStr = '%.3f'
        plot.yaxis.fwhmFormatStr = '%.2f'
        plot.xaxis.fwhmFormatStr = None
        plot.ax2dHist.xaxis.set_ticks([0])

    def plot_generator():
        fluxs1 = np.zeros((len(energyBands), len(emittances)))
        fwhms1 = np.zeros_like(fluxs1)
        sigma1 = np.zeros_like(fluxs1)

        for ienergyBand, energyBand in enumerate(energyBands):
            print 'energy band:', ienergyBand+1, 'of', len(energyBands)
            beamLine.sources[0].energies = E_FZP, E_FZP*energyBand
            for plot in plots:
                plot.caxis.limits = [E_FZP - 3*E_FZP*energyBand,
                                     E_FZP + 3*E_FZP*energyBand]
#            beamLine.repeats = 1
            beamLine.repeats = int((ienergyBand/2.+1)**2)
            for iemittance, emittance in enumerate(emittances):
                print 'emittance:', iemittance+1, 'of', len(emittances)
                dx = beamLine.sources[0].dx[1]
                beamLine.sources[0].dxprime = 0, emittance/dx
                for plot in plots:
                    plot.baseName = plot.title + 'band{0}em{1}'.format(
                        ienergyBand, iemittance)
                    plot.saveName = [plot.baseName + '.png']
#                    plot.persistentName = plot.baseName + '.pickle'
                yield
                z = beamLine.zglo.reshape(maxOrder+1, Nr)

                fluxs1[ienergyBand, iemittance] = plot1F.intensity
                fwhms1[ienergyBand, iemittance] = 2 * plot1I.dy  # 2*[0, max]
                Iz1 = plot1I.yaxis.total1D
                sigma1[ienergyBand, iemittance] = (
                    (Iz1 * z[1, :]**2).sum() / Iz1.sum())**0.5

        dump = [beamLine.zglo, energyBands, emittances,
                fluxs1, fwhms1, sigma1]
        with open(pickleName, 'wb') as f:
            pickle.dump(dump, f, -1)

    xrtr.run_ray_tracing(plots, repeats=160, beamLine=beamLine, processes=1,
                         generator=plot_generator)


def visualize_efficiency():
    with open(pickleName, 'rb') as f:
        zglo, energyBands, emittances, fluxs1, fwhms1, sigma1 = pickle.load(f)
    emittances *= 1e6  # emittance = 1 is 1nmrad

    fig1 = plt.figure(figsize=(8, 6), dpi=72)
    rect2d = [0.15, 0.1, 0.7, 0.8]
    ax = fig1.add_axes(rect2d, aspect='auto')
    ax.set_title('Focus size and efficiency for an FZP with\n' +
                 r'a {0:.0f}-nm-thick outer zone'.format(thinnestZone*1e6),
                 fontsize=14)

    ax.set_xlabel(r'relative energy band width, $\sigma_E/E$', fontsize=14)
    ax.set_ylabel(r'input beam phase-space volume (nm$\cdot$rad)', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(energyBands[0]/2, energyBands[-1]*2)
    ax.set_ylim(emittances[0]/2, emittances[-1]*2)

    rect2d = [0.95, 0.1, 0.02, 0.8]
    ax1c = fig1.add_axes(rect2d, aspect='auto')
    ax1c.set_ylabel('relative efficiency', fontsize=14)
    ax1c.yaxis.labelpad = 0
    plt.setp(ax1c, xticks=())
    yLim = 0, 1
    ax1c.set_ylim(yLim[0], yLim[1])
    a = np.outer(np.arange(0, 1, 0.01), np.ones(10))
    ax1c.imshow(a, aspect='auto', cmap=cmap, origin="lower",
                extent=[0, 1, yLim[0], yLim[1]])

    ens, ems = np.meshgrid(energyBands, emittances)
    ax.scatter(ens, ems, s=sigma1.T*1e6*10, c=fluxs1.T, cmap=cmap)
#    ax.scatter(ens, ems, s=fwhms1.T*10, c=fluxs1.T, cmap=cmap)

#    wheres = [0, 0], [4, 0]
    wheres = [0, 0], [0, -1], [-1, 0], [-1, -1], [2, 0], [2, -1]
    for ix, iy in wheres:
        plt.text(energyBands[ix], emittances[iy]/1.3,
                 r'$\sigma_R$ = {0:.0f} nm'.format(sigma1[ix, iy]*1e6),
                 transform=ax.transData, ha='center', va='top',
                 color='grey')
        plt.text(energyBands[ix], emittances[iy]*1.3,
                 r'$\Delta_R$ = {0:.0f} nm'.format(fwhms1[ix, iy]),
                 transform=ax.transData, ha='center', va='bottom',
                 color='grey')

    fig1.savefig(prefix + '2D.png')
    plt.show()


if __name__ == '__main__':
#    get_focus()
    visualize_efficiency()
