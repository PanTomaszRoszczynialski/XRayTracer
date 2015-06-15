# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "10 Apr 2015"

import numpy as np
import matplotlib.pyplot as plt


def read_spectra_2D(fname, what='total'):
    f, l1, l2, l3 = np.loadtxt(fname, skiprows=10, usecols=[2, 3, 4, 5],
                               unpack=True)
    print f.shape
    if what == 's':
        f *= (1 + l1) / 2.
    elif what == 'p':
        f *= (1 - l1) / 2.
    elif what == 'circ':
        f *= (l2 + 1j*l3) / 2.

    f = f.reshape((128, 128))
#    f = np.concatenate((f[:0:-1, :], f), axis=0)
#    f = np.concatenate((f[:, :0:-1], f), axis=1)
    print f.shape, f.min(), f.max()
    return f


def show_spectra_2D(f2, xmax, zmax, saveName):
    fig1 = plt.figure(1, figsize=(4, 3.2))
    ax1 = fig1.add_axes([0.15, 0.15, 0.8, 0.8], aspect='equal', label='1')
    ax1.set_xlabel(u'$x$ (mm)')
    ax1.set_ylabel(u'$z$ (mm)')
    ax1.set_xlim(-xmax, xmax)
    ax1.set_ylim(-zmax, zmax)
    if f2 is not None:
        ax1.imshow(f2, vmin=0., extent=[-xmax, xmax, -zmax, zmax],
                   cmap='gray')
    for axXY in (ax1.xaxis, ax1.yaxis):
        for line in axXY.get_ticklines():
            line.set_color('grey')
#    ax1.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
#    ax1.set_yticks([-0.2, -0.1, 0, 0.1, 0.2])

    fig1.savefig('{0}.png'.format(saveName))
    plt.show()

if __name__ == '__main__':
    fName = 'spectra_far_R0=25mZoom'
    lim = 1
    fXZ = read_spectra_2D(fName+'.dta', what='total')
    show_spectra_2D(fXZ, lim, lim, fName)
    fXZ = read_spectra_2D(fName+'.dta', what='s')
    show_spectra_2D(fXZ, lim, lim, fName+'_s')
    fXZ = read_spectra_2D(fName+'.dta', what='p')
    show_spectra_2D(fXZ, lim, lim, fName+'_p')
