# -*- coding: utf-8 -*-
__author__ = "Konstantin Klementiev", "Roman Chernikov"
__date__ = "10 Apr 2015"
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
if sys.version_info < (3, 1):
    from string import maketrans
    transFortranD = maketrans('dD', 'ee')
else:
#    import string
    transFortranD = str.maketrans('dD', 'ee')


def plotOwn(ax, pickleName, style, label, lw=2):
    pickleFile = open(pickleName, 'rb')
#    flux = [plot.intensity, plot.nRaysAll, plot.nRaysAccepted,
#            plot.nRaysSeeded]
    _f, binEdges, total1D = pickle.load(pickleFile)
    flux = _f[0] / _f[1] * _f[2] / _f[3]
    pickleFile.close()
    ax.set_xlim(binEdges[0], binEdges[-1])
    dE = binEdges[1] - binEdges[0]
    E = binEdges[:-1] + dE/2.
    if pickleName == '1u_xrt3-n-monoE-BW.pickle':  # sampled per BW
        y = total1D / sum(total1D) * flux / dE * 1e-3  # because dE is in keV
    else:  # sampled per eV
        y = total1D / sum(total1D) * flux * E / dE * 1e-3
    print flux, sum(total1D), dE, sum(y*dE/E*1e3)
    ax.plot(E, y, style, label=label, lw=lw)


def main():
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(r'Integrated flux into 0.5$\times$0.5 mrad$^2$', fontsize=14)
#    fig.subplots_adjust(right=0.88)
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'energy (keV)')
    ax.set_ylabel(r'flux (ph/s/0.1%bw)')

    plotOwn(ax, '1u_urgent3-n-monoE-.pickle', '-m',
            'ray-traced, xrt with Urgent')
#    plotOwn(ax, '1u_xrt3-n-monoE-BW.pickle', '.b',
#            'ray-traced, xrt, distE=BW')
    plotOwn(ax, '1u_xrt3-n-monoE-noEspread.pickle', '--b',
            'ray-traced, xrt internal, no E spread')
    plotOwn(ax, '1u_xrt3-n-monoE-Espread.pickle', '-b',
            'ray-traced, xrt internal, with E spread', lw=3)

    E, fl = np.loadtxt(
        "urgentOpt.out", skiprows=32, usecols=(0, 2), unpack=True,
        converters={2: lambda s: float(s.translate(transFortranD))},
        comments=" TOTAL")
    ax.plot(E*1e-3, fl, '-y', lw=2, label='XOP Urgent, icalc=1, mode4')

    E, fl = np.loadtxt("us-mode3.out", skiprows=23, usecols=(0, 1),
                       unpack=True)
    ax.plot(E*1e-3, fl, '-c', lw=2, label='XOP US, mode3')

    E, fl = np.loadtxt("Spectra_FluxZeroESpread.dc0", skiprows=10,
                       usecols=(0, 1), unpack=True)
    ax.plot(E*1e-3, fl, '--r', lw=2, label='Spectra, no E spread')
    E, fl = np.loadtxt("Spectra_Flux.dc0", skiprows=10, usecols=(0, 1),
                       unpack=True)
    ax.plot(E*1e-3, fl, '-r', lw=3, label='Spectra, with E spread')

    ax.legend(loc='lower left')
    ax.set_ylim(0, None)

    fig.savefig('compareUndulators.png')
    plt.show()

if __name__ == '__main__':
    main()
