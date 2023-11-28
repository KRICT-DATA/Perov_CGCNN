import pandas as pd
import numpy as np
Compounds = ['CsGe0.5625Sn0.4375Br3', 'CsGe0.3125Sn0.6875I3', 'CsPb0.3125Sn0.6875Br3', 'CsGe0.375Pb0.625I3',
             'CsGe0.0625Pb0.3125Sn0.625Br3', 'CsGe0.4375Pb0.0625Sn0.5Br3', 'CsGe0.25Pb0.5625Sn0.1875I3',
             'CsGe0.5625Hg0.3125Sn0.125Cl3', 'CsCd0.0625Ge0.25Pb0.5Sn0.1875I3', 'CsCd0.125Ge0.5625Hg0.1875Sn0.125Cl3']

compounds = [r"$\mathregular {CsGe_{0.5625}Sn_{0.4375}Br_3}$", r"$\mathregular {CsGe_{0.3125}Sn_{0.6875}I_3}$",
             r"$\mathregular {CsPb_{0.3125}Sn_{0.6875}Br_3}$", r"$\mathregular {CsGe_{0.375}Pb_{0.625}I_3}$",
             r"$\mathregular {CsGe_{0.0625}Pb_{0.3125}Sn_{0.625}Br_3}$", r"$\mathregular {CsGe_{0.4375}Pb_{0.0625}Sn_{0.5}Br_3}$",
             r"$\mathregular {CsGe_{0.25}Pb_{0.5625}Sn_{0.1875}I_3}$", r"$\mathregular {CsGe_{0.5625}Hg_{0.3125}Sn_{0.125}Cl_3}$",
             r"$\mathregular {CsCd_{0.0625}Ge_{0.25}Pb_{0.5}Sn_{0.1875}I_3}$", r"$\mathregular {CsCd_{0.125}Ge_{0.5625}Hg_{0.1875}Sn_{0.125}Cl_3}$"]
bandgaps = [1.39, 1.34, 1.73, 1.77, 1.71, 1.48, 1.71, 1.75, 1.75, 1.87]
for Compound, compound, bandgap in zip(Compounds, compounds, bandgaps):
    E, alpha = np.loadtxt(f'optical_data/{Compound}/absorption.dat', usecols=[0, 1], unpack=True)
    df = pd.read_csv(f'optical_data/{Compound}/efficiency.csv')
    thickness = np.array(df.iloc[:, 0])
    SLME = 100*np.array(df.iloc[:, 1])
    from matplotlib.pylab import *
    fontdict = {'font': 'Times New Roman', 'size': 24, 'fontweight': 'bold'}
    figure(figsize = (3.2, 2.5), dpi = 200)
    title(compound, size=10)
    semilogx(thickness*1e6, SLME)
    xlabel('Thickness ($\mu m$)')
    ylabel('SLME (%)')
    minorticks_on()
    ax=gca()
    ax.set_xlim(0.01, ax.get_xlim()[1])
    ax.set_ylim(0, ax.get_ylim()[1])
    tight_layout(pad = 0.5)
    axins = ax.inset_axes([0.55, 0.25, 0.425, 0.45])
    axins.plot(E, alpha)
    axins.set_xlim(0, 4)
    axins.set_ylim(0, 1e5)
    axins.set_xlabel('Energy (eV)', size=8)
    axins.set_ylabel(r'Absorption (cm$^{-1}$)', size=8)
    axins.set_xticks([0, 1,2,3,4], size=7)
    axins.axvline(bandgap, ls='dotted')
    from matplotlib.ticker import ScalarFormatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((0, 4))
    axins.yaxis.set_major_formatter(formatter)
    show()
