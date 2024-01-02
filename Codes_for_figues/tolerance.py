n = {'Cs': 1, 'Rb': 1, 'K': 1, 'Cd': 2, 'Ge': 2, 'Hg': 2, "Pb": 2, 'Sn': 2, 'Zn': 2, 'Br': -1, 'Cl': -1, 'I': -1}
r = {'Cs': 1.88, 'Rb': 1.72, 'K': 1.64, 'Cd': 0.95, 'Ge': 0.73, 'Hg': 1.02, "Pb": 1.19, 'Sn': 1.15, 'Zn': 0.74,
          'Br': 1.96, 'Cl': 1.81, 'I': 2.2}
import numpy as np
def Filip_tolerance (mu_bar, delta_mu, t):
    if t <= 1 and mu_bar >= np.sqrt(2) - 1 + delta_mu and t >= (0.44 * mu_bar + 1.37) / (
            np.sqrt(2) * (mu_bar + 1)) and t >= (0.73 * mu_bar + 1.13) / (np.sqrt(2) * (mu_bar + 1)) \
            and t < 2.46 / np.sqrt(2 * (mu_bar + 1) ** 2 + delta_mu ** 2) and mu_bar <= 1.14:
        return "Perovskite"
    else:
        return "Non-perovskite"

def Bartel_tolerance (rB, A, X):
    tol = r[X] / (rB) - n[A] * (n[A] - r[A] / (rB * np.log(r[A] / rB)))
    if tol < 4.18:
        return "Perovskite"
    else:
        return "Non-perovskite"

A = "Cs"
B1s = ['Ge', 'Ge', 'Ge', 'Ge', 'Pb', 'Pb', 'Pb', 'Ge']
B2s = ['Sn', 'Sn', 'Sn', 'Sn', 'Sn', 'Sn', 'Sn', 'Pb']
Xs = ['Br', 'Br', 'Br', 'I', 'Br', 'Br', 'Br', 'I']
ratio1s = [0.25, 0.5, 0.75, 0.5, 0.3, 0.5, 0.7, 0.1]
ratio2s = [0.75, 0.5, 0.25, 0.5, 0.7, 0.5, 0.3, 0.9]
for B1, B2, ratio1, ratio2, X in zip(B1s, B2s, ratio1s, ratio2s, Xs):
    elems = f"{A}{B1}{ratio1}{B2}{ratio2}{X}"
    mu_bar = (ratio1 * r[B1] + ratio2 * r[B2]) / r[X]
    delta_mu = (np.abs(ratio1 * r[B1] - ratio2 * r[B2])) / r[X]
    t = (r[A] / r[X] + 1) / np.sqrt(2 * (mu_bar + 1) ** 2 + (delta_mu) ** 2)
    rB = ratio1*r[B1] + ratio2*r[B2]
    print(f"{elems}: Filip's prediction: {Filip_tolerance(mu_bar, delta_mu, t)}, Bartel's prediction: {Bartel_tolerance(rB, A, X)}")