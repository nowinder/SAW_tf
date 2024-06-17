import math
# from scipy import special
# import cmath
# import sawcom7 as sc
import random

import numpy as np

pI = 3600E-9  # The period of IDT, normally is one wavelength
h = 0.08 * pI  # The thickness of IDT, Al or Al-Cu1%
W1 = 20 * pI  # Width of IDT (acoustic aperture), in m
m_ratio = 0.6  # The metallization ratio
epsilon_0 = 8.8541878128e-12  # The permittivity of vacuum
npiezo_1 = 50.3562796837374
eta = (0.182 + 0.349 * (2 * h / pI)) * math.sqrt(2 * np.pi / pI)
e = (0.0388 + 0.618 * (2 * h / pI)) * math.sqrt(2 * np.pi / pI)
alpha = 0.05
c = 1 + (0.0678 + 1.057 * (2 * h / pI)) ** 2
k2 = 0.0655 + 0.206 * (2 * h / pI)
vb = 4226.54
ratio = [0.2, 0.1, 0.05]
x = np.zeros((60000, 7))
for i in range(3):
    npiezo_1_num = [i for i in np.linspace(npiezo_1 * (1-ratio[i]), npiezo_1 * (1+ratio[i]), 10000)]
    eta_num = [i for i in np.linspace(eta * (1-ratio[i]), eta * (1+ratio[i]), 10000)]
    e_num = [i for i in np.linspace(e * (1-ratio[i]), e * (1+ratio[i]), 10000)]
    alpha_num = [i for i in np.linspace(alpha * (1-ratio[i]), alpha * (1+ratio[i]), 10000)]
    c_num = [i for i in np.linspace(c * max(0.01, 1-5*ratio[i]), c * (1+5*ratio[i]), 10000)]
    k2_num = [i for i in np.linspace(k2 * 0.01, k2 * (1+20*ratio[i]), 10000)]
    vb_num = [i for i in np.linspace(vb * (1-0.005), vb * (1+0.005), 10000)]
    # vb_num = [i for i in np.linspace(vb * (1-ratio[i]), vb * (1+ratio[i]), 10000)]

    for j in range(0, 20000):
        random.seed()
        npi = random.sample(npiezo_1_num, 1)[0]
        et = random.sample(eta_num, 1)[0]
        e = random.sample(e_num, 1)[0]
        al = random.sample(alpha_num, 1)[0]
        c = random.sample(c_num, 1)[0]
        k = random.sample(k2_num, 1)[0]
        v = random.sample(vb_num, 1)[0]
        x[j+20000*i, :] = [npi, et, e, al, c, k, v]

x_max = x.max(axis=0).reshape((1, 7))
x_min = x.min(axis=0).reshape((1, 7))
file = 'G:/Zheng_caizhi/Pycharmprojects/SAW_tf/datas/out/' + 'vb.005' + '.csv'
with open(file, 'w', newline='') as f:
    np.savetxt(f, x, delimiter=',', newline='\n')
file = file + 'maxmin.csv'
with open(file, 'w', newline='') as f:
    np.savetxt(f, x_max, delimiter=',', newline='\n')
    np.savetxt(f, x_min, delimiter=',', newline='\n')
