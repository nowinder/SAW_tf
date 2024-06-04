# %load data_generatev2
# code by Zhengcaizhi
import numpy as np
# import matplotlib.pyplot as plt
import math
from scipy import special
import cmath
# from scipy.signal import argrelextrema
import sawcom7 as sc


def cacul(paras):
    npiezo_1 = paras[0]
    eta = paras[1]
    e = paras[2]
    alpha = paras[3]
    c = paras[4]
    k2 = paras[5]
    vb = paras[6]
    epsilon_0 = 8.8541878128e-12 #The permittivity of vacuum
    pI = 3600E-9 #The period of IDT, normally is one wavelength
    h = 0.08*pI #The thickness of IDT, Al or Al-Cu1%
    W1 = 20*pI # Width of IDT (acoustic aperture), in m
    m_ratio = 0.6 #The metallization ratio

    eta_b = (eta+2*abs(e))/2
    epsilon = npiezo_1*epsilon_0
    x1 = np.cos(np.pi*m_ratio )
    m1 = math.sqrt((1-x1)/2) 
    km1 = special.ellipk(m1, out=None)
    p1 = 2*km1/np.pi
    x2 = -np.cos(np.pi*m_ratio )
    m2 = math.sqrt((1-x2)/2) 
    km2 = special.ellipk(m2, out=None)
    p2 = 2*km2/np.pi
    p_factor = p1/p2
    freq = np.linspace(0.5E9, 1.5E9, 501)
    # freq_mhz = freq/1e6
    delta_v = - (eta**2)/2
    k = abs(e)*(eta+abs(e)/2)
    kb = -(abs(e)**2)*eta/(eta+2*abs(e))
    delta_b = -((eta**2)-2*((abs(e)**2)))/4
    omega = freq*2*np.pi
    delta = omega/vb - 2*np.pi/pI - 1j*alpha

    v_delta = []
    for i in range(0,len(delta)):
        v_delta_0 = eta_b/((cmath.sqrt(delta_b-delta[i]))+ eta_b)# wave velocity in m/s
        v_delta.append(v_delta_0)
    v_delta = np.array(v_delta)
    omega = freq*2*np.pi
    C = (W1*epsilon*p_factor)/pI ##To check
    xi = []
    for i in range(0,len(omega)):
        xi_0 = c*cmath.sqrt((omega[i]*C*k2)/(pI*np.pi))
        xi_0 = -1j*xi_0
        xi.append(xi_0)
    xi = np.array(xi)

    lam1 = pI # Wavelength in m of SAW filters 
    c12 = -1j*c*(k+kb*v_delta) # Reflectivity per unit length (~1.7% reflected per IDT spaced at lam/2)
    a1 = -xi # The transduction coefficient
    n1 = 100 # The number of IDT pairs
    L1 = n1*lam1 # Length of total IDT the grating, in m
    #W1 = 22*lam1 # Width of IDT (acoustic aperture), in m
    #d = sc.delta(freq,v1,lam1) - 500j
    Ct=n1*W1*epsilon # Static capacitance of total IDT
    #d1 = sc.delta(freq,v1,lam1)
    d1 = sc.thetau(c,delta,delta_v,kb,v_delta)
    C1 = sc.C0(freq,Ct)
    idt_ref_1 = sc.pmatrix(lam1,c12,a1,L1,d1,C1) #The P-Matrix of SAW resonator with refelection 
    # y11 = 20 * np.log10(abs(idt_ref_1.p33)/5)
    y11 = (idt_ref_1.p33)/5
    return y11


def YtoZS(Y_COM, freq):
    # Y11 = 20*np.log10(abs(Y_COM))
    Z_COM = 1/Y_COM
    z = 50 
    S11_COM = (1-Y_COM*z)/(1+Y_COM*z)

    group_delay = -np.diff(np.unwrap(np.angle(S11_COM))) / np.diff(2 * np.pi * freq)
    group_delay = np.concatenate(([group_delay[0]], group_delay))
    Q_COM = 2*np.pi*(freq)*group_delay *abs(S11_COM)/(1-abs(S11_COM)**2)

    # DSP_COM = 10*np.log10(1-abs(S11_COM)**2)
    return Y_COM.real, Y_COM.imag, Z_COM.real, S11_COM.real, S11_COM.imag, Q_COM


results = []
origin_paras = np.genfromtxt('G:/Zheng_caizhi/Pycharmprojects/SAW_tf/datas/out/MP60.csv', delimiter=',')
for i in range(0,len(origin_paras)):
    # x = np.array([eta,e,alpha,c,k2,npiezo_1,vb,m_ratio])
    # x = np.append(origin_paras[i], 0.6)
    freq = np.linspace(0.5E9, 1.5E9, 501)
    x = origin_paras[i]
    y = cacul(x)
    [Y0_R, Y0_I, Y0_Z, Y0_SR, Y0_SI, Y0_Q] = YtoZS(y, freq)
    Inputs = np.stack((Y0_R, Y0_I, Y0_Z, Y0_SR, Y0_SI, Y0_Q), axis=-1)
    results.append(Inputs)
    print(i)

result = np.array(results)
# file_path = 'D:\\data\\7p\\input2w/2w.npymusi.csv'
# mu = np.loadtxt(file_path)[:6]
# sigma = np.loadtxt(file_path)[6:]

mu = result.reshape(-1, 6).mean(axis=0)
sigma = result.reshape(-1, 6).std(axis=0)
result = (result - mu) / sigma
# if (sigma == 0).all() != True:
#     result = (result - mu )/ sigma
# else:
#     result = result - mu

file_path = 'G:/Zheng_caizhi/Pycharmprojects/SAW_tf/datas/input/6w.npy'
with open(file_path, 'wb') as f:
    np.save(f, result)
file_path1 = file_path + 'musi.csv'
with open(file_path1, 'w') as f:
    np.savetxt(f, mu)
    np.savetxt(f, sigma)
