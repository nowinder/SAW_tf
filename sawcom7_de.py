from math import cos
import numpy as np
import scipy.optimize as op
'''
Solutions to Coupling of modes equations as P-matrix elements
Reproduced from "Surface Acoustic Wave Filters" by David Morgan, 2nd edition
Chapter 8, page 245

'''
sin_exp_values = []
sin_BB =[]
cos_exp_values = []
cos_BB =[]
def sin_d(x):
    a = np.real(x)
    b = np.imag(x)
    sin_exp_values.append(2*np.exp(-b))
    sin_BB.append(b)
    return np.divide(1, 2*np.exp(-b)+1e-30)*(np.sin(a)*(1+np.exp(-2*b)) + 1j*(np.cos(a)*(1-np.exp(-2*b))))
    # m = np.abs(x)
    # phi = np.angle(x)
    # ee = m*np.exp(phi*1j)
    # return (np.exp(ee*1j)-np.exp(-ee*1j))/(2j)

def cos_d(x):
    a = np.real(x)
    b = np.imag(x)
    cos_exp_values.append(2*np.exp(-b))
    cos_BB.append(b)
    return np.divide(1, 2*np.exp(-b)+1e-30)*(np.cos(a)*(1+np.exp(-2*b)) - 1j*(np.sin(a)*(1-np.exp(-2*b))))
    # m = np.abs(x)
    # phi = np.angle(x)
    # ee = m*np.exp(phi*1j)
    # return (np.exp(ee*1j)+np.exp(-ee*1j))/2


def thetau(c,delta,delta_v,kb,v_delta):
    #returns the detuning wavenumber given a list of frequencies, a SAW velocity, and the lithographically
    #defined SAW wavelength
    theta = c*(delta-delta_v+abs(kb)*v_delta)
    return theta


#def delta(freq,v,lam):
    #returns the detuning wavenumber given a list of frequencies, a SAW velocity, and the lithographically
    #defined SAW wavelength
    #omega_0 = 2*np.pi*v/lam
    #omega = freq*2*np.pi
    #return (omega-omega_0)/v

def C0(freq,Ct):
    omega = freq*2*np.pi
    return 1j*omega*Ct

def K1(a1,c12, delta):
    # Returns particular integral solution 1
    s = np.sqrt(delta**2 - np.abs(c12)**2 + 0j)
    return (np.conj(a1)*c12 - 1j*delta*a1)/(delta**2 - np.abs(c12)**2)


def K2(a1,c12, delta):
    # Returns particular integral solution 2 
    s = np.sqrt(delta**2 - np.abs(c12)**2 + 0j)
    return (a1*np.conj(c12) + 1j*delta*np.conj(a1))/((delta**2 - np.abs(c12)**2))

def p11(c12,L,delta):
    s = np.sqrt(delta**2 - np.abs(c12**2) + 0j)
    # s = np.around(s,6)
    # L = np.around(L,6)
    a = s*L
    # a = a.real%(2*np.pi)+1j*(a.imag%(2*np.pi))
    # s =  np.real(s)*np.sign(delta) + 1j*np.imag(s)
    # return -np.conj(c12)*sin_d(a)/(s*np.cos(a) + 1j*delta*sin_d(a))
    return -np.conj(c12)*sin_d(a)/(s*cos_d(a) + 1j*delta*sin_d(a))

def p22(c12,L,delta, lam):
    s = np.sqrt(delta**2 - np.abs(c12**2) + 0j)
    kc = 2*np.pi/lam
    # s = np.around(s,6)
    # L = np.around(L,6)
    a = s*L
    # a = a.real%(2*np.pi)+1j*(a.imag%(2*np.pi))
    # return c12*sin_d(a)*np.exp(-2j*kc*L)/(s*cos_d(a) + 1j*delta*sin_d(a))
    return c12*sin_d(a)*np.exp(-2j*kc*L)/(s*cos_d(a) + 1j*delta*sin_d(a))

def p12(c12, L, delta, lam):
    s = np.sqrt(delta**2 - np.abs(c12**2) + 0j)
    kc = 2*np.pi/lam
    # s = np.around(s,6)
    # L = np.around(L,6)
    a = s*L
    # a = a.real%(2*np.pi)+1j*(a.imag%(2*np.pi))
    return s*np.exp(-1j*kc*L)/(s*cos_d(a) + 1j*delta*sin_d(a))

def p31(c12,a1,L, delta):
    K_2 = K2(a1,c12,delta)
    s = np.sqrt(delta**2 - np.abs(c12)**2 + 0j)
    # s = np.around(s,6)
    # L = np.around(L,6)
    a = s*L
    # a = a.real%(2*np.pi)+1j*(a.imag%(2*np.pi))
    P_31 = (2*np.conj(a1)*sin_d(a)- 2*s*K_2*(cos_d(a)-1))/(s*cos_d(a) + 1j*delta*sin_d(a))
    return P_31

def p32(c12,a1,L,delta,lam):
    K_1 = K1(a1,c12,delta)
    kc = 2*np.pi/lam
    s = np.sqrt(delta**2 - np.abs(c12)**2 + 0j)
    # s = np.around(s,6)
    # L = np.around(L,6)
    a = s*L
    # a = a.real%(2*np.pi)+1j*(a.imag%(2*np.pi))
    P_32 = np.exp(-1j*kc*L)*(-2*a1*sin_d(a)- 2*s*K_1*(cos_d(a)-1))/(s*cos_d(a) + 1j*delta*sin_d(a))
    return P_32

def p33(c12,a1,L,delta,lam,C0):
    K_1 = K1(a1,c12,delta)
    K_2 = K2(a1,c12,delta)
    kc = 2*np.pi/lam
    P_33 = -K_1*p31(c12,a1,L, delta) - K_2*p32(c12,a1,L,delta,lam)*np.exp(1j*kc*L)+ 2*(np.conj(a1)*K_1 - a1*K_2)*L+C0
    return P_33, sin_exp_values, sin_BB, cos_exp_values, cos_BB



'''
Define a P-matrix data structure, and a function for concatenating of two P-matrices
'''


class pmatrix:
    def __init__(self,lam, c12, a1, L, delta, C0):
        self.p11 = p11(c12, L, delta)
        self.p22 = p22(c12, L, delta, lam)
        self.p12 = p12(c12, L, delta, lam)
        self.p21 = self.p12
        self.p31 = p31(c12,a1,L, delta)
        self.p13 = self.p31/(-2)
        self.p32 = p32(c12,a1,L,delta,lam)
        self.p23 = self.p32/(-2)       
        self.p33 = p33(c12,a1,L,delta,lam,C0)
        
    @classmethod
    def blank(cls):
        #Return a blank p-matrix
        n = np.array([])
        return cls(n, n, n, n, n, n)
        

class tmatrix:
    def __init__(self,p):
        self.t11 = p.p12-p.p22*p.p11/p.p12
        self.t12 = p.p22/p.p12
        self.t13 = p.p32/(-2) - p.p22*(p.p31/(-2))/p.p12 
        self.t21 = -p.p11/p.p12
        self.t22 = 1/p.p12
        self.t23 = -(p.p31/(-2))/p.p12
        self.t31 = p.p31 - p.p32*p.p11/p.p12
        self.t32 = p.p32/p.p12
        self.t33 = p.p33 - p.p32*(p.p31/(-2))/p.p12
        

        
def concat(pl,pr):
    new = pmatrix.blank()
    D = 1-pl.p22*pr.p11
    new.p11 = pl.p11 + pr.p11*pl.p12**2/D
    new.p12 = pr.p12*pl.p12/D
    new.p21 = new.p12
    new.p22 = pr.p22 + pl.p22*pr.p12**2/D
    new.p13 = pl.p13 + pl.p12*(pr.p11*pl.p23 + pr.p13)/D
    new.p31 = -2*new.p13
    new.p23 = pr.p23 + pr.p12*(pr.p13*pl.p22 + pl.p23)/D
    new.p32 = -2*new.p23
    new.p33 = pl.p33 + pr.p33 - 2*pl.p23*(pr.p11*pl.p23 + pr.p13)/D - 2*pr.p13*(pl.p22*pr.p13 + pl.p23)/D
    return new


def a1_finder(v, lam, ksquared, c_s, w):
    '''
    Takes in SAW velocity (in m/s), SAW wavelength (in m), ksquared
    IDT capacitance per unit length (in pF/cm) and IDT overlap (in m)
    Spits out an approximation for the piezoelectric conversion constant a1.
    '''
    
    c_s = c_s/1E10 # Convert C_s to F/m
    omega = 2*np.pi*v/lam
    n_p = np.linspace(10,100,91) # create an array for n_p and minimize over array
    if type(w) != np.ndarray:
        w = np.array([w])
    a1 = np.zeros(len(w))
    
    for i in range(len(w)):
        # Define the cost function
        G_alg = 1.3*ksquared*n_p**2*omega*w[i]*c_s
        def f(x):
            return np.sum((G_alg - p33(0,x,lam*n_p,1,lam).real)**2)
        # Minimize the cost function
        a1[i] = float(op.minimize(f, 500, method = 'Powell').x)
    return 1j*a1


