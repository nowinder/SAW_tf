# %load ../PackClass.py
import matplotlib.pyplot as plt
import numpy as np
# from data_gene_ori import sample
# from data_genev4 import sample
class Plot_all:
    def __init__(self, Y_COM, Y_DL, freq, name1,name2='normalizated', path=None, choice=False, yzsqd1 = [], yzsqd2 = []):
        self.Y_COM = Y_COM
        self.Y_DL = Y_DL
        self.freq = freq
        self.name1 = name1
        self.name2 = name2
        self.path = path
        if choice == False:
            self.y1, self.z1, self.s1r, self.s1i, self.q1, self.d1 = self.YtoZS(Y_COM, freq)
            self.y2, self.z2, self.s2r, self.s2i, self.q2, self.d2 = self.YtoZS(Y_DL, freq)
        else:
            self.y1 = yzsqd1[0]
            self.z1 = yzsqd1[1]
            self.s1r = yzsqd1[2]
            self.s1i = yzsqd1[3]
            self.q1 = yzsqd1[4]
            self.d1 = yzsqd1[5]
            self.y2 = yzsqd2[0]
            self.z2 = yzsqd2[1]
            self.s2r = yzsqd2[2]
            self.s2i = yzsqd2[3]
            self.q2 = yzsqd2[4]
            self.d2 = yzsqd2[5]
    
    # def gene_teset(freq, Y_COM, path):
    #     Y11 = 20*np.log10(abs(Y_COM))
    #     Y_com_fre = np.stack((Y11, freq), axis = -1)
    #     Y11_mag = sample(Y_com_fre)
    #     Z_COM = 1/Y_COM
    #     Z_com_fre = np.stack((Z_COM.real, freq), axis=-1)
    #     Z11_real = sample(Z_com_fre)
    #     z = 50 
    #     S11_COM = (1-Y_COM*z)/(1+Y_COM*z)
    #     S_real_fre = np.stack((S11_COM.real, freq), axis=-1)
    #     S11_real = sample(S_real_fre)
    #     S_imag_fre = np.stack((S11_COM.imag, freq), axis=-1)
    #     S11_imag = sample(S_imag_fre)
    #     Inputs = np.stack((Y11_mag, Z11_real, S11_real, S11_imag), axis=-1)
    #     # 归一化
    #     ima = np.genfromtxt(path)[:2]
    #     imi = np.genfromtxt(path)[2:4]
    #     Inputs = (Inputs-imi)/(ima - imi)
    #     return Inputs
    
    def gene_teset2(Y_COM, path):
        Y11 = 20*np.log10(abs(Y_COM))
        Z_COM = 1/Y_COM
        Z11 = 20*np.log10(abs(Z_COM.real))
        z = 50 
        S11_COM = (1-Y_COM*z)/(1+Y_COM*z)
        DSP_COM = 10*np.log10(1-abs(S11_COM)**2)
        Inputs = np.stack((Y11, Z11, S11_COM.real, S11_COM.imag, DSP_COM), axis=-1)
        # 归一化
        ima = np.genfromtxt(path)[:5]
        imi = np.genfromtxt(path)[5:10]
        Inputs = (Inputs-imi)/(ima - imi)
        return Inputs

    def YtoZS(self, Y_COM, freq):
        Y11 = 20*np.log10(abs(Y_COM))
        Z_COM = 1/Y_COM
        z = 50 
        S11_COM = (1-Y_COM*z)/(1+Y_COM*z)

        group_delay = -np.diff(np.unwrap(np.angle(S11_COM))) / np.diff(2 * np.pi * freq)
        group_delay = np.concatenate(([group_delay[0]], group_delay))
        Q_COM = 2*np.pi*(freq)*group_delay *abs(S11_COM)/(1-abs(S11_COM)**2)

        DSP_COM = 10*np.log10(1-abs(S11_COM)**2)
        return Y11, Z_COM.real, S11_COM.real, S11_COM.imag, Q_COM, DSP_COM

    def plot_all(self):
        # y1,z1,s1r,s1i,q1,d1 = YtoZS(Y_COM, freq)
        # y2,z2,s2r,s2i,q2,d2 = YtoZS(Y_DL, freq)
        
        plt.figure(figsize=[16, 12])
        plt.subplot(3,2,1)
        self.plot_YDB()
        plt.subplot(3,2,2)
        self.plot_ZR()
        plt.subplot(3,2,3)
        self.plot_SR()
        plt.subplot(3,2,4)
        self.plot_SI()
        plt.subplot(3,2,5)
        self.plot_Q()
        plt.subplot(3,2,6)
        self.plot_DSP()

    def plot_YDB(self):
        plt.plot(self.freq,self.y1,label = self.name1)
        plt.plot(self.freq,self.y2,label = self.name2)
        plt.ylabel('Y_real/log')
        # plt.title('Absolute Value of Admittance (db)/ Y_R')
        plt.title('Impedance (Real part)')
        plt.grid('on')
        plt.legend()    
    def plot_ZR(self):
        plt.plot(self.freq,self.z1,label = self.name1)
        plt.plot(self.freq,self.z2,label = self.name2)
        plt.ylabel('Y_imag/log')
        # plt.yscale('log')
        plt.title('Impedance (Imaginary part)')
        plt.grid('on')
        plt.legend()
    def plot_SR(self):
        plt.plot(self.freq,self.s1r,label = self.name1)
        plt.plot(self.freq,self.s2r,label = self.name2)
        plt.ylabel('absolute value')
        plt.title('S11 real part / Z_R')
        # plt.yscale('log')
        plt.grid('on')
        plt.legend()        
    def plot_SI(self):
        plt.plot(self.freq,self.s1i,label = self.name1)
        plt.plot(self.freq,self.s2i,label = self.name2)
        plt.ylabel('absolute value')
        plt.title('S11 imaginary part / S_R')
        plt.grid('on')
        plt.legend()
    def plot_Q(self):
        plt.plot(self.freq,self.q1,label = self.name1)
        plt.plot(self.freq,self.q2,label = self.name2)
        plt.ylabel('absolute value')
        plt.title('Q value / S_I')
        plt.grid('on')
        plt.legend()        
    def plot_DSP(self):
        plt.plot(self.freq,self.d1,label = self.name1)
        plt.plot(self.freq,self.d2,label = self.name2)
        plt.ylabel('absolute value')
        plt.title('DSP / Q')
        plt.yscale('log')
        plt.grid('on')
        plt.legend()