import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad # così non importa scriversi dopo nel testo scipy.integrate.quad

## Function to calculate chi
def chiCalc_custom(funcNoise,para,totT,pulses_times,sP=False,sP4debug=False):
    
    """
    Function to calculate chi
    funcNoise : NSD function of the type f(x,*para)
    para : parameters of the NSD function
    totT : total time for which we want to calculate chi
    pulses_times : array with the time at which pi-pulses are applied
    Returns
        chi(totT)
    """
    if sP:
        xx = np.linspace(0.1,0.9,1000)*2*np.pi
        plt.figure(figsize = (6,2))
        plt.plot(xx,funcNoise(xx,*para) )
        plt.xlabel(r'$\omega\; [MHz]$',fontsize=18)
        plt.ylabel(r'$S(\omega)\; [MHz]$',fontsize=18)
        plt.show()

    n = len(pulses_times) #  Number of pulses
    distPi = pulses_times/(totT) #  define a normalized vector \in[0,1] with the spacing between pi-pulses

    if n == 1:
        def Fn(ω,t):
            return 2*8*np.sin(ω*t/4/n)**4
    else:
        def Fn(ω,t):
            return 2*8*(np.sin(ω*t/2)*np.sin(ω*t/4/n)**2/np.cos(ω*t/2/n))**2
    
    # If-cycle thought for debug purposes 
    if sP4debug: # To see Fn in case something might be wrong
        #t0 = 1; dt = 1; tf = 180;
        #tt=np.arange(t0,tf,dt)
        #plt.plot(tt,Fn(0.5,tt)) # feel free to change ω (first arg of Fn)
        xx=np.arange(0,180,0.2)
        plt.plot(xx,Fn(1,xx))
        plt.plot(xx,Fn_bis(1,xx))
        #plt.plot(xx,np.sqrt(Fn(xx,30)))
        #plt.ylim(0,50)
        plt.title('Fn',fontsize=16)
        plt.grid()
        plt.show()

    # Define the coherence function
    def Chi(t):
        res, err = np.abs(quad( lambda x: np.abs(funcNoise(x,*para)*Fn(x,t)/(np.pi*x**2)), 0.001, 8.5)) # integrate from ω=0.001 to ω=8.5
        return res
    
    # If-cycle thought for debug purposes 
    if sP4debug:
        ww=np.arange(0.01, 8.5+0.01,0.01)
        tt=2
        plt.plot(ww,np.abs(funcNoise(ww,*para)*Fn(ww,tt)/(np.pi*ww**2)))
        plt.xlabel(r'$\omega/2\pi$ [$MHz$]',fontsize=20)
        plt.title('Integrand of $\chi$',fontsize=18)
        plt.show()
    ######## Here is the end of the definitions ########

    ######## Now we calculate ########
    return Chi(totT)
