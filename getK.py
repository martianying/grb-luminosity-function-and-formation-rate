#AIM: plotting the Luminosity-redshift data with flux limit with a changing k.



#-------------------IMPORT--------------------#
import numpy as np
from numpy import loadtxt
import matplotlib.pyplot as plt
import math
from astropy.cosmology import WMAP9 as cosmo
from astropy import constants as c
from astropy import units as u
from astropy.visualization import quantity_support 
import scipy.integrate as integrate
from pynverse import inversefunc

#-------------------PARAMETERS----------------#
#the speed of light
c=(c.c).to(u.centimeter/u.second)
#pi
Pi=math.pi
#Hubble constant in centimeter
H=cosmo.H(0).to(u.centimeter/(u.centimeter*u.second))
#flux limit give by the paper
FluxLimit=(2.0*10**-8)*u.erg/(u.centimeter**2*u.second)
Omiga=0.27
X=np.arange(0.1,10,0.01)
#we can change the value of k!!
k=0

#--------------------LOADING DATA-------------#
x=loadtxt('Desktop/DATA.txt',unpack=True,usecols=[0])
y=np.log10(loadtxt('Desktop/DATA.txt',unpack=True,usecols=[1]))
y1=loadtxt('Desktop/DATA.txt',unpack=True,usecols=[1])
y2=np.log10(loadtxt('Desktop/DATA.txt',unpack=True,usecols=[1])/(1+x)**k)

#-------------------FUNCTION for luminosity limit--------#
def Limit(value):
    trans=integrate.quad(lambda z:(1-Omiga+Omiga*(1+z)**3)**(-1/2),0,value)
    result1=(c/H)*(1+value)*list(trans)[0]
    result2=result1**2*4*Pi*FluxLimit
    result=result2.value/(1+value)**k
    return result #got the Luminosity limit of a specific redshit

ysome=[]
for i in range(len(X)):
    ysome=ysome+[Limit(X[i])]
#got the luminosity limit.
Y=np.log10(np.array([ysome]))

#-------------------PLOT THE Fig.1------------#
plt.scatter(X,Y,s=0.5)
plt.scatter(x,y2,s=6)
plt.grid(True)
plt.show()