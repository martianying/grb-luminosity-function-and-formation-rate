#AIM:get the k-eta plot.

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
#flux limit give by the paperv
FluxLimit=(2.0*10**-8)*u.erg/(u.centimeter**2*u.second)
Omiga=0.27
X=np.arange(0.1,10,0.01)
bound=0.757

#--------------------LOADING DATA-------------#
x=loadtxt('Desktop/swift_DATA_Yu.txt',unpack=True,usecols=[0])
y=np.log10(loadtxt('Desktop/swift_DATA_Yu.txt',unpack=True,usecols=[1]))
y1=loadtxt('Desktop/swift_DATA_Yu.txt',unpack=True,usecols=[1])

Eta=[]
k=np.arange(0,3.5,0.05)
for i in range(len(k)):
    y2=loadtxt('Desktop/swift_DATA_Yu.txt',unpack=True,usecols=[1])/(1+x)**k[i]

#-------------------FUNCTION for luminosity limit--------#

#Function ONE
#INPUT:redshift(scalar) OUTPUT:luminosity upper limit with this redshift(scalar)
    def Limit(value):
        trans=integrate.quad(lambda z:(1-Omiga+Omiga*(1+z)**3)**(-1/2),0,value)
        result1=(c/H)*(1+value)*list(trans)[0]
        result2=result1**2*4*Pi*FluxLimit
        result=result2.value/(1+value)**k[i]
        return result 

#Function TWO 
#INPUT:luminosity(scalar) OUTPUT:redshift upper limit with this luminosity(scalar)
    def inv(value):
        if value>=Limit(bound):
            function=(lambda m:Limit(m))
            z=inversefunc(function,y_values=value)
        else:
            z=0
            L=value
            while L-Limit(z)>=10**(-4):
                z=z+0.001
        return z


#------------------CALCULATE Eta--------------------#
    n=[]
    r=[]
    for t in range(len(x)):
        A=x[y2>=y2[t]]
        B=A[A<=inv(y2[t])]
        C=A[A<=x[t]]
        r=r+[len(C)]
        n=n+[len(B)]

#calculating the parameters involved in Eta
    N=np.array(n)-1
    E=(N+1)/2
    V=(N**2-1)/12
    R=np.array(r) 

    eta=sum((R-E).tolist())/math.sqrt(sum(V.tolist()))
    Eta=Eta+[eta]
plt.scatter(k,Eta,color='black')
plt.grid()
plt.xlabel('luminoisity-redshift Evolution Parameter k',fontsize = 12)
plt.xlim(0,3.5)
plt.ylabel('EP Method Statistical Value tau',fontsize = 12)
plt.title("(PL-CPL) Fig.2",fontsize = 12)
plt.show()