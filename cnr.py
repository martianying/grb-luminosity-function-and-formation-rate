#AIM：画出 differential cumulative redshift distribution
#Note: 这里没有画前面的文献中的图了，只花了最关键的原cumulative number redshift的图和它的微分图。

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
from matplotlib.pyplot import *
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
k=2.996
bound=0.757

#--------------------LOADING DATA-------------#
x=loadtxt('Desktop/swift_DATA.txt',unpack=True,usecols=[0])
y=np.log10(loadtxt('Desktop/swift_DATA.txt',unpack=True,usecols=[1]))
y1=loadtxt('Desktop/swift_DATA.txt',unpack=True,usecols=[1])
y2=loadtxt('Desktop/swift_DATA.txt',unpack=True,usecols=[1])/(1+x)**k

#-------------------FUNCTION for luminosity limit--------#

#Function ONE
#INPUT:redshift(scalar) OUTPUT:luminosity upper limit with this redshift(scalar)
def Limit(value):
    trans=integrate.quad(lambda z:(1-Omiga+Omiga*(1+z)**3)**(-1/2),0,value)
    result1=(c/H)*(1+value)*list(trans)[0]
    result2=result1**2*4*Pi*FluxLimit
    result=result2.value/(1+value)**k
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

#Function Three
#INPUT:sample data wanted to get the theoretical value, known x and y data used for estimating fit parameters, the degree
#OUTPUT:theoretical values of the sample data, cofficients of the fit equation.
def  fit(sample,In,Out,n):
    para=[]
    a=[0]*(n+1)
    b=[0]*(len(sample))
    result=np.array([b])
    for i in range(len(a)):
        a[i]=np.polyfit(In,Out,n)[n-i]
        para=para+[a[i]]
        Afit=a[i]*sample**i
        result=result+Afit
    return result,para
n_=3


#------------------CALCULATE Eta--------------------#
n=[]
r=[]
m=[]
for i in range(len(x)):
    A=x[y2>=y2[i]]
    B=A[A<=inv(y2[i])]
    C=A[A<=x[i]]
    D1=x[y2>=Limit(x[i])]
    D2=D1[D1<x[i]]
    r=r+[len(C)]
    n=n+[len(B)]
    m=m+[len(D2)]

#calculating the parameters involved in Eta

N=np.array(n)
M=np.array(m)

E=(N+1)/2
V=(N**2-1)/12
R=np.array(r) 

#----------------YONETOKU method-------------------#
eta=sum((R-E).tolist())/math.sqrt(sum(V.tolist()))

#-----------------------PLOT-----------------------#

#plotting the cumulative redshift number
Nlf=[]
Red=[]
for p in range(len(x)):
    red=1
    for q in range(len(x)):
        if x[p]>x[q]:
            if M[q]!=0:
                red=red*(1+1/M[q])
            else:
                red=red*1
        else:
            red=red*1
    Red=Red+[red]
     
hh=np.sort(Red/max(Red)) #1)


#plotting the differential number evolutions--redshift figure

#plotting the original data
plt.scatter(np.sort(x),hh,s=5,color='black',marker='+',label="GRBs sample")
#ploting the sample data and its theoretical values out of the fit equation
std=np.arange(0,10,0.01)
stdy=np.reshape(fit(std,np.sort(x),hh,n_)[0],len(std))
plt.plot(std,stdy,color='black',label="Polynomial Fit n=3")
plt.legend(loc="lower right")

#we need to get the coffients, more precisely, the fit equation to get the differential values at each point using Mathematica
cofficients=fit(std,np.sort(x),hh,n_)[1]; #there's no need to print them out but we need this command in case we need to check them later.

#we plot fit line and data above together
plt.xlabel("Redshift",fontsize=12)
plt.ylabel("Normalized Culmulative Formation Rate",fontsize=12)
plt.title("(266 samples)Fig.7",fontsize=12)
plt.show()

x_data=loadtxt('Desktop/DNE4.txt',unpack=True,usecols=[0])
y_data=loadtxt('Desktop/DNE4.txt',unpack=True,usecols=[1])
plt.plot(x_data,y_data,color='orange')
plt.xlabel("redshift z")
plt.ylabel("Differential Number Evolution")
plt.show()

#now we are tring to get the formation rate
#----------------define a function----------------#
#INPUT=redshift
#OUTPUT=(dV(z)/dz)**-1
def dif(value):
    trans=integrate.quad(lambda z:(1-Omiga+Omiga*(1+z)**3)**(-1/2),0,value)
    result1=4*Pi*(c/H)**3*(list(trans)[0])**2
    result=1/math.sqrt(1-Omiga+Omiga*(1+value)**3)
    out=1/(result*result1).value
    return out
#calculating the dif value of the group x_data_
later=[]
x_data_=x_data[1:]
y_data_=y_data[1:]
for i in range(len(x_data_)):
    later=later+[dif(x_data_[i])]
#getting the formation rate value
later_=y_data_*(1+x_data_)*later
ax = subplot(111)
ax.set_yscale('log')
ax.set_xscale('log')

plt.plot(x_data_+1,later_,color='black')
plt.title("(266 samples)Fig.8 Formation Rate of GRBs",fontsize = 12)
plt.xlabel("log (z+1)",fontsize = 12)
plt.ylabel("log Formation Rate",fontsize = 12)

plt.show()
