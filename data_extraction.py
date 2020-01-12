#------------------------IMPORT------------------------#
import pandas as pd
import numpy as np
from numpy import loadtxt
import math
from astropy.cosmology import WMAP9 as cosmo
from astropy import constants as c
from astropy import units as u
from astropy.visualization import quantity_support 
import scipy.integrate as integrate
import matplotlib.pyplot as plt

#------------------------stat_PL_info & stat_CPL_info----------------

#把1s peak best model的文件中的GRBs names(string)提取出来
f1=open('Desktop/1s_peak_best_model.txt')
lines = f1.readlines()  # return a list of lines in file
f1_data = []
for line in lines:
    f1_data += line.split()
f1.close()
GRBs=[f1_data[i] for i in np.arange(0,len(f1_data),3)]
#extracting the best fit model type in the txt file.
GRBs_best_fit=[f1_data[i] for i in np.arange(2,len(f1_data),3)]
#得到所有GRB的名字和对应最佳model的表格
best_fit_sea={'best fit':GRBs_best_fit}
DF1=pd.DataFrame(data=best_fit_sea,index=GRBs)

#得到存在已知比较精确GRBs redshift所对应的GRB name。loc_red
f2=open('Desktop/redshift_sea.txt')
lines1 = f2.readlines()  # return a list of lines in file
f2_data = []
for line in lines1:
    f2_data += line.split()
f2.close()
loc_red=[f2_data[i] for i in np.arange(0,len(f2_data),2)]

#------------------------------------------------------------------------
#得到best fit是分别是CPL和PL的表格，表格包含GRB name和best fit type
new_DF1=DF1.loc[loc_red]
fit_CPL=new_DF1[new_DF1['best fit']=='CPL']
fit_PL=new_DF1[new_DF1['best fit']=='PL']
#---------------------------------------------------------------------------

#得到PL model的参数（name--alpha--norm）
f3=open('Desktop/PL_para_sum.txt')
lines = f3.readlines()  # return a list of lines in file
f3_data = []
for line in lines:
    f3_data += line.split()
f3.close()
PL_name_sum=[f3_data[i] for i in np.arange(0,len(f3_data),16)]
PL_alpha_sum=[f3_data[i] for i in np.arange(2,len(f3_data),16)]
PL_norm_sum=[f3_data[i] for i in np.arange(5,len(f3_data),16)]
PL_sum={'alpha':PL_alpha_sum,'norm':PL_norm_sum}
DF2=pd.DataFrame(data=PL_sum,index=PL_name_sum)
#得到best fit is PL model的GRB的参数
DF2_asis=DF2.loc[fit_PL.index]

#整理后得到的best fit model is PL的peak flux
PL_15_150_peak_flux=loadtxt('Desktop/PL_peak_flux.py',unpack=True,usecols=[1])
Fp150={'Fp150':PL_15_150_peak_flux}
DF3=pd.DataFrame(data=Fp150,index=fit_PL.index)

#得到best fit是PL的GRB的红移
f5=open('Desktop/redshift_sea.txt')
lines = f5.readlines()  # return a list of lines in file
f5_data = []
for line in lines:
    f5_data += line.split()
f5.close()
redshift=[f5_data[i] for i in np.arange(1,len(f5_data),2)]
redshift_massage={'redshift':redshift}
DF4=pd.DataFrame(data=redshift_massage,index=loc_red).loc[fit_PL.index]

#将前面所有的数据合在一起
stat_PL=pd.concat([DF4,fit_PL,DF2_asis, DF3], axis=1)

#得到CPL model的参数（name--alpha--norm-Epeak）
f6=open('Desktop/CPL_para_sum.txt')
lines = f6.readlines()  # return a list of lines in file
f6_data = []
for line in lines:
    f6_data += line.split()
f6.close()
CPL_name_sum=[f6_data[i] for i in np.arange(0,len(f6_data),19)]
CPL_alpha_sum=[f6_data[i] for i in np.arange(2,len(f6_data),19)]
CPL_Epeak_sum=[f6_data[i] for i in np.arange(5,len(f6_data),19)]
CPL_norm_sum=[f6_data[i] for i in np.arange(8,len(f6_data),19)]
CPL_sum={'alpha':PL_alpha_sum,'norm':PL_norm_sum,'Epeak':CPL_Epeak_sum}
DF5=pd.DataFrame(data=CPL_sum,index=CPL_name_sum)
#得到best fit is CPL model的GRB的参数
DF5_asis=DF5.loc[fit_CPL.index]
#得到best fit是CPL的GRB的红移
DF6=pd.DataFrame(data=redshift_massage,index=loc_red).loc[fit_CPL.index]
#带入photon flux peak
f7=open('Desktop/1s_peak_photon_CPL_flux.txt')
lines = f7.readlines()  # return a list of lines in file
f7_data = []
for line in lines:
    f7_data += line.split()
f7.close()
CPL_photon_flux=[f7_data[i] for i in np.arange(17,len(f7_data),26)]
CPL_Fp={'Fp150':CPL_photon_flux}
DF7=pd.DataFrame(data=CPL_Fp,index=CPL_name_sum)
DF7_asis=DF7.loc[fit_CPL.index]
#将前面所有的数据合在一起
stat_CPL=pd.concat([DF6,DF5_asis,DF7_asis], axis=1)

#----------------------bolometric peak luminosity calculation---------------

#the speed of light
c=(c.c).to(u.centimeter/u.second)
#pi
Pi=math.pi
#Hubble constant in centimeter
H=cosmo.H(0).to(u.centimeter/(u.centimeter*u.second))
Omiga=0.27
enorm=50

#function1 
#PL model energy spectrum
def k_correct_PL(norm,alpha,redshift):
    def N(value):
        return norm*(value/enorm)**alpha
    int_A=integrate.quad(lambda E:E*N(E),1/(1+redshift),10**4/(1+redshift))
    #int_A=integrate.quad(lambda E:E*N(E),15*(1+redshift),150*(1+redshift))
    int_B=integrate.quad(lambda E:N(E),15,150)
    return int_A[0]/int_B[0]
#function2
#CPL model energy spectrum
def k_correct_CPL(norm,alpha,redshift,Epeak):
    def N(value):
        return norm*(value/enorm)**alpha*math.exp(-(2+alpha)*value/Epeak)
    int_A=integrate.quad(lambda E:E*N(E),1/(1+redshift),10**4/(1+redshift))
    #int_A=integrate.quad(lambda E:E*N(E),15*(1+redshift),150*(1+redshift))
    int_B=integrate.quad(lambda E:N(E),15,150)
    return int_A[0]/int_B[0]
#function3
#bolometric luminosity
def Bolometric(redshift,peak_flux,k_correction):
    trans=integrate.quad(lambda z:(1-Omiga+Omiga*(1+z)**3)**(-1/2),0,redshift)
    result1=(c/H)*(1+redshift)*list(trans)[0]
    result2=result1**2*4*Pi*peak_flux*k_correction*u.erg/(u.centimeter**2*u.second)
    result=result2.value
    return result
#calculation for PL model
PL_para1=pd.to_numeric(stat_PL['norm'])
PL_para2=pd.to_numeric(stat_PL['alpha'])
PL_para3=pd.to_numeric(stat_PL['redshift'])
PL_para4=pd.to_numeric(stat_PL['Fp150'])

PL_para_K=[k_correct_PL(PL_para1[i],PL_para2[i],PL_para3[i]) for i in range(len(stat_PL))]
PL_para_L=[Bolometric(PL_para3[i],PL_para4[i],PL_para_K[i])*1.602*10**-9 for i in range(len(stat_PL))]
PL_para_k={'k correction':PL_para_K,'Luminosity':PL_para_L}
DF5=pd.DataFrame(data=PL_para_k,index=stat_PL.index)

#getting GRBs parameter table whose best fit model is PL
stat_PL_info=pd.concat([stat_PL,DF5], axis=1)

#calculation for CPL model
CPL_para1=pd.to_numeric(stat_CPL['norm'])
CPL_para2=pd.to_numeric(stat_CPL['alpha'])
CPL_para3=pd.to_numeric(stat_CPL['redshift'])
CPL_para4=pd.to_numeric(stat_CPL['Fp150'])
CPL_para5=pd.to_numeric(stat_CPL['Epeak'])
CPL_para_K=[k_correct_CPL(CPL_para1[i],CPL_para2[i],CPL_para3[i],CPL_para5[i]) for i in range(len(stat_CPL))]
CPL_para_L=[Bolometric(CPL_para3[i],CPL_para4[i],CPL_para_K[i])*1.602*10**-9 for i in range(len(stat_CPL))]
CPL_para_k={'k correction':CPL_para_K,'Luminosity':CPL_para_L}
DF8=pd.DataFrame(data=CPL_para_k,index=stat_CPL.index)

#getting GRBs parameter table whose best fit model is CPL
stat_CPL_info=pd.concat([stat_CPL,DF8], axis=1)

CPL_Luminosity=pd.to_numeric(stat_CPL_info['Luminosity'])
PL_Luminosity=pd.to_numeric(stat_PL_info['Luminosity'])
f=open('Desktop/swift_DATA.txt',"w")
for i in range(len(CPL_Luminosity)):
    f.write("%f\t %f\t \n"%(CPL_para3[i],CPL_Luminosity[i]))
for i in range(len(PL_Luminosity)):
    f.write("%f\t %f\t \n"%(PL_para3[i],PL_Luminosity[i]))
f.close
g=open('Desktop/swift_DATA.txt',"r")
if g.mode=='r':
    contents=g.read()

x=loadtxt('Desktop/swift_DATA.txt',unpack=True,usecols=[0])
y=np.log10(loadtxt('Desktop/swift_DATA.txt',unpack=True,usecols=[1]))
FluxLimit=(2.0*10**-8)*u.erg/(u.centimeter**2*u.second)
Omiga=0.27
X=np.arange(0.1,10,0.01)
#-------------------FUNCTION for luminosity limit--------#
def Llimit(value):
    trans=integrate.quad(lambda z:(1-Omiga+Omiga*(1+z)**3)**(-1/2),0,value)
    #trans=integrate.quad(lambda z:(1-Omiga+Omiga*(1+z)**3)**(-1/2),0,value)
    result1=(c/H)*(1+value)*list(trans)[0]
    #result1=(c/H)*list(trans)[0]
    result2=result1**2*4*Pi*FluxLimit
    result=result2.value
    return result #got the Luminosity limit of a specific redshit
ysome=[]
for i in range(len(X)):
    ysome=ysome+[Llimit(X[i])]
#got the luminosity limit.
Y=np.log10(np.array([ysome]))

#-------------------PLOT THE Fig.1------------#
plt.scatter(X,Y,s=0.5)
plt.scatter(x,y,s=6)
plt.ylim(48,55)
plt.show()


