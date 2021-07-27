# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 22:25:22 2017

@author: CYarman
"""

#from temp import *

import numpy as np
import pylab as plt
import scipy as sp
import scipy.special
import time


#==============================================================================
# sinc IN TERMS OF GAUSSIANS
#==============================================================================
def h_n_sinc_in_Gaussian(n):
#    f_n = np.divide(1.j**n/sp.special.factorial(n+1)) # TAYLOR SERIES EXPANSIONS OF sinc+1i*cosinc
#    g_n = np.divide(1.j**n/sp.special.gamma((n+2)/2) # TAYLOR SERIES EXPANSIONS OF exp(-x^2)+1i*2*Dawson/s\qrt(\pi)
    h_n = np.divide(sp.special.gamma((n+2)*0.5),sp.special.factorial(n+1))
    return h_n

#==============================================================================
# SOLVING MOMENT PROBLEM USING BALANCED TRUNCATION
#==============================================================================
def YF_quadratures(h_n,N,B):

    p_hankel = lambda v,K: \
        sp.linalg.hankel(v[0:K:],v[K-1::]) # RECTANGULAR HANKEL MATRIX        

    h = h_n(np.arange(N))*(B**np.arange(N))
    H = p_hankel(h,int(np.floor(N*0.5)+1))
    UH,SH,VH = np.linalg.svd(H,full_matrices=False)
    tol = 2e-15 # tolerance for balanced truncation - SHALL BE CHOSEN BASED ON NOISE LEVEL ON h_n
    Trank = np.unique(SH/SH[0]<tol,return_index = True)[1] 
    Trank = Trank[Trank.size-1]-1 # INDEX FOR BALANCED TRUNCATION
    Srh = np.diag(SH[0:Trank+1]**0.5)
    b = np.dot(Srh,VH[0:Trank+1,0])
    c = np.dot(UH[0,0:Trank+1],Srh)
    A = np.dot(np.linalg.pinv(np.dot(UH[:(UH.shape[0]-1),0:Trank+1],Srh)),np.dot(UH[1::,0:Trank+1],Srh))
    D,V = np.linalg.eig(A)
    bt = np.dot(np.linalg.pinv(V),b)
    ct = np.dot(c,V)
    omega_m = D
    
    alpha_m = ct*bt 
    ' INSTEAD OF THIS ONE MAY USE LEAST SQUARE FIT OF h_n WITH alpha_m '
    
    omega_m = omega_m/B

    return alpha_m,omega_m


#==============================================================================
# APPROXIMATING f(x) IN TERMS OF g(x): f(x) = \sum_m \alpha_m g(\gamma_m x)
#==============================================================================
def approx_f_in_g(g,alpha_m,gamma_m):
    approx_f = lambda x:     np.reshape(g(np.outer(x.flatten(),gamma_m)).dot(alpha_m),x.shape)
    return approx_f


#==============================================================================
# CALCULATING k_tilde - OUR OWN ASYMPTOTICS
#==============================================================================

# \tilde{k}(\xi) = int exp(1j*omega*x^2) *\tilde\chi_{[-1,1]}(x)*exp(1j*\xi*x)
def k_tilde(xi,omega , B = np.pi):
    alpha_m, gamma_m = YF_quadratures(h_n_sinc_in_Gaussian,101,B)

    print(gamma_m.shape)
    
    'TESTING SINC APPROXIMATION'
    '''
    g = lambda x: np.exp(-x**2) + 1j*2./np.sqrt(np.pi)*sp.special.dawsn(x)
    sinc = lambda x: np.sin(x)/(x + (x==0)) + (x==0)
    x_temp = np.linspace(-50,50,2001)
#    plt.plot(x_temp,g(x_temp/(6**.5)),'b',x_temp,sinc(x_temp),'r--')
    plt.subplot(2,1,1).plot(x_temp,np.exp(-np.outer(x_temp,gamma_m)**2).dot(alpha_m),x_temp,sinc(x_temp),'--')
    plt.subplot(2,1,2).plot(x_temp,np.exp(-np.outer(x_temp,gamma_m)**2).dot(alpha_m)-sinc(x_temp))
    plt.show()
    eB2 = lambda B,alpha_m,gamma_m: 1/(4*B) \
        - 2/(4*B)*(alpha_m.dot(sp.special.erf(B*sp.pi/sp.sqrt(gamma_m)))).real \
        + sp.sqrt(sp.pi)/2*((alpha_m[:,None]*alpha_m[None,:]).flatten()).dot(1/sp.sqrt(gamma_m[:,None]+gamma_m[None,:]).flatten())
    eB2(1/(2*sp.pi),alpha_m,gamma_m**2) # 
    '''
    alpha_m = alpha_m/gamma_m/np.sqrt(np.pi)
    gamma_m = 1./(2.*gamma_m)
    gamma_m = -gamma_m**2
 
    'TESTING CHARACTERISTIC APPROXIMATION'
    '''x_temp = np.linspace(-2,2,2001)
    chi = lambda x: np.abs(x)<=1.
    plt.plot(x_temp,np.exp(np.outer(x_temp**2,gamma_m)).dot(alpha_m),'b',x_temp,chi(x_temp),'r--')
    plt.show()
    '''

    term_amp =  alpha_m*(-np.pi/(1j*omega+gamma_m))**.5
    term_exp =  1/(4.*(1j*omega+gamma_m))

    k_tilde_temp = np.exp(np.outer(xi.flatten()**2,term_exp)).dot(term_amp)
    
    return k_tilde_temp


#==============================================================================
# CALCULATING k_hat - EQUATION (3.5) FROM Böttcher, Brunner, Iserles, Norsett 2003
#==============================================================================

def k_hat(xi,omega,epsilon):
    temp = (np.pi*(epsilon+1j)/(1+epsilon**2))**0.5  \
    *np.exp(-epsilon*xi**2/(4*(1+epsilon**2))) \
    *np.exp(-1j*xi**2/(4*(1+epsilon**2)))

    return temp


#==============================================================================
# CALCULATING A_mn_tilde - OUR MATRIX REPRESENTATION OF THE OPERATOR
#==============================================================================

def A_mn_tilde(omega = 5,m = np.arange(0,101),n = np.arange(0,101), B = 4.5):
    alpha_m, gamma_m = YF_quadratures(h_n_sinc_in_Gaussian,101,B)
#    g = lambda x: np.exp(-x**2) + 1j*2./np.sqrt(np.pi)*sp.special.dawsn(x)
    alpha_m = alpha_m/gamma_m/np.sqrt(np.pi)
    gamma_m = 1./(2.*gamma_m)
    gamma_m = -gamma_m**2

    Alpha_k = alpha_m[:,None,None,None]
    Alpha_l = alpha_m[None,:,None,None]
    Gamma_k = gamma_m[:,None,None,None]
    Gamma_l = gamma_m[None,:,None,None]
    M = m[None,None,:,None]
    N = n[None,None,None,:]
    
    temp = Alpha_k*Alpha_l\
    *np.exp((np.pi*N)**2/(4.*(1j*omega+Gamma_k)))\
           *(-np.pi/(1j*omega+Gamma_k))**.5 \
            *(-np.pi/(1j*omega+Gamma_l+omega**2./(1j*omega+Gamma_k)))**.5 \
             *np.exp(-(1j*np.pi*M+np.pi*N*omega/(1j*omega+Gamma_k))**2./ \
                     (4.*(1j*omega+Gamma_l+omega**2./(1j*omega+Gamma_k))))
    
    A_mn_temp = np.sum(temp,(0,1))
    A_mn_temp[np.isnan(A_mn_temp)] = 0
#    plt.imshow(np.abs(A_mn_temp))
##    plt.imshow(np.log10(np.abs(A_mn_temp)))
#    plt.show()
           
    

    return A_mn_temp


#==============================================================================

B = 1 #4.5 #1e-10 #4.5
omega = 200.; print([B,omega])
m_max = 2*omega


#==============================================================================
'#------------------- PLOTTING AND GENERATING k_tilde ------------------- '
#==============================================================================
# xi = np.linspace(0,4*m_max,25*m_max+1)
# #xi = xi[np.arange(100)+700]
# k_tilde_temp = k_tilde(xi,omega,B)
# r_temp = abs(k_tilde_temp)
# phi_temp = np.log(k_tilde_temp).imag
# plt.subplot(131).plot(k_tilde_temp.real,k_tilde_temp.imag,'.')
# plt.gca().set_aspect('equal'); plt.gca().autoscale(tight=True)
# #plt.subplot(211).plot(np.log10(r_temp)*np.cos(phi_temp),np.log10(r_temp)*np.sin(phi_temp),'.-')
# #plt.plot((r_temp)*np.cos(phi_temp),(r_temp)*np.sin(phi_temp),'.-')
# #plt.axis(np.array([-2,2,-2,2])/5.)
# #plt.axis('equal')
# #plt.tight_layout()
# #plt.axis(np.array([-1,1,-1,1])/10.*3.)
# plt.title(r'$\tilde{k}(\xi)$ vs. $\xi$, $\omega$')
# #plt.show()
# 
#==============================================================================

#==============================================================================
'#------------------- PLOTTING AND GENERATING k_hat --------------------------'
#==============================================================================
#xi = xi
#omega = 500
#epsilon = 0.25
#plt.plot(k_hat(xi,omega,epsilon).real,k_hat(xi,omega,epsilon).imag,'.')
#plt.axis([-2,2,-2,2])
#plt.title(r'$\hat{k}(\xi)$ vs. $\xi$')
#plt.show()
#==============================================================================



#==============================================================================
'#------------------- PLOTTING AND GENERATING A_mn_tilde ------------------- '
#==============================================================================
#m = np.arange(0,np.floor(m_max)+1)
#xi = np.linspace(0,omega*1.1,np.pi*omega)
xi = np.linspace(0,np.min([omega*1.1,1000]),np.min([np.pi*omega*2.2,1000*2.]))
#xi = np.linspace(0,250,500)
m = xi
n = m
A_mn_temp = A_mn_tilde(omega = omega, m = m, n = n, B=B)
lambda_tilde = np.linalg.eigvals(A_mn_temp)
lambda_ind = np.argsort(np.abs(lambda_tilde))[::-1] # Reverse sorting is achieved by [::-1]
lambda_tilde = lambda_tilde[lambda_ind]                          
#plt.plot(lambda_tilde.real,lambda_tilde.imag,'.-')
#temp_axis = plt.axis('equal')
#plt.show()
plt.subplot(121).plot(lambda_tilde.real,lambda_tilde.imag,'.')
#lambda_max = np.abs(A_mn_temp[0,0])/np.abs(lambda_tilde).max()
#plt.subplot(121).plot(A_mn_temp.diagonal().real/lambda_max,A_mn_temp.diagonal().imag/lambda_max,'.')
#plt.subplot(121).plot(lambda_tilde.real,lambda_tilde.imag,'.',A_mn_temp.diagonal().real,A_mn_temp.diagonal().imag,'.')
plt.gca().set_aspect('equal'); plt.gca().autoscale(tight=True)
#plt.axis('equal')
#plt.subplot(132).plot((np.abs(lambda_tilde[lambda_ind])),'.')
##plt.show()
#plt.axis('equal')
levels = [1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1]
plt.subplot(122).contour(np.meshgrid(m,m)[0],np.meshgrid(m,m)[0].T,abs(A_mn_temp), levels)
plt.gca().set_aspect('equal'); plt.gca().autoscale(tight=True)
#plt.axis('equal')
#plt.tight_layout()
#plt.show()

plt.show()
#%%
#k_test = lambda a,c,m,omega: a/((-1j*omega)**.5)*np.exp(-(np.pi*m/(2*omega))**2*(c+1j*omega))
k_test = lambda a,c,m,omega: a/((-1j*omega)**.5)*np.exp(-(np.pi*m/(2*omega))**2*(c+1j*omega))
plt.plot(lambda_tilde.real,lambda_tilde.imag,'.')
#c = 25*1.7**(omega/100-1)
c = omega/4
k_0 = k_test(1,c,0,omega)
a = lambda_tilde[0]/k_0
temp = k_test(a,c,xi,omega)
temp2 = k_hat(xi/np.pi,omega,.25)*lambda_tilde[0]/k_hat(0,omega,.25)
k_hat_temp = k_hat(xi/np.pi,omega,.25)
plt.plot(temp.real,temp.imag,'-',temp2.real,temp2.imag,'-')#,k_hat_temp.real,k_hat_temp.imag,'-')
plt.gca().set_aspect('equal'); plt.gca().autoscale(tight=True)
plt.show()


##%%
##omega = 100
#def A_mm_tilde_asymp(omega = 5,m = np.arange(0,101), B = 4.5):
#    alpha_m, gamma_m = YF_quadratures(h_n_sinc_in_Gaussian,101,B)
##    g = lambda x: np.exp(-x**2) + 1j*2./np.sqrt(np.pi)*sp.special.dawsn(x)
#    alpha_m = alpha_m/gamma_m/np.sqrt(np.pi)
#    gamma_m = 1./(2.*gamma_m)
#    gamma_m = -gamma_m**2
#
#    Alpha_k = alpha_m[:,None,None]
#    Alpha_l = alpha_m[None,:,None]
#    Gamma_k = gamma_m[:,None,None]
#    Gamma_l = gamma_m[None,:,None]
#    M = m[None,None,:]
#    
#    temp = np.pi/np.sqrt(-1j*omega)*np.exp(-1j*np.pi**2*M**2/(4*omega)) \
#        *Alpha_k*Alpha_l/(Gamma_k+Gamma_l)**.5 \
#        *np.exp(np.pi**2*m**2*(np.abs(Gamma_k)**2*Gamma_l+np.abs(Gamma_l)**2*Gamma_k) / \
#                (4*omega**2*np.abs(Gamma_k+Gamma_l)**2))
#
##    temp = np.pi/np.sqrt(-1j*omega)*np.exp(-1j*np.pi**2*M**2/(4*omega)) \
##        *Alpha_k*Alpha_l/(Gamma_k+Gamma_l)**.5 \
##        *np.exp(np.pi**2*m**2/(4*(Gamma_k+Gamma_l)**2))
#
#    
#    A_mn_temp = np.sum(temp,(0,1))
#    A_mn_temp[np.isnan(A_mn_temp)] = 0
##    plt.imshow(np.abs(A_mn_temp))
###    plt.imshow(np.log10(np.abs(A_mn_temp)))
##    plt.show()
#
#    return A_mn_temp
#
#temp = A_mm_tilde_asymp(omega,xi/5)
#plt.plot(temp.real,temp.imag,'.')