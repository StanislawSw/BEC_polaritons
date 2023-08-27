import torch
import pandas as pd
import time
import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from tqdm import tqdm
cuda0 = torch.device('cuda:0')
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm
from matplotlib import ticker
from matplotlib import colors
from tqdm import tqdm
import csv

font = {'family' : 'serfi','weight' : 'normal','size': 10}
plt.rc('font', **font)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# SIMULATION PARAMETERS
# -----------------------------------------------------------------------------#
Lx=20; Ly=20; n=8; Nx=2**n; Ny=2**n; dt=0.0001; tmax=2; t0 = 0.1; 
Lx=80; Ly=80; n=8; Nx=2**n; Ny=2**n; dt=0.001; tmax=40; t0 = 0.1;    
# -----------------------------------------------------------------------------#
# PHYSICAL PARAMETERS
# -----------------------------------------------------------------------------#
hbar    =  0.65821195;              # Dirac constant                  [meV ps]
E_0     =  510998946.1;             # Free electron energy               [meV]
c       =  299.792458;              # Light speed                     [mum/ps]
m_e0    =  E_0/(c**2);              # Free electron mass    [(meV ps^2)/mum^2]
n_cav   =  0.65#0.65;               # Refraction index of microcavites     [-]
# -----------------------------------------------------------------------------#
Delta   = 0                         # Detuning                           [meV]
E_x0  = 0;                          # Exciton resonance enrgy            [meV]
E_c0  = -2;                         # Photon resonance energy            [meV]
OmegaR  = 9.8/hbar                  # Rabi splitting
# -----------------------------------------------------------------------------#
m_cav   = 0.5*(10**(-4))*m_e0;      # Photon effective mass  [(meV ps^2)/mum^2]
m_exc   = 0.1*m_e0;                 # Exciton effective mass [(meV ps^2)/mum^2]
tauC    = 1;                        # Photon lifetime                      [ps]
tauX    = 400;                      # Exciton lifetime                     [ps]
g       = 1*(0.2/hbar)*10**(-3);    # Exciton-exciton interaction         [meV]
t0      = 0.1;                      # initial impulse time                [ps]
W       = np.sqrt(0.4);             # gaussian variance of signal         [um]
Tp      = np.sqrt(0.003);           # impulse duration                    [ps]
F0      = 200;                      # pump amplitude               

-----------------------------------------------------------------------------#
def second_deriv_diff(fun):                                              # 2D second derivative (laplace) operator implementation in the finite difference method 
  return (np.roll(fun,1)+np.roll(fun,-1)-2*fun)/(dx**2)   


def second_deriv_k(fun):                                                  # 2D second derivative (laplace) operator implementation in the pseudospectral method
  k = np.concatenate((np.array(np.arange(0,Nx/2,1)*2*np.pi/Lx),np.array(np.arange(-Nx/2,0,1)*2*np.pi/Lx)))
  k2 = k*k
  return np.fft.ifft(-k2*np.fft.fft(fun))


def pump(x,t):                                                             # Gaussian pump in space and time representing the laser pulse that excites the system
  return F0*np.exp(-(x**2)/(W**2) - (t/Tp)**2 - (1j/hbar)*0*(t-t0))


x = np.arange(-Lx/2,Lx/2,dx)        
psi_x = 0.0025*np.exp(-x**2)      #initial state of the excitonic component of exciton-polariotons
psi_c = x*0                       #initial state of the photonic component of exciton-polariotns
time = np.arange(t0,tmax,dt)
psi_x_data = np.zeros([len(time),Nx],dtype = 'complex')
psi_c_data = np.zeros([len(time),Nx],dtype = 'complex')
F_data = np.zeros([len(time),Nx],dtype = 'complex')
psiC_data_ks = np.zeros([len(time),Nx],dtype = 'complex')
xm,tm = np.meshgrid(x,time)
t = t0
#---------------------------------- EVOLUTION OF THE SYSTEM CALCULATED USING THE 4TH ORDER RUNGE-KUTTA ALGORITHM------------------------#
for i in tqdm(range(len(time))):
  psi_x_data[i,:] = psi_x
  psi_c_data[i,:] = psi_c

  psifft=abs(np.fft.fftshift(np.transpose(np.fft.fft((psi_c[:])))))
  F_data[i,:]=abs(pump(x,t))
  psiC_data_ks[i,:]=psifft
  du1_psi_x = (-1j/hbar)*((E_x0*psi_x-((hbar**2)/(2*m_exc))*second_deriv_diff(psi_x))+0.5*hbar*OmegaR*psi_c-((1j*hbar)/(2*tauX))*psi_x +g*(abs(psi_x)**2)*psi_x)
  du1_psi_c = (-1j/hbar)*((E_c0*psi_c-((hbar**2)/(2*m_cav))*second_deriv_diff(psi_c))+0.5*hbar*OmegaR*psi_x-((1j*hbar)/(2*tauC))*psi_c +pump(x,t))
  psi_x_sim = psi_x + 0.5*du1_psi_x*dt
  psi_c_sim = psi_c + 0.5*du1_psi_c*dt


  du2_psi_x = (-1j/hbar)*((E_x0*psi_x_sim-((hbar**2)/(2*m_exc))*second_deriv_diff(psi_x_sim))+0.5*hbar*OmegaR*psi_c_sim-((1j*hbar)/(2*tauX))*psi_x_sim +g*(abs(psi_x_sim)**2)*psi_x_sim)
  du2_psi_c = (-1j/hbar)*((E_c0*psi_c_sim-((hbar**2)/(2*m_cav))*second_deriv_diff(psi_c_sim))+0.5*hbar*OmegaR*psi_x_sim-((1j*hbar)/(2*tauC))*psi_c_sim +pump(x,t+0.5*dt))
  psi_x_sim = psi_x + 0.5*du2_psi_x*dt
  psi_c_sim = psi_c + 0.5*du2_psi_c*dt


  du3_psi_x = (-1j/hbar)*((E_x0*psi_x_sim-((hbar**2)/(2*m_exc))*second_deriv_diff(psi_x_sim))+0.5*hbar*OmegaR*psi_c_sim-((1j*hbar)/(2*tauX))*psi_x_sim +g*(abs(psi_x_sim)**2)*psi_x_sim)
  du3_psi_c = (-1j/hbar)*((E_c0*psi_c_sim-((hbar**2)/(2*m_cav))*second_deriv_diff(psi_c_sim))+0.5*hbar*OmegaR*psi_x_sim-((1j*hbar)/(2*tauC))*psi_c_sim +pump(x,t+0.5*dt))
  psi_x_sim = psi_x + du3_psi_x*dt
  psi_c_sim = psi_c + du3_psi_c*dt


  du4_psi_x = (-1j/hbar)*((E_x0*psi_x_sim-((hbar**2)/(2*m_exc))*second_deriv_diff(psi_x_sim))+0.5*hbar*OmegaR*psi_c_sim-((1j*hbar)/(2*tauX))*psi_x_sim +g*(abs(psi_x_sim)**2)*psi_x_sim)
  du4_psi_c = (-1j/hbar)*((E_c0*psi_c_sim-((hbar**2)/(2*m_cav))*second_deriv_diff(psi_c_sim))+0.5*hbar*OmegaR*psi_x_sim-((1j*hbar)/(2*tauC))*psi_c_sim +pump(x,t+dt))

  psi_x = psi_x + (du1_psi_x +2*du2_psi_x+2*du3_psi_x+du4_psi_x)*dt/6
  psi_c = psi_c + (du1_psi_c +2*du2_psi_c+2*du3_psi_c+du4_psi_c)*dt/6
  t = t + dt
fft_energy=abs(np.fft.fftshift(np.fft.fft2(psi_c_data)))
F_fft_energy=abs(np.fft.fftshift(np.fft.fft2(F_data)))
psi_x_data_k = abs(np.fft.fftshift(np.fft.fft2(psi_x_data)))
psi_c_data_k = abs(np.fft.fftshift(np.fft.fft2(psi_c_data)))
k = np.arange(-Nx/2,Nx/2,1)*(2*np.pi)/Lx
def Ex(k):
  Ex = E_x0 + ((hbar**2)*(k**2)/(2*m_exc))
  return Ex
def Ec(k):
  Ec = E_c0 + ((hbar**2)*(k**2)/(2*m_cav))
  return Ec

upper = []
lower = []
for i in range(len(k)):
  matrix = np.zeros([2,2])
  matrix[0][0] = Ex(k[i])
  matrix[1][1] = Ec(k[i])
  matrix[0][1]=matrix[1][0] = hbar*OmegaR/2
  eigs = np.real(np.linalg.eigvals(matrix))
  upper.append(max(eigs))
  lower.append(min(eigs))
en = np.linspace(-1,1,nmax)*(np.pi*hbar)/dt

Ecav=(hbar**2)*(k**2)/(2*m_cav)
Eexc=(hbar**2)*(k**2)/(2*m_exc)
E_UP = 1/2*(Eexc+Ecav+np.sqrt((hbar*OmegaR)**2+(Eexc-Ecav)**2));
E_LP = 1/2*(Eexc+Ecav-np.sqrt((hbar*OmegaR)**2+(Eexc-Ecav)**2));




km,em = np.meshgrid(k,en[10000:-10000])
changey = 0
changeyd = 0
changex = 0
km = np.real(km)
em = np.real(em)
#dispersion_km = km
#dispersion_em = em
#dispersion_k = k
#dispersion = fft_energy
print(np.shape(km),np.shape(em))
fig,ax = plt.subplots(figsize=(5,10))

#ax.pcolor(dispersion_km,dispersion_em,np.flipud(abs(dispersion[10000:-10000,:])**2),cmap ='YlGn',shading='auto')

#dispersion_km = km
#dispersion_em = em
#dispersion_k = k
#dispersion = fft_energy
