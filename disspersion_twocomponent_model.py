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
def second_deriv_diff(fun):                                              # 1D second derivative (laplace) operator implementation in the finite difference method 
  return (np.roll(fun,1)+np.roll(fun,-1)-2*fun)/(dx**2)   


def second_deriv_k(fun):                                                  # 1D second derivative (laplace) operator implementation in the pseudospectral method
  k = np.concatenate((np.array(np.arange(0,Nx/2,1)*2*np.pi/Lx),np.array(np.arange(-Nx/2,0,1)*2*np.pi/Lx)))
  k2 = k*k
  return np.fft.ifft(-k2*np.fft.fft(fun))


def pump(x,t):                                                             # Gaussian pump in space and time representing the laser pulse that excites the system
  return F0*np.exp(-(x**2)/(W**2) - (t/Tp)**2 - (1j/hbar)*0*(t-t0))


x = np.arange(-Lx/2,Lx/2,dx)        
psi_x = 0.0025*np.exp(-x**2)      #initial state of the excitonic component of exciton-polariotons
psi_c = x*0                       #initial state of the photonic component of exciton-polariotns
time = np.arange(t0,tmax,dt)
psi_x_data = np.zeros([len(time),Nx],dtype = 'complex')  # array that will be storing the excitonic component in each time step
psi_c_data = np.zeros([len(time),Nx],dtype = 'complex')  # array that will be storing the photonic component in each time step
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


psi_x_data_k = abs(np.fft.fftshift(np.fft.fft2(psi_x_data)))      # Fourier transform from space-time to momentum-energy for the excitonic component
psi_c_data_k = abs(np.fft.fftshift(np.fft.fft2(psi_c_data)))      # Fourier transform from space-time to momentum-energy for the photonic component

k = np.arange(-Nx/2,Nx/2,1)*(2*np.pi)/Lx                          # The wavevector in reciprocal space

def Ex(k):                                          # The analytical excitonic dispersion relation
  Ex = E_x0 + ((hbar**2)*(k**2)/(2*m_exc))
  return Ex
def Ec(k):                                          # The analytical photonic dispersion relation
  Ec = E_c0 + ((hbar**2)*(k**2)/(2*m_cav))
  return Ec

upper = []
lower = []
for i in range(len(k)):                             # Calculating the analytical dispersion relations of the lower (LP) and upper (UP) polariton branch
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
E_UP = 1/2*(Eexc+Ecav+np.sqrt((hbar*OmegaR)**2+(Eexc-Ecav)**2));  # Analytical dispersion of the upper polariton
E_LP = 1/2*(Eexc+Ecav-np.sqrt((hbar*OmegaR)**2+(Eexc-Ecav)**2));  # Analytical dispersion of the lower polariton




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

#######################   plotting #################

class OOMFormatter(ticker.ScalarFormatter):                                      #usefull class to make more aesthetic ticks on plot
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
             self.format = r'$\mathdefault{%s}$' % self.format

fig3 = plt.figure(figsize=(9,7.5),dpi=400)
gs = fig3.add_gridspec(3,3)
ax1 = fig3.add_subplot(gs[0, 0])
ax2 = fig3.add_subplot(gs[1, 0])
ax3 = fig3.add_subplot(gs[2,0])
ax4 = fig3.add_subplot(gs[:, 1])

im4 = ax4.pcolor(dispersion_km,dispersion_em,np.flipud(abs(dispersion[10000:-10000,:])**2),cmap ='YlGn',shading='auto')


cb4 = fig.colorbar(im4, orientation='horizontal',pad=0.1,format=OOMFormatter(5, mathText=False))
tick_locator = ticker.MaxNLocator(nbins=4)

cb4.locator = tick_locator
cb4.update_ticks()
ax4.set_xlabel("k ($\mathdefault{\mathrm{\mu m^{-1}}}$)")
ax4.set_ylabel("E (meV)")
m = dispersion_k

upper = []
lower = []
for i in range(len(m)):
  matrix = np.zeros([2,2])
  matrix[0][0] = Ex(m[i])
  matrix[1][1] = Ec(m[i])
  matrix[0][1]=matrix[1][0] = hbar*OmegaR/2
  eigs = np.real(np.linalg.eigvals(matrix))
  upper.append(max(eigs))
  lower.append(min(eigs))

ax4.plot(m,Ec(m),color = 'DarkOrange',linestyle='--',label = 'foton',linewidth=2,alpha=0.6)
ax4.plot(m,Ex(m),color = 'DarkGreen',linestyle='--',label = 'ekscyton',linewidth=2,alpha=0.6)
ax4.plot(m,upper,color = 'DarkOrange',linewidth=2,label='UP',alpha=0.6)
ax4.plot(m,lower,color = 'DarkGreen',linewidth=2,label='LP',alpha=0.6)
ax4.text(-1,5,'$\mathdefault{E_{\mathrm{UP}}(k)}$')
ax4.text(3,2.5,'$\mathdefault{E_{\mathrm{C}}(k)}$')
ax4.text(-4,0.5,'$\mathdefault{E_{\mathrm{X}}(k)}$')
ax4.text(-1,-5,'$\mathdefault{E_{\mathrm{LP}}(k)}$')
ax4.set_xlim(-5,5)
ax4.set_ylim(-10,10)
ax4.yaxis.tick_right()
ax4.yaxis.set_label_position("right")
ax4.text(-6, 9.5, 'd', fontweight='bold',fontsize=12)

im1 = ax1.pcolor(xm,tm,(abs(psi_c_data))**2,cmap='Oranges')
cb1 = fig.colorbar(im1, orientation='horizontal',pad=0.2,format=OOMFormatter(-2, mathText=False))
tick_locator = ticker.MaxNLocator(nbins=4)
cb1.locator = tick_locator
cb1.update_ticks()

ax1.set_ylabel('t (ps)')
ax1.text(-12, 1.8, 'a', fontweight='bold',fontsize=12)
ax1.text(-8, 1.5, '$\mathdefault{|\psi_{\mathrm{C}}|^2}$')
ax1.set_xlim(-10,10)
im2 = ax2.pcolor(xm,tm,(abs(psi_x_data))**2,cmap='Greens')
ax2.set_ylabel('t (ps)')
ax2.set_xlabel('x ($\mathdefault{\mathrm{\mu m}}$)')
ax2.text(-12, 1.8, 'b', fontweight='bold',fontsize=12)
cb2 = fig.colorbar(im2, orientation='horizontal',pad=0.3,format=OOMFormatter(-2, mathText=False))
tick_locator = ticker.MaxNLocator(nbins=4)
cb2.locator = tick_locator
cb2.update_ticks()
ax2.set_xlim(-10,10)
ax2.text(-8, 1.5, '$\mathdefault{|\psi_{\mathrm{X}}|^2}$')
ax3.plot(np.append(np.array([0]),time),np.append(np.array([0]),(abs(psi_x_data[:,int(Nx/2)]))**2*100),color='DarkGreen',label='excitonic component')
ax3.plot(np.append(np.array([0]),time),np.append(np.array([0]),(abs(psi_c_data[:,int(Nx/2)]))**2*100),color='DarkOrange',label = 'photonic component')
ax3.set_xlabel('t (ps)')
ax3.set_ylabel('$\mathdefault{|\psi_{\mathrm{C}}|^2}$,$\mathdefault{|\psi_{\mathrm{X}}|^2}$')
ax3.text(-0.2, 1.7, 'c', fontweight='bold',fontsize=12)
ax3.set_xlim(0,2)

fig.tight_layout()
