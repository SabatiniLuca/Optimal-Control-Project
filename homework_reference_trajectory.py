#
#Gradient Method for Optimal Control
#Reference Trajectory Generation
#Marco Falotico
#Bologna, 8/11/2024
#


import numpy as np
# import dynamics_msd as dyn
import dynamics_pend as dyn

from scipy.linalg import solve_discrete_are
from scipy.integrate import solve_ivp



def gen(step_reference, tf, dt, ns, ni):

      TT = int(tf/dt)
      
      ref_deg_0 = 0
      ref_deg_T = 30

      
      xx_ref = np.zeros((ns, TT))
      uu_ref = np.zeros((ni, TT))

      
      if not step_reference:
            ...
            #TODO, try generating a smooth trajectory

            t = np.linspace(0, tf, TT)
            
            # Polynomial trajectory
            # xx_ref[0, :] = ref_deg_0 + (3 * (ref_deg_T - ref_deg_0) / tf**2) * t**2 - (2 * (ref_deg_T - ref_deg_0) / tf**3) * t**3

            # Trigonometric trajectory
            # xx_ref[0, :] = ref_deg_0 + (ref_deg_T - ref_deg_0) * (1 - np.cos(np.pi * t / tf)) / 2

            # Sigmoid trajectory
            # xx_ref[0, :] = ref_deg_0 + (ref_deg_T - ref_deg_0) / (1 + np.exp(-10 * (t / tf - 0.5)))
            
            # Bell-shaped trajectory
            xx_ref[0, :] = ref_deg_0 + (ref_deg_T - ref_deg_0) * np.exp(-((t - tf / 2) ** 2) / (2 * (tf / 4) ** 2))

            '''
            Loock for a smooth function thatbring then system between two equilibria
            a possibility is a bell-shaped function but look for something else also\ 
            '''

      else:

            KKeq = dyn.KKeq

            xx_ref[0,int(TT/2):] = np.ones((1,int(TT/2)))*np.ones((1,int(TT/2)))*np.deg2rad(ref_deg_T)
            uu_ref[0,:] = KKeq*np.sin(xx_ref[0,:])
   
      return xx_ref, uu_ref