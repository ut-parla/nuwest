from typing import Callable

import cupy as cp
import numba
from numba import cuda
import numpy as np
import math
import sys
from scipy.linalg import cholesky_banded
import pykokkos as pk
import time

from parla import Parla, TaskSpace, spawn
from parla.cython.device_manager import cpu, gpu
from parla.cython.tasks import AtomicTaskSpace as TaskSpace
from parla.tasks import get_current_context


#NUM_GPUS = 3

class BoltzmannDSMC:
    def __init__(self, zero_dim, zerod_ef, null_coll, num_steps, opc, name, restart_cyclenum, Nc, gridpts, bnf, num_nodes, pyk, recomb, ecov, num_gpus):
        self.NUM_GPUS = num_gpus    

        self.EF_kernel: Callable
        self.electron_kernel: Callable
        self.heavies_kernel_fluid: Callable
        self.data_write: Callable
        self._init_kernels(zero_dim, pyk, recomb, null_coll)

        self.pyk: bool = pyk
        self.is_zero_dim = zero_dim

        # used for indexing in some places for clarity
        self.xx: int = 0
        self.xy: int = 1
        self.xz: int = 2
        self.vx: int = 3
        self.vy: int = 4
        self.vz: int = 5
        self.wt: int = 0
        self.ai: int = 1
        self.ae: int = 2
        self.en: int = 3
        self.ec: int = 4

        ## Heavy parameters
        P: int = 1
        self.nn = 3.22e16 * P  # only one in normal units
        self.nn_Di = 2.07e18
        self.nn_mui = 4.65e19
        self.nn_Ds = 2.42e18
        self.D_i = self.nn_Di / self.nn
        self.D_s = self.nn_Ds / self.nn
        self.mu_i = self.nn_mui / self.nn
        self.epsilon = 5.5263e5  # permitivity
        self.m_e = 9.10938*10**(-31)
        self.j_ev = 6.241509*(10**18)

        ## 1-D electric field
        self.freq = 13.56e6  
        self.tau = 1.0 / self.freq
        self.V0 = 100

        ## 0-D electric field
        self.zd_ef = zerod_ef

        ## Base numerical stuff findme
        self.N: int = gridpts - 1
        self.big_N: int = bnf*(gridpts)
        self.L: float = 2.00
        self.tau: float = 1.0 / self.freq  # period
        self.dx: float = self.L / self.N
        
        ## Time
        self.steps_per_cycle: int = 50000
        self.dt_big: float = (self.tau / 50000)
        self.dt_ratio: int = 10
        self.dt_el: float = self.dt_big / self.dt_ratio
        self.dt_col = self.dt_el
        self.curr_t: float = 0.0

        ## Particles
        self.Nc_list = []
        self.Na_list = []
        self.Nmax_list = []
        self.Nnew_list = []
        self.Nmax_fac = 3.0
        Nsplit = int(Nc/self.NUM_GPUS) # taking input
        for ng in range(self.NUM_GPUS):
            self.Nc_list.append(Nsplit)
            self.Na_list.append(Nsplit)
            self.Nmax_list.append(int(self.Nmax_fac*Nsplit))
            self.Nnew_list.append(0)
            
        ## Initial conditions
        self.E_cov: float = ecov
        ic_dens: int = 1e9

        ## Setup work arrays
        self.E_ar = np.zeros(self.N)
        self.Ji_ar = np.zeros(self.N)
        self.Js_ar = np.zeros(self.N)
        self.ni_rhs = np.zeros(self.N + 1)
        self.ns_rhs = np.zeros(self.N + 1)
        self.V_ar = np.zeros(self.N - 1)
        self.V_rhs = np.zeros(self.N - 1)
        self.ne_ar = np.zeros(self.N + 1)
        self.ni_ar = np.ones(self.N + 1) * ic_dens
        self.ns_ar = np.zeros(self.N + 1)
        self.nrg_ar = np.zeros(self.N + 1) 
        self.counter_g0_ar = np.zeros(self.N + 1) 
        self.counter_g1_ar = np.zeros(self.N + 1) 
        self.counter_g2_ar = np.zeros(self.N + 1) 
        self.counter_g3_ar = np.zeros(self.N + 1) 
        self.Te_ar = np.zeros(self.N + 1) 
        
        # Defining uniform particle wt with base IC
        self.ww: float = ic_dens * (self.N / (Nc * num_nodes))

        # For Dilip's sum algorithm
        self.data_out_list = []
        self.data_out_np_list = []
        self.data_out_list2 = []
        self.data_out_np_list2 = []
        self.temp_x_list = []
        for ng in range(self.NUM_GPUS):
            cp.cuda.Device(ng).use()
            self.data_out_list.append(cp.zeros((self.big_N, 4), dtype=np.float64))
            self.data_out_np_list.append(np.zeros((self.big_N, 4), dtype=np.float64))
            self.data_out_list2.append(cp.zeros((self.big_N, 4), dtype=np.float64))
            self.data_out_np_list2.append(np.zeros((self.big_N, 4), dtype=np.float64))

            if (zero_dim):
                self.temp_x_list.append(cp.random.rand(self.Nmax_list[ng]))
            else:
                self.temp_x_list.append(cp.zeros(self.Nmax_list[ng]))

        # For MPI
        self.ni_src = np.zeros(self.N + 1)
        self.ns_src = np.zeros(self.N + 1)
        self.ne_ar_mpi = np.zeros(self.N + 1)
        self.ni_src_mpi = np.zeros(self.N + 1)
        self.ns_src_mpi = np.zeros(self.N + 1)
        self.nrg_ar_mpi = np.zeros(self.N + 1)

        # GPU Stuff
        self.threads_per_block: int = 64
        self.num_blocks: int = 1024

        # EF stuff
        self.Vc_diag, self.Vc_lower_diag, self.V_tempy = self._chol_EF(self.N, self.dx)

        # INITIALIZE

        todd_flag = 0 # =0 is not used, =1 is todd soln, =3 is milinda soln

        # Initializing to Todd's data 
        if (todd_flag == 3):
            print("USING LIU INITIAL CONDITIONS")
            temp_x_ar = np.linspace(0, self.L, self.N + 1)
            temp_grid = np.copy(temp_x_ar)
            jab_ne = np.zeros(self.N+1)
            jab_Te = 0.5*np.ones(self.N+1)

            for tx in range(self.N+1):
                self.ni_ar[tx] = 1e7 + 1e9 * (1-temp_x_ar[tx]/self.L)**2 * (temp_x_ar[tx]/self.L)**2 
                jab_ne[tx] = 1e7 + 1e9 * (1-temp_x_ar[tx]/self.L)**2 * (temp_x_ar[tx]/self.L)**2 

            print("init ne = ", jab_ne)
            print("init ni = ", self.ni_ar)

            ng = 0
            cp.cuda.Device(0).use()
            
            Nc_temp = self.Nc_list[ng]
            temp_local_Nc = (Nc_temp * jab_ne / np.sum(jab_ne)).astype(int) 
            self.Nc_list[ng] = int(np.sum(temp_local_Nc))
            self.Nmax_list[ng] = int(self.Nmax_fac * self.Nc_list[ng])
            Nmax_temp = self.Nmax_list[ng]
            print("total num particles = ", np.sum(temp_local_Nc))

            ## Setting up particles and particle bins
            kTm = 1.5 * 1.1697 * 10**11 # 1 eV
         
            self.big_data_ar_list = []
            self.big_tosum_ar_list = []
            self.big_curr_xbins_list = []
            self.big_forE_xbins_list = []
            self.nni_list = []
            
            tot_data_ar = cp.zeros((Nmax_temp, 6))
            tot_tosum_ar = cp.zeros((Nmax_temp, 4))
            
            curr_Nc = 0
            for ii in range(self.N+1):
                num_cell = int(temp_local_Nc[ii])
                if (num_cell > 0):            
                    # put in the x's
                    tot_data_ar[curr_Nc : curr_Nc+num_cell , self.xx] = cp.ones(num_cell)*temp_grid[ii]
                    if (ii == 0):
                        tot_data_ar[curr_Nc : curr_Nc+num_cell , self.xx] = cp.ones(num_cell)*temp_grid[ii] + 1e-6
                    elif (ii == self.N):
                        tot_data_ar[curr_Nc : curr_Nc+num_cell , self.xx] = cp.ones(num_cell)*temp_grid[ii]
                    else:
                        tot_data_ar[curr_Nc : curr_Nc+num_cell , self.xx] = cp.ones(num_cell)*temp_grid[ii] - 1e-6

                    # put in the v's
                    E_cell = jab_Te[ii] * kTm
                    cov = [[E_cell, 0, 0], [0, E_cell, 0], [0, 0, E_cell]]
                    tot_data_ar[curr_Nc : curr_Nc+num_cell, self.vx:self.vz+1] = cp.random.multivariate_normal([0,0,0], cov, num_cell)
       
                    m_e = 9.10938*10**(-31)
                    j_ev = 6.241509*(10**18)

                    curr_Nc = curr_Nc + num_cell

            self.ww = np.sum(jab_ne)/np.sum(temp_local_Nc) / num_nodes
            tot_data_ar[curr_Nc:, self.xx] = self.L * cp.random.rand(Nmax_temp-curr_Nc)

            # Weights
            tot_tosum_ar[0:curr_Nc, self.wt] = self.ww * cp.ones(curr_Nc)
            # Energy
            en_xx = cp.multiply(tot_data_ar[0:curr_Nc,self.vx], tot_data_ar[0:curr_Nc,self.vx])
            en_yy = cp.multiply(tot_data_ar[0:curr_Nc,self.vy], tot_data_ar[0:curr_Nc,self.vy])
            en_zz = cp.multiply(tot_data_ar[0:curr_Nc,self.vz], tot_data_ar[0:curr_Nc,self.vz])
            tot_tosum_ar[0:curr_Nc, self.en] = 0.5*self.m_e*self.j_ev*cp.add(en_xx, cp.add(en_yy, en_zz))

            self.big_collct_ar_list = []
            temp_big_collct_ar = cp.zeros((Nmax_temp, 4))
            self.big_collct_ar_list.append(temp_big_collct_ar)

            # Shuffling
            cp.random.seed(1)
            cp.random.shuffle(tot_data_ar[0:curr_Nc,:])
            cp.random.seed(1)
            cp.random.shuffle(tot_tosum_ar[0:curr_Nc,:])
            
            # Putting back into list    
            self.big_data_ar_list.append(tot_data_ar)
            self.big_tosum_ar_list.append(tot_tosum_ar)
                
            # Bins            
            curr_xbins_temp = cp.zeros(Nmax_temp).astype(int)
            forE_xbins_temp = cp.zeros(Nmax_temp).astype(int)
            curr_xbins_temp[0:Nc_temp] = ((self.big_data_ar_list[ng][0:Nc_temp, self.xx] + 0.5*self.dx)/self.dx).astype(int)
            forE_xbins_temp[0:Nc_temp] = (self.big_data_ar_list[ng][0:Nc_temp, self.xx]/self.dx).astype(int)
            self.big_curr_xbins_list.append(curr_xbins_temp)
            self.big_forE_xbins_list.append(forE_xbins_temp) 

            # Updating particle count
            self.nni_list.append(0)
            print("FINISHED INIT 3 SPECIES SETUP")
            
        # Uniform IC
        else:
            #print("UNIFORM IC")
            ## Setting up particles and particle bins
            self.m_e = 9.10938*10**(-31)
            self.j_ev = 6.241509*(10**18)
            kTm = 1.5 * 1.1697 * 10**11 # 1 eV
            cov = [[self.E_cov * kTm, 0, 0], [0, self.E_cov * kTm, 0], [0, 0, self.E_cov * kTm]]
         
            self.big_data_ar_list = []
            self.big_tosum_ar_list = []
            self.big_curr_xbins_list = []
            self.big_forE_xbins_list = []
            self.big_collct_ar_list = []
            self.nni_list = []

            for ng in range(self.NUM_GPUS):
                cp.cuda.Device(ng).use()
    
                Nc_temp = self.Nc_list[ng]
                Nmax_temp = self.Nmax_list[ng]

                #print("Nc temp = ", Nc_temp)
                #print("Nmax temp = ", Nmax_temp)
                tot_data_ar = cp.zeros((Nmax_temp, 6))
                tot_tosum_ar = cp.zeros((Nmax_temp, 4))

                # Particle Position + Velocity
                tot_data_ar[0:Nc_temp, self.vx:self.vz+1] = cp.random.multivariate_normal([0,0,0], cov, Nc_temp)
                tot_data_ar[0:Nc_temp, self.xx] = cp.arange(Nc_temp) * (self.L / (Nc_temp-1))
                tot_data_ar[0,self.xx] += 0.000001*self.dx
                tot_data_ar[Nc_temp-1,self.xx] -= 0.000001*self.dx
                
                # Dummy entries for reduction nonsense
                tot_data_ar[Nc_temp:Nmax_temp,self.xx] = self.L*cp.random.rand(Nmax_temp-Nc_temp)

                # Weights
                tot_tosum_ar[0:Nc_temp, self.wt] = self.ww * cp.ones(Nc_temp)
                # Energy
                en_xx = cp.multiply(tot_data_ar[0:Nc_temp,self.vx], tot_data_ar[0:Nc_temp,self.vx])
                en_yy = cp.multiply(tot_data_ar[0:Nc_temp,self.vy], tot_data_ar[0:Nc_temp,self.vy])
                en_zz = cp.multiply(tot_data_ar[0:Nc_temp,self.vz], tot_data_ar[0:Nc_temp,self.vz])
                tot_tosum_ar[0:Nc_temp, self.en] = 0.5*self.m_e*self.j_ev*cp.add(en_xx, cp.add(en_yy, en_zz))
            
                temp_big_collct_ar = cp.zeros((Nmax_temp, 4))
                self.big_collct_ar_list.append(temp_big_collct_ar)

                # Shuffling
                cp.random.seed(1)
                cp.random.shuffle(tot_data_ar[0:Nc_temp,:])
                cp.random.seed(1)
                cp.random.shuffle(tot_tosum_ar[0:Nc_temp,:])

                self.big_data_ar_list.append(tot_data_ar)
                self.big_tosum_ar_list.append(tot_tosum_ar)

                # Bins            
                curr_xbins_temp = cp.zeros(Nmax_temp).astype(int)
                forE_xbins_temp = cp.zeros(Nmax_temp).astype(int)
                curr_xbins_temp[0:Nc_temp] = ((self.big_data_ar_list[ng][0:Nc_temp, self.xx] + 0.5*self.dx)/self.dx).astype(int)
                forE_xbins_temp[0:Nc_temp] = (self.big_data_ar_list[ng][0:Nc_temp, self.xx]/self.dx).astype(int)
                self.big_curr_xbins_list.append(curr_xbins_temp)
                self.big_forE_xbins_list.append(forE_xbins_temp) 

                # Updating particle count
                self.nni_list.append(0)

        # Null collision method setup
        self.P_null = 0.
        self.nu_max = 0. 
        self.Ncoll_offset = 0
        self.Nnull_max = self.Nmax_list[0]
        if (null_coll):
            self.nu_max = (1e6*self.nn) * 2.0e-19 * 3e6 # made up for now
            self.P_null = 1 - math.exp(-1.0*self.nu_max*self.dt_big)

        self.need_to_reshuffle = False

        ## Grid arrays
        self.ne_ar_list = []
        self.ni_src_list = []
        self.ns_src_list = []
        self.nrg_ar_list = []
        self.counter_g0_ar_list = []
        self.counter_g1_ar_list = []
        self.counter_g2_ar_list = []
        self.counter_g3_ar_list = []
        for ng in range(self.NUM_GPUS):
            self.ne_ar_list.append(np.zeros(self.N+1))
            self.ni_src_list.append(np.zeros(self.N+1))
            self.ns_src_list.append(np.zeros(self.N+1))
            self.nrg_ar_list.append(np.zeros(self.N+1))
            self.counter_g0_ar_list.append(np.zeros(self.N+1))
            self.counter_g1_ar_list.append(np.zeros(self.N+1))
            self.counter_g2_ar_list.append(np.zeros(self.N+1))
            self.counter_g3_ar_list.append(np.zeros(self.N+1))

        self.ne_ar_sum = np.zeros((self.N+1,self.NUM_GPUS))
        self.ni_src_sum = np.zeros((self.N+1,self.NUM_GPUS))
        self.ns_src_sum = np.zeros((self.N+1,self.NUM_GPUS))
        self.nrg_ar_sum = np.zeros((self.N+1,self.NUM_GPUS))

        self.ne_ar.fill(0)
        self.ni_src.fill(0)
        self.ns_src.fill(0)
        self.nrg_ar.fill(0)
    
        self.ne_ar[:] = np.sum(self.ne_ar_sum,axis=1)
        self.ni_src[:] = np.sum(self.ni_src_sum,axis=1)
        self.ns_src[:] = np.sum(self.ns_src_sum,axis=1)
        self.nrg_ar[:] = np.sum(self.nrg_ar_sum,axis=1)
        self.Te_ar.fill(0)
        self.Te_ar[:] = np.divide((2./3.)*self.ww * self.nrg_ar, self.ne_ar, where=self.ne_ar>0.9*self.ww)
        self.Te_ar[np.where(self.Te_ar > 20)] = 0.

        # Initializing EF
        self.Vcarry_ar, self.E_ar = self.EF_kernel(
            self.E_ar,
            self.ne_ar,
            self.ni_ar,
            self.curr_t,
            self.V_ar,
            self.V_rhs,
            self.Vc_diag,
            self.Vc_lower_diag,
            self.V_tempy,
            self.N,
            self.dx,
            self.epsilon,
            self.V0,
            self.freq,
        )

        ## Random #s, recombination, atomics, null coll method
        self.forgpu_R_vec_list = []
        self.forrecomb_v_ar_list = []
        self.curr_count_list = []

        self.nc_flag_ar_list = []
        self.nc_inds_ar_list = []

        for ng in range(self.NUM_GPUS):
            cp.cuda.Device(ng).use()
            forgpu_R_vec_temp = cp.zeros((self.Nmax_list[ng], 7)) 
            forgpu_R_vec_temp[:,:] = cp.random.rand(self.Nmax_list[ng],7)
            self.forgpu_R_vec_list.append(forgpu_R_vec_temp)

            forrecomb_v_ar_temp = cp.zeros((self.Nmax_list[ng],3))
            self.forrecomb_v_ar_list.append(forrecomb_v_ar_temp)

            curr_count_temp = cp.zeros(1).astype(int)
            self.curr_count_list.append(curr_count_temp)

            nc_inds_ar_temp = (cp.zeros(self.Nmax_list[ng])).astype(int)
            nc_inds_ar_temp[:] = cp.random.randint(0, self.Nc_list[ng], self.Nnull_max) 
            self.nc_inds_ar_list.append(nc_inds_ar_temp)

            nc_flag_ar_temp = cp.zeros(self.Nmax_list[ng]).astype(int)
            self.nc_flag_ar_list.append(nc_flag_ar_temp)

        ## For copying to GPU and device arrays for electron kernel
        self.gpu_E_ar_list = []
        self.gpu_ne_ar_list = []
        self.gpu_ni_ar_list = []
        self.gpu_ns_ar_list = []
        self.gpu_Te_ar_list = []
        self.np_data_ar = np.zeros(5*self.N + 4)
        self.cp_data_ar_list = []
        
        self.d_curr_count_list = []
        self.d_currxbins_list = []
        self.d_forExbins_list = []
        self.d_bigRvec_list = []
        self.d_nc_flag_ar_list = []
        self.d_nc_inds_ar_list = []
        self.d_forrecomb_v_ar_list = []
        self.d_data_ar_list = []
        self.d_tosum_ar_list = []
        self.d_collct_ar_list = []
        self.d_E_ar_list = []
        self.d_ne_ar_list = []
        self.d_ni_ar_list = []
        self.d_ns_ar_list = []
        self.d_Te_ar_list = []
      
        for ng in range(self.NUM_GPUS):
            cp.cuda.Device(ng).use()
            gpu_E_ar_temp = cp.zeros(self.N)
            gpu_ne_ar_temp = cp.zeros(self.N + 1)
            gpu_ni_ar_temp = cp.zeros(self.N + 1)
            gpu_ns_ar_temp = cp.zeros(self.N + 1)
            gpu_Te_ar_temp = cp.zeros(self.N + 1)
            cp_data_ar_temp = cp.zeros(5*self.N + 4)
            self.gpu_E_ar_list.append(gpu_E_ar_temp)
            self.gpu_ne_ar_list.append(gpu_ne_ar_temp)
            self.gpu_ni_ar_list.append(gpu_ni_ar_temp)
            self.gpu_ns_ar_list.append(gpu_ns_ar_temp)
            self.gpu_Te_ar_list.append(gpu_Te_ar_temp)
            self.cp_data_ar_list.append(cp_data_ar_temp)

            d_curr_count_temp = cuda.to_device(self.curr_count_list[ng])
            d_currxbins_temp = cuda.to_device(self.big_curr_xbins_list[ng])
            d_forExbins_temp = cuda.to_device(self.big_forE_xbins_list[ng])
            d_bigRvec_temp = cuda.to_device(self.forgpu_R_vec_list[ng])
            d_nc_flag_ar_temp = cuda.to_device(self.nc_flag_ar_list[ng])
            d_nc_inds_ar_temp = cuda.to_device(self.nc_inds_ar_list[ng])
            d_forrecomb_v_ar_temp = cuda.to_device(self.forrecomb_v_ar_list[ng])
            d_data_ar_temp = cuda.to_device(self.big_data_ar_list[ng])
            d_tosum_ar_temp = cuda.to_device(self.big_tosum_ar_list[ng])
            d_collct_ar_temp = cuda.to_device(self.big_collct_ar_list[ng])
            d_E_ar_temp = cuda.to_device(self.gpu_E_ar_list[ng])
            d_ne_ar_temp = cuda.to_device(self.gpu_ne_ar_list[ng])
            d_ni_ar_temp = cuda.to_device(self.gpu_ni_ar_list[ng])
            d_ns_ar_temp = cuda.to_device(self.gpu_ns_ar_list[ng])
            d_Te_ar_temp = cuda.to_device(self.gpu_Te_ar_list[ng])
           
            self.d_curr_count_list.append(d_curr_count_temp)
            self.d_currxbins_list.append(d_currxbins_temp)
            self.d_forExbins_list.append(d_forExbins_temp)
            self.d_bigRvec_list.append(d_bigRvec_temp)
            self.d_nc_flag_ar_list.append(d_nc_flag_ar_temp)
            self.d_nc_inds_ar_list.append(d_nc_inds_ar_temp)
            self.d_forrecomb_v_ar_list.append(d_forrecomb_v_ar_temp)
            self.d_data_ar_list.append(d_data_ar_temp)
            self.d_tosum_ar_list.append(d_tosum_ar_temp)
            self.d_collct_ar_list.append(d_collct_ar_temp)
            self.d_E_ar_list.append(d_E_ar_temp)  
            self.d_ne_ar_list.append(d_ne_ar_temp)  
            self.d_ni_ar_list.append(d_ni_ar_temp)  
            self.d_ns_ar_list.append(d_ns_ar_temp)  
            self.d_Te_ar_list.append(d_Te_ar_temp)  

        ## For recombination
        #self.collflag = cp.zeros(self.Nmax)
        #self.max_partners_percell_ever = int(2*self.Nc/self.N)
        #self.PROB_FACTOR = 4
        #self.maxprob_bycells = cp.ones(self.N+1)*0.1
        #self.num_percell = cp.zeros(self.N+1).astype(int)
        #self.partners = cp.zeros(self.Nmax).astype(int)
        #self.potential_partners_percell = cp.zeros((self.N+1, self.max_partners_percell_ever)).astype(int)
        #self.max_partners_percell = cp.zeros(self.N+1).astype(int)
        #self.num_partners_percell = cp.zeros(self.N+1).astype(int)
        #self.used_partner_counts = cp.zeros(self.N+1).astype(int)
        #self.max_partners = 0
        #
        #self.d_collflag = cuda.to_device(self.collflag)
        #self.d_max_partners_percell = cuda.to_device(self.max_partners_percell)
        #self.d_num_partners_percell = cuda.to_device(self.num_partners_percell)
        #self.d_potential_partners_percell = cuda.to_device(self.potential_partners_percell)
        #self.d_used_partner_counts = cuda.to_device(self.used_partner_counts)

        ## For recombination and for counting collisions
        self.collflag_list = []
        self.d_collflag_list = []
        self.did_g0_ar_list = []
        self.did_g1_ar_list = []
        self.did_g2_ar_list = []
        self.did_g3_ar_list = []
        self.d_g0_ar_list = []
        self.d_g1_ar_list = []
        self.d_g2_ar_list = []
        self.d_g3_ar_list = []
        for ng in range(self.NUM_GPUS):
            cp.cuda.Device(ng).use()
            
            collflag_temp = cp.zeros(self.Nmax_list[ng])
            d_collflag_temp = cuda.to_device(collflag_temp)
            self.collflag_list.append(collflag_temp)
            self.d_collflag_list.append(d_collflag_temp)

            did_g0_ar_temp = cp.zeros(self.Nmax_list[ng])
            did_g1_ar_temp = cp.zeros(self.Nmax_list[ng])
            did_g2_ar_temp = cp.zeros(self.Nmax_list[ng])
            did_g3_ar_temp = cp.zeros(self.Nmax_list[ng])
            self.did_g0_ar_list.append(did_g0_ar_temp)
            self.did_g1_ar_list.append(did_g1_ar_temp)
            self.did_g2_ar_list.append(did_g2_ar_temp)
            self.did_g3_ar_list.append(did_g3_ar_temp)
            
            d_g0_ar_temp = cuda.to_device(self.did_g0_ar_list[ng])
            d_g1_ar_temp = cuda.to_device(self.did_g1_ar_list[ng])
            d_g2_ar_temp = cuda.to_device(self.did_g2_ar_list[ng])
            d_g3_ar_temp = cuda.to_device(self.did_g3_ar_list[ng])
            self.d_g0_ar_list.append(d_g0_ar_temp)
            self.d_g1_ar_list.append(d_g1_ar_temp)
            self.d_g2_ar_list.append(d_g2_ar_temp)
            self.d_g3_ar_list.append(d_g3_ar_temp)

        self.ne_toplot = np.zeros((int(num_steps/10), self.N+1))
        self.ni_toplot = np.zeros((int(num_steps/10), self.N+1))
        self.Te_toplot = np.zeros((int(num_steps/10), self.N+1))
        self.E_toplot = np.zeros((int(num_steps/10), self.N))
        self.V_toplot = np.zeros((int(num_steps/10), self.N-1))
        

        # Output frequency, and thus total # outputs
        self.output_freq = int(self.steps_per_cycle)
        self.num_outputs = int(num_steps/ self.output_freq)
        self.out_zdne = np.zeros(self.num_outputs)
        self.out_zdTe = np.zeros(self.num_outputs)


        # For Coulomb kernel
        self.write_inds_ar_list = []
        self.pairing_ar_list = []
        self.num_percell_list = []
        for ng in range(self.NUM_GPUS):
            cp.cuda.Device(ng).use()
            temp_pairing_ar = cp.zeros(self.Nmax_list[ng]).astype(int)
            self.pairing_ar_list.append(temp_pairing_ar) 

            temp_write_inds_ar = cp.zeros(self.N).astype(int)
            self.write_inds_ar_list.append(temp_write_inds_ar)

            temp_num_percell = cp.zeros(self.N).astype(int)
            self.num_percell_list.append(temp_num_percell)
        
        self.ne_CA = np.zeros(self.N+1)
        self.ni_CA = np.zeros(self.N+1)
        self.ns_CA = np.zeros(self.N+1)
        self.Te_CA = np.zeros(self.N+1)
        self.E_CA = np.zeros(self.N)
        self.V_CA = np.zeros(self.N-1)
                    
        self.g0_coeff_avg = 0.
        self.g1_coeff_avg = 0.
        self.g2_coeff_avg = 0.
        self.g3_coeff_avg = 0.
        self.g0_coeffs_ar = np.zeros(self.N + 1) 
        self.g1_coeffs_ar = np.zeros(self.N + 1) 
        self.g2_coeffs_ar = np.zeros(self.N + 1) 
        self.g3_coeffs_ar = np.zeros(self.N + 1) 

        self.tsum_g0 = 0
        self.tsum_g1 = 0
        self.tsum_g2 = 0
        self.tsum_g3 = 0

        self.t1_time = np.zeros((num_steps,self.NUM_GPUS))
        self.t2_time = np.zeros((num_steps,self.NUM_GPUS))
        self.cp_time = np.zeros((num_steps,self.NUM_GPUS)) 
        self.rg_time = np.zeros((num_steps,self.NUM_GPUS)) 
        self.ek_time = np.zeros((num_steps,self.NUM_GPUS)) 
        self.pp_time = np.zeros((num_steps,self.NUM_GPUS)) 
        self.fl_time = np.zeros((num_steps,self.NUM_GPUS)) 

        self._setup_kernels(pyk)

    ## MAIN TIMESTEPPING LOOP
    def run(self, num_steps, name, zero_dim, verbose, restart_cyclenum):
        ## PRE-COMPILING 
        for ng in range(self.NUM_GPUS):
            cp.cuda.Device(ng).use()
            pk.set_device_id(ng)
            self.Ncoll_offset = self.electron_kernel(
                self.Nc_list[ng], self.d_data_ar_list[ng], self.d_tosum_ar_list[ng], self.d_collct_ar_list[ng],
                self.d_E_ar_list[ng], self.d_ne_ar_list[ng], self.d_ni_ar_list[ng], self.d_ns_ar_list[ng],self.d_Te_ar_list[ng],
                self.forgpu_R_vec_list[ng], self.d_bigRvec_list[ng], self.d_currxbins_list[ng], self.d_forExbins_list[ng],
                1e6 * self.nn, self.dt_ratio, self.dt_el, self.d_curr_count_list[ng], self.d_collflag_list[ng], self.d_forrecomb_v_ar_list[ng],
                self.num_blocks, self.threads_per_block, self.d_g0_ar_list[ng], self.d_g1_ar_list[ng], self.d_g2_ar_list[ng], self.d_g3_ar_list[ng]
            )
            self.pairing_ar_list[ng].fill(-1)
            cp.cuda.runtime.deviceSynchronize()
            numba.cuda.synchronize()
            cp.cuda.get_current_stream().synchronize()
        
        self.np_data_ar[0:self.N] = self.E_ar[0:self.N]
        self.np_data_ar[self.N : 2*self.N + 1] = self.ne_ar[0 : self.N + 1]
        self.np_data_ar[2*self.N + 1 : 3*self.N + 2] = self.ni_ar[0:self.N + 1]
        self.np_data_ar[3*self.N + 2: 4*self.N + 3] = self.ns_ar[0:self.N + 1]
        self.np_data_ar[4*self.N + 3: 5*self.N + 4] = self.Te_ar[0:self.N + 1]

        print("Launching main Parla task\n")
        @spawn(placement=cpu, vcus=0)
        async def main_task():

            mytaskspace = TaskSpace("mytaskspace")
            ta = time.time()
            for bigstep in range(num_steps):
                ## Task 0: GPU Steps
                for ng in range(self.NUM_GPUS):
                    if (bigstep == 0):
                        print("Launching first step on GPU",ng,". No further steps will have prints")
                    deps = [mytaskspace[1,bigstep-1,0]] if bigstep != 0 else []
                    @spawn(mytaskspace[0,bigstep,ng], placement=gpu(ng), dependencies=deps)
                    def gpu_stuff():
                        cp.cuda.Device(ng).use()
                        pk.set_device_id(ng)

                        ## COPY TO GPU
                        self.cp_data_ar_list[ng][:] = cp.asarray(self.np_data_ar[:])
                        self.gpu_E_ar_list[ng][:] = self.cp_data_ar_list[ng][0 : self.N] 
                        self.gpu_ne_ar_list[ng][:] = self.cp_data_ar_list[ng][self.N : 2*self.N+1] 
                        self.gpu_ni_ar_list[ng][:] = self.cp_data_ar_list[ng][2*self.N+1 : 3*self.N+2] 
                        self.gpu_ns_ar_list[ng][:] = self.cp_data_ar_list[ng][3*self.N+2 : 4*self.N+3] 
                        self.gpu_Te_ar_list[ng][:] = self.cp_data_ar_list[ng][4*self.N+3 : 5*self.N+4] 
                        self.curr_count_list[ng].fill(0)
                        self.collflag_list[ng].fill(False)
                        
                        self.forgpu_R_vec_list[ng][0:self.Nc_list[ng],:] = cp.random.rand(self.Nc_list[ng],7)

                        ## ELECTRON KERNEL
                        self.Ncoll_offset = self.electron_kernel(
                            self.Nc_list[ng], self.d_data_ar_list[ng], self.d_tosum_ar_list[ng], self.d_collct_ar_list[ng],
                            self.d_E_ar_list[ng], self.d_ne_ar_list[ng], self.d_ni_ar_list[ng], self.d_ns_ar_list[ng],self.d_Te_ar_list[ng],
                            self.forgpu_R_vec_list[ng], self.d_bigRvec_list[ng], self.d_currxbins_list[ng], self.d_forExbins_list[ng],
                            1e6 * self.nn, self.dt_ratio, self.dt_el, self.d_curr_count_list[ng], self.d_collflag_list[ng], self.d_forrecomb_v_ar_list[ng],
                            self.num_blocks, self.threads_per_block, self.d_g0_ar_list[ng], self.d_g1_ar_list[ng], self.d_g2_ar_list[ng], self.d_g3_ar_list[ng]
                        )

                        ## POST PROCESSING            
                        self.nni_list[ng] = int(self.curr_count_list[ng][0])
                        
                        ## FILLS AND BINNING
                        self.big_data_ar_list[ng][self.Nnew_list[ng]:,self.vx:self.vz+1].fill(0)
                        self.big_tosum_ar_list[ng][self.Nnew_list[ng]:,:].fill(0)
                        self.big_tosum_ar_list[ng][:,self.ai:self.ae+1].fill(0)
                        self.big_forE_xbins_list[ng][0:self.Nnew_list[ng]] = (self.big_data_ar_list[ng][0:self.Nnew_list[ng], self.xx]/self.dx).astype(int)
                        self.big_curr_xbins_list[ng][0:self.Nnew_list[ng]] = ((self.big_data_ar_list[ng][0:self.Nnew_list[ng], self.xx] + 0.5*self.dx)/self.dx).astype(int)
                        # Update # particles
                        self.Nc_list[ng] = self.Nnew_list[ng]
                        self.curr_count_list[ng].fill(0)
                        self.collflag_list[ng].fill(0)
                        cp.cuda.runtime.deviceSynchronize()
                        numba.cuda.synchronize()
                        cp.cuda.get_current_stream().synchronize()

                ## Task 1: Sync + CPU Work
                deps = [mytaskspace[0,bigstep,gg] for gg in range(self.NUM_GPUS)]
                @spawn(mytaskspace[1,bigstep,0],placement=cpu,dependencies=deps)
                def barrier_task():
                    if (bigstep == 0):
                        print("Launching first step on CPU. No further steps will have prints")

                    # Making sure we calc these properly each step
                    self.ne_ar[:] = np.sum(self.ne_ar_sum,axis=1)
                    self.ni_src[:] = np.sum(self.ni_src_sum,axis=1)
                    self.ns_src[:] = np.sum(self.ns_src_sum,axis=1)
                    self.nrg_ar[:] = np.sum(self.nrg_ar_sum,axis=1)
                   
                    # Update time
                    self.curr_t += self.dt_big
            
                    # PDE Solves - Fluid + EF
                    self.ni_ar, self.ns_ar = self.heavies_kernel_fluid(
                        self.E_ar, self.ni_ar, self.ni_rhs, self.Ji_ar,
                        self.ns_ar, self.ns_rhs, self.Js_ar, self.dx,
                        self.dt_big, self.mu_i, self.D_i,self.D_s, self.N
                    )
                    self.Vcarry_ar, self.E_ar = self.EF_kernel(
                        self.E_ar, self.ne_ar, self.ni_ar, self.curr_t,
                        self.V_ar, self.V_rhs, self.Vc_diag, self.Vc_lower_diag,
                        self.V_tempy, self.N, self.dx, self.epsilon, self.V0, self.freq
                    )
                    self.np_data_ar[0:self.N] = self.E_ar[0:self.N]
                    self.np_data_ar[self.N : 2*self.N + 1] = self.ne_ar[0 : self.N + 1]
                    self.np_data_ar[2*self.N + 1 : 3*self.N + 2] = self.ni_ar[0:self.N + 1]
                    self.np_data_ar[3*self.N + 2: 4*self.N + 3] = self.ns_ar[0:self.N + 1]
                    self.np_data_ar[4*self.N + 3: 5*self.N + 4] = self.Te_ar[0:self.N + 1]
            
            mytaskspace.wait()
            tb = time.time()
            print("Total runtime =", tb - ta)
            print("\n\n")

    def _chol_EF(self, N: int, dx: float):
        """
        Set up Cholesky arrays
        """

        Ac = np.zeros((2,N-1))
        Ac[0,1:N-1] = -1./(dx**2)
        Ac[1,0:N-1] = 2./(dx**2)
        c = cholesky_banded(Ac) 
        Vc_lower_diag = np.zeros(N-1)
        Vc_lower_diag[1:N-1] = np.copy(c[0,1:N-1])
        Vc_diag = np.zeros(N-1)
        Vc_diag[0:N-1] = np.copy(c[1,0:N-1])

        V_tempy = np.zeros(N + 1)
        return (Vc_diag, Vc_lower_diag, V_tempy)

    def _init_kernels(self, zero_dim, pyk, recomb, null_coll) -> None:
        """
        Initialize the kernels stored as member variables

        :param pyk: whether to use pykokkos versions of kernels
        """

        print("Loading PyKokkos kernels\n")


        from kernels.electron_kernel_pyk_1D_kevin_global import electron_kernel
        from kernels.EF_kernel_1D import EF_kernel
        from kernels.heavies_kernel_1D import heavies_kernel_fluid
        from kernels.data_write_1D import data_write        
       

        self.EF_kernel = EF_kernel
        self.electron_kernel = electron_kernel
        self.heavies_kernel_fluid = heavies_kernel_fluid
        self.data_write = data_write

    def _setup_kernels(self, pyk: bool) -> None:
        """
        Setup the arrays and functor objects as needed by the kernels

        :param pyk: whether pykokkos is enabled
        """

        if pyk:
            import pykokkos as pk
            pk.set_default_space(pk.Cuda)

            for ng in range(self.NUM_GPUS):        
                cp.cuda.Device(ng).use()
                pk.set_device_id(ng)
                self.d_curr_count_list[ng] = pk.array(cp.asarray(self.d_curr_count_list[ng]))
                self.d_forExbins_list[ng] = pk.array(cp.asarray(self.d_forExbins_list[ng]))
                self.d_bigRvec_list[ng] = pk.array(cp.asarray(self.d_bigRvec_list[ng]))
                self.d_data_ar_list[ng] = pk.array(cp.asarray(self.d_data_ar_list[ng]))
                self.d_tosum_ar_list[ng] = pk.array(cp.asarray(self.d_tosum_ar_list[ng]))
                self.d_E_ar_list[ng] = pk.array(cp.asarray(self.gpu_E_ar_list[ng]))
                self.d_ne_ar_list[ng] = pk.array(cp.asarray(self.gpu_ne_ar_list[ng]))
                self.d_ni_ar_list[ng] = pk.array(cp.asarray(self.gpu_ni_ar_list[ng]))
                self.d_ns_ar_list[ng] = pk.array(cp.asarray(self.gpu_ns_ar_list[ng]))
                self.d_Te_ar_list[ng] = pk.array(cp.asarray(self.gpu_Te_ar_list[ng]))
                self.d_currxbins_list[ng] = pk.array(cp.asarray(self.d_currxbins_list[ng]))
                self.d_collflag_list[ng] = pk.array(cp.asarray(self.d_collflag_list[ng]))
                self.d_g0_ar_list[ng] = pk.array(cp.asarray(self.d_g0_ar_list[ng]))
                self.d_g1_ar_list[ng] = pk.array(cp.asarray(self.d_g1_ar_list[ng]))
                self.d_g2_ar_list[ng] = pk.array(cp.asarray(self.d_g2_ar_list[ng]))
                self.d_g3_ar_list[ng] = pk.array(cp.asarray(self.d_g3_ar_list[ng]))
                self.d_collct_ar_list[ng] = pk.array(cp.asarray(self.d_collct_ar_list[ng]))
