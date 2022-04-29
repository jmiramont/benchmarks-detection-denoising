import numpy as np
from scipy.integrate import cumtrapz

# for R to work
# print('Importing modules for R to work...')
import rpy2.robjects as robjects
# from rpy2.robjects.packages import importr
# Activate automatic conversion of numpy floats and arrays to corresponding R objects
from rpy2.robjects import numpy2ri
# print('Finished.')
# numpy2ri.activate() #numpy2ri.deactivate()
# from spatstat_interface.utils import to_pandas_data_frame
# print('Importing SpatstatInterface...')
from spatstat_interface.interface import SpatstatInterface
# print('Finished.')

from benchmark_demo.utilstf import *
# import time
from methods.contours_utils import zeros_finder
# import pickle


class ComputeStatistics():
    """A class that encapsulates the code for computing functional statistics.

    """
    
    def __init__(self, spatstat=None):
        if spatstat is None:
            
        # print('Starting spatstat-interface...')
            self.spatstat = SpatstatInterface(update=False)  
        else:
            self.spatstat = spatstat
        
        self.spatstat.import_package("core", "geom", update=False)
        


    def compute_positions_and_bounds(self, pos):
        u_r = robjects.FloatVector(pos[:, 1])                       
        v_r = robjects.FloatVector(pos[:, 0])
        bounds_u = np.array([np.min(pos[:, 1]), np.max(pos[:, 1])]) 
        bounds_v = np.array([np.min(pos[:, 0]), np.max(pos[:, 0])])

        return u_r, v_r, bounds_u, bounds_v

    def compute_Lest(self, pos, r_des):
        radius_r = robjects.FloatVector(r_des)                      

        u_r, v_r, bounds_u, bounds_v = self.compute_positions_and_bounds(pos)
        b_u = robjects.FloatVector(bounds_u)        
        b_v = robjects.FloatVector(bounds_v)

        ppp_r = self.spatstat.geom.ppp(u_r, v_r, b_u, b_v)
        numpy2ri.deactivate()
        L_r = self.spatstat.core.Lest(ppp_r, r=radius_r) 

        radius = np.array(L_r.rx2('r')) 
        Lborder = np.array(L_r.rx2('border'))
        Ltrans = np.array(L_r.rx2('trans'))
        Liso = np.array(L_r.rx2('iso'))

        L = [Lborder, Ltrans, Liso]
        return L[2], radius


    def compute_Fest(self, pos, r_des = None, estimator_type = 'rs'):
        radius_r = robjects.FloatVector(r_des)

        u_r, v_r, bounds_u, bounds_v = self.compute_positions_and_bounds(pos)
        b_u = robjects.FloatVector(bounds_u)        
        b_v = robjects.FloatVector(bounds_v)
        
        ppp_r = self.spatstat.geom.ppp(u_r, v_r, b_u, b_v)
        numpy2ri.deactivate()
        F_r = self.spatstat.core.Fest(ppp_r, r=radius_r) 
        # F_r = self.spatstat.core.Fest(ppp_r) 

        radius = np.array(F_r.rx2('r')) 
        Fborder = np.array(F_r.rx2(estimator_type))
        
        return Fborder, radius


def pairCorrPlanarGaf(r, L):
    a = 0.5*L*r**2
    num = (np.sinh(a)**2+L**2/4*r**4)*np.cosh(a)-L*r**2*np.sinh(a)
    den = np.sinh(a)**3
    rho = num/den

    if r[0] == 0:
        rho[0] = 0
    return rho


def Kfunction(r, rho):
    K = np.zeros(len(rho))
    K[1:] = 2*np.pi*cumtrapz(r*rho, r)

    return K


def ginibreGaf(r, c):
    rho = 1-np.exp(-c*r**2)
    return rho


def compute_S0(radius, statistic = None, Sm = None):
    """_summary_

    Args:
        radius (_type_): _description_
        statistic (_type_, optional): _description_. Defaults to None.
        Sm (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if statistic == 'L':
        # compute true GAF Lfunc
        rho_gaf = pairCorrPlanarGaf(radius, np.pi)
        Krho_gaf = Kfunction(radius, rho_gaf)
        Lrho_gaf = np.sqrt(Krho_gaf/np.pi)   # El valor teórico que necesita para comparar.
        return Lrho_gaf

    if statistic != 'L':
        if Sm is not None:
            Smean = np.mean(Sm, axis=0)
            return Smean
        else:
            return None


def compute_T_statistic(radius, rmax, Sm, S0, pnorm=2):
    """_summary_

    Args:
        radius (_type_): _description_
        rmax (_type_): _description_
        Sm (_type_): _description_
        S0 (_type_): _description_
        pnorm (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    if len(Sm.shape) == 1:
        Sm.resize((1,Sm.size))

    S0.resize((1,S0.size))
    # t2 = np.sqrt(np.cumsum((Sm-S0)**2, axis=1))      
    # tm = np.zeros_like(Sm)
    tm = np.zeros((Sm.shape[0], len(rmax)))
    for k in range(len(rmax)):
        int_ub = np.where(rmax[k]<=radius)[0][0]
        for i in range(Sm.shape[0]):
            tm[i,k] = np.linalg.norm(Sm[i,:int_ub+1]-S0[0,:int_ub+1], ord=pnorm)
    
    return tm


def compute_mc_sim(signal,
                    sc=None,
                    Nfft=None,
                    MC_reps = 199,
                    statistic='L',
                    pnorm = 2,
                    radius = None,
                    rmax = None):
    """Compute the Montecarlo simulations.

    Args:
        signal (_type_): _description_
        sc (_type_, optional): _description_. Defaults to None.
        Nfft (_type_, optional): _description_. Defaults to None.
        MC_reps (int, optional): _description_. Defaults to 199.
        statistic (str, optional): _description_. Defaults to 'L'.
        pnorm (int, optional): _description_. Defaults to 2.
        radius (_type_, optional): _description_. Defaults to None.
        rmax (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """

    # rmax*(base)/np.sqrt(2*base) Use this for rmax in samples!
    N = len(signal)
    if Nfft is None:
        Nfft = N
        
    if radius is None:
        # radius = np.linspace(0, 4.0, 50)
        radius = np.arange(0.0, 4.0, 0.01)

    if isinstance(rmax,float):
        rmax = (rmax,)

    if rmax is None:
        rmax = radius
        
    g, T = get_round_window(Nfft)

    # Compute empirical statistic Sexp:
    stf, _, _, _ = get_spectrogram(signal, window=g)
    pos_exp = zeros_finder(stf)/T
    simulation_pos = list()

    # Compute noise distribution of zeros.
    # start = time.time()
    for i in range(MC_reps):   
        x = np.random.randn(N)
        stf, _, _, _ = get_spectrogram(x, window=g)
        # pos_aux = find_zeros_of_spectrogram(stf)
        pos = zeros_finder(stf)/T    
        simulation_pos.append(pos)

    # end = time.time()
    # print(end-start)
    
    output = dict()
    for sts in statistic:
        tm, t_exp = compute_statistics(sts,sc,simulation_pos,pos_exp,radius,rmax,pnorm)
        output[sts] = (tm,t_exp)
   
    return output, radius

def compute_statistics(sts, sc, simulation_pos, pos_exp, radius, rmax, pnorm):
    """Compute the given functional statistics.

    Args:
        sts (_type_): _description_
        sc (_type_): _description_
        simulation_pos (_type_): _description_
        pos_exp (_type_): _description_
        radius (_type_): _description_
        rmax (_type_): _description_
        pnorm (_type_): _description_

    Returns:
        _type_: _description_
    """

    MC_reps = len(simulation_pos)
    tm = np.zeros((MC_reps, len(rmax)))
    Sm = np.zeros((MC_reps, len(radius)))

    # A dictionary with the functions to compute the statistics.
    if sc is None:
        sc = ComputeStatistics()

    stats_dict = dict()
    stats_dict['L'] = sc.compute_Lest
    stats_dict['F'] = sc.compute_Fest
    stats_dict['Frs'] = lambda a,b: sc.compute_Fest(a,b, estimator_type='rs')
    stats_dict['Fkm'] = lambda a,b: sc.compute_Fest(a,b, estimator_type='km')
    stats_dict['Fcs'] = lambda a,b: sc.compute_Fest(a,b, estimator_type='cs')

    # Compute empirical S.
    Sexp, _ = stats_dict[sts](pos_exp, radius)

    # Compute the statistic Sm for the m-th realization of noise.
    for i, pos in enumerate(simulation_pos):
        Sm[i,:], _ = stats_dict[sts](pos, radius) 

    S0 = compute_S0(radius, statistic = sts, Sm=Sm)
    # Compute the T statistic as T= Sm-S_0 where S_0 is the theoretic or average value.
    tm = compute_T_statistic(radius, rmax, Sm, S0, pnorm=pnorm)
    # sort values of t for null hypothesis
    tm = np.sort(tm, axis=0)[::-1, :]
    # Compute empirical statistic texp
    t_exp = compute_T_statistic(radius, rmax, Sexp, S0, pnorm=pnorm)

    return tm, t_exp.squeeze()


def compute_hyp_test(signal, 
                    sc=None, 
                    MC_reps = 199, 
                    alpha = 0.05, 
                    statistic='L',
                    pnorm = 2, 
                    radius=None, 
                    rmax=None,
                    return_values=False):
    """ Compute hypothesis tests based on Montecarlo simulations.

    Args:
        signal (ndarray): Numpy ndarray with the signal.
        sc (ComputeStatistics, optional): This is an object of the ComputeStatistics
        class, that encapsulates the initialization of the spatstat-interface python
        package. This allows avoiding reinitialize the interface each time. 
        Defaults to None.
        MC_reps (int, optional): Repetitions of the Montecarlo simulations. 
        Defaults to 199.
        alpha (float, optional): Significance of the tests. Defaults to 0.05.
        statistic (str, optional): Functional statistic computed on the point process 
        determined by the zeros of spectrogram of the signal. Defaults to 'L'.
        pnorm (int, optional): Summary statistics for the envelope-tests. Common values 
        for this parameter are "np.inf" for the supremum norm, or "2" for the usual 2 
        norm. Defaults to 2.
        radius (float, optional): Vector of radius used for the computation of the 
        functional statistics. Defaults to None.
        rmax (float, optional): Maximum radius to compute the test. One test per value 
        in rmax is compute. If it is None, the values given in "radius" are used. 
        Defaults to None.
        return_values (bool, optional): If False, returns a dictionary with the 
        results of the test for each statistic and value of rmax. If True, also returns 
        the empirical statistic and the simulated statistics. Defaults to False.

    Returns:
        dict: Returns a dictionary with the results. If more than one statistic is 
        given as input parameter, the dictionary will have one entry per statistic.
    """
    
    N = len(signal)
    Nfft = 2*N
    k = int(np.floor(alpha*(MC_reps+1))) # corresponding k value
    if isinstance(statistic,str):
        statistic = (statistic,)

    output_dict, radius = compute_mc_sim(signal,
                                        sc,
                                        Nfft,
                                        MC_reps=MC_reps,
                                        statistic=statistic,
                                        pnorm=pnorm,
                                        radius=radius,
                                        rmax=rmax)
    
    for sts in statistic:
        tm, t_exp = output_dict[sts]
        reject_H0 = np.zeros(tm.shape[1], dtype=bool)
        reject_H0[np.where(t_exp > tm[k])] = True
        if return_values:
            output_dict[sts] = {'reject_H0': reject_H0,
                                'tm': tm,
                                't_exp': t_exp,
                                'k': k}
        else:
            output_dict[sts] = reject_H0
    
    if len(statistic)>1:    
        return output_dict 
    else: # returns (tm, t_exp) if only one statistic was computed.
        return output_dict[statistic[0]] 



def compute_scale(signal, Nfft, sc):
    # sc = ComputeStatistics()
    # output = np.zeros((reps,len(radius)))
    statistics = 'F' #('L','Frs','Fcs','Fkm')
    pnorm = np.inf
    radius = np.arange(0.0, 4.0, 0.01)

    hyp_test_dict = compute_hyp_test(signal,
                                sc=sc,
                                MC_reps = 199,
                                alpha = 0.05,
                                statistic=statistics,
                                pnorm = pnorm,
                                radius = radius,
                                return_values=True)

    k =  hyp_test_dict['k']
    tm = hyp_test_dict['tm']
    t_exp = hyp_test_dict['t_exp']
    reject_H0 = hyp_test_dict['reject_H0']

    radi_of_rejection = radius[np.where(reject_H0 == True)]
    t_exp_rejection = t_exp[np.where(reject_H0 == True)]
    t_exp_rejection *= 10000
    t_exp_rejection = np.round(t_exp_rejection)/10000
    max_t_exp = np.max(t_exp_rejection)

    radius_of_rejection = 0.0

    for i in range(len(t_exp_rejection)-5):
        if np.all(t_exp_rejection[i:i+5]==max_t_exp):
            radius_of_rejection = radi_of_rejection[i]
            break

    if radius_of_rejection < 0.6:
        radius_of_rejection = 0.6
    
    return radius_of_rejection


    # def spatialStatsFromR(pos):

    # # load spatstat
    # spatstat = importr('spatstat.geom')
    # spatstatCore = importr('spatstat.core')

    # u_r = robjects.FloatVector(pos[:, 0])
    # v_r = robjects.FloatVector(pos[:, 1])

    # bounds_u = np.array([np.min(pos[:, 0]), np.max(pos[:, 0])])
    # bounds_v = np.array([np.min(pos[:, 1]), np.max(pos[:, 1])])

    # b_u = robjects.FloatVector(bounds_u)
    # b_v = robjects.FloatVector(bounds_v)

    # ppp_r = spatstat.ppp(u_r, v_r, b_u, b_v)

    # K_r = spatstatCore.Kest(ppp_r)
    # L_r = spatstatCore.Lest(ppp_r)
    # pcf_r = spatstatCore.pcf(ppp_r)

    # radius = np.array(K_r[0])
    # Kborder = np.array(K_r[2])

    # if len(pos[:, 0]) < 1024:
    #     Ktrans = np.array(K_r[3])
    #     Kiso = np.array(K_r[4])

    #     K = [Kborder, Ktrans, Kiso]
    # else:
    #     K = [Kborder]

    # Lborder = np.array(L_r[2])
    # Ltrans = np.array(L_r[3])
    # Liso = np.array(L_r[4])

    # L = [Lborder, Ltrans, Liso]

    # pcftrans = np.array(pcf_r[2])
    # pcfiso = np.array(pcf_r[3])

    # pcf = [pcftrans, pcfiso]

    # return radius, K, L, pcf


