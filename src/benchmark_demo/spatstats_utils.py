import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg
import importlib.util 
rpy2_is_present = importlib.util.find_spec('rpy2')
if rpy2_is_present:
    import rpy2.robjects as robjects
    from rpy2.robjects import numpy2ri
    from rpy2.robjects.packages import importr
    from spatstat_interface.interface import SpatstatInterface

from benchmark_demo.utilstf import find_zeros_of_spectrogram, get_round_window, get_spectrogram
from scipy.integrate import cumtrapz
import multiprocessing
import pickle

def compute_positions_and_bounds(pos):
        """Parse the python vector of positions of points ```pos``` into a R object.
        Then, computes the bounds of the observation window for the computation of 
        functional statistics. 

        Args:
            pos (numpy.array): Array with the coordinates of each point in the plane.

        Returns:
            u_r (FloatVector): R float vector with horizontal coordinates of the points.
            v_r (FloatVector): R float vector with vertical coordinates of the points.
            bounds_u (numpy.array): Array with the horizontal bounds of the window.
            bounds_v (numpy.array): Array with the vertical bounds of the window.
        """

        u_r = robjects.FloatVector(pos[:, 1])                       
        v_r = robjects.FloatVector(pos[:, 0])
        bounds_u = np.array([np.min(pos[:, 1]), np.max(pos[:, 1])]) 
        bounds_v = np.array([np.min(pos[:, 0]), np.max(pos[:, 0])])
        b_u = robjects.FloatVector(bounds_u)        
        b_v = robjects.FloatVector(bounds_v)

        return u_r, v_r, b_u, b_v



def get_white_noise_zeros(stft_params, complex_noise=False):
    """Get the zeros of the spectrogram of (real or complex) white Gaussian noise.
    If the noise generated is real, the zeros considered are those "far" from the time
    axis.

    Args:
        stft_params (_type_): _description_
        complex_noise (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    N, Nfft = stft_params
    wnoise = np.random.randn(N)

    if complex_noise:
        wnoise += 1j*np.random.randn(N)

    # Get round window and scale for normalization.    
    g, T = get_round_window(Nfft)
    # g, T = roundgauss(Nfft,k=1e-6)
    # spec = Spectrogram(wnoise, fwindow=g, n_fbins=Nfft)
    # spec.run()
    # stf = np.abs(spec.tfr[0:N+1,:])
    stf, _, _, _ = get_spectrogram(wnoise, window=g)
    pos = find_zeros_of_spectrogram(np.abs(stf))

    # Get only the zeros "far" from the real axis if the noise is real.
    if not complex_noise:
        margin = T
        valid_ceros = np.zeros((pos.shape[0],),dtype=bool)
        valid_ceros[(margin<pos[:,0])] = True 
        pos = pos[valid_ceros]

    return pos


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
        

    def compute_Lest(self, pos, r_des):
        """ Compute the functional statistic L, also referred as variance-normalized 
        Ripley's K, for a point-process, the coordinates of the points given in ```pos```.
        L is computed for the radius given in ```r_des```.

        Args:
            pos (_type_): Positions of the points in the point-process realization.
            r_des (_type_): Desired radius to compute the L statistic.

        Returns:
            _type_: _description_
        """

        radi = robjects.FloatVector(r_des)                      
        u_r, v_r, b_u, b_v = compute_positions_and_bounds(pos)
        ppp_r = self.spatstat.geom.ppp(u_r, v_r, b_u, b_v)
        numpy2ri.deactivate()
        L_r = self.spatstat.core.Lest(ppp_r, r=radi) 

        radius = np.array(L_r.rx2('r')) 
        Lborder = np.array(L_r.rx2('border'))
        Ltrans = np.array(L_r.rx2('trans'))
        Liso = np.array(L_r.rx2('iso'))

        L = [Lborder, Ltrans, Liso]
        return L[2], radius


    def compute_Fest(self, pos, r_des = None, correction = 'rs', var_stb=False):
        """Compute the functional statistic F, also referred as empty space function, 
        for a point-process, the coordinates of the points given in ```pos```.
        F is computed for the radius given in ```r_des```.
        Args:
            pos (_type_): Positions of the points in the point-process realization.
            r_des (_type_, optional): Desired radius to compute the L statistic. 
            Defaults to None.
            estimator_type (str, optional): _description_. Defaults to 'rs'.

        Returns:
            _type_: _description_
        """

        radius_r = robjects.FloatVector(r_des)
        u_r, v_r, b_u, b_v = compute_positions_and_bounds(pos)        
        ppp_r = self.spatstat.geom.ppp(u_r, v_r, b_u, b_v)
        numpy2ri.deactivate()
        F_r = self.spatstat.core.Fest(ppp_r, r=radius_r) 
        # F_r = self.spatstat.core.Fest(ppp_r) 
        radius = np.array(F_r.rx2('r')) 
        Fborder = np.array(F_r.rx2(correction))

        if var_stb:
            Fborder = np.arcsin(np.sqrt(Fborder))

        return Fborder, radius


def pairCorrPlanarGaf(r, L):
    """_summary_

    Args:
        r (_type_): _description_
        L (_type_): _description_

    Returns:
        _type_: _description_
    """
    a = 0.5*L*r**2
    num = (np.sinh(a)**2+L**2/4*r**4)*np.cosh(a)-L*r**2*np.sinh(a)
    den = np.sinh(a)**3
    rho = num/den

    if r[0] == 0:
        rho[0] = 0
    return rho


def Kfunction(r, rho):
    """_summary_

    Args:
        r (_type_): _description_
        rho (_type_): _description_

    Returns:
        _type_: _description_
    """
    K = np.zeros(len(rho))
    K[1:] = 2*np.pi*cumtrapz(r*rho, r)

    return K


def ginibreGaf(r, c):
    """_summary_

    Args:
        r (_type_): _description_
        c (_type_): _description_

    Returns:
        _type_: _description_
    """
    rho = 1-np.exp(-c*r**2)
    return rho


def compute_S0(radius, Sexp, statistic = None, Sm = None):
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
            Smean = np.mean(np.concatenate((Sm,Sexp.reshape(1,Sexp.size)),axis=0), axis=0)
            return Smean
        else:
            return None


def compute_T_statistic(radius, rmax, Sm, S0, pnorm=2, rmin=0.0, one_sided=False):
    """Computes the T sumary statistic using the norm given as an argument.

    Args:
        radius (_type_): _description_
        rmax (_type_): _description_
        Sm (_type_): _description_
        S0 (_type_): _description_
        pnorm (int, optional): _description_. Defaults to 2.
        rmin (float, optional): _description_. Defaults to 0.0.
        one_sided (bool, optional): _description_. Defaults to False.

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
        int_lb = np.where(rmin<=radius)[0][0]    
        int_ub = np.where(rmax[k]<=radius)[0][0]


        for i in range(Sm.shape[0]):
            if int_ub<int_lb:
                tm[i,k] = 0
            else:
                difS = (S0[0,int_lb:int_ub+1]-Sm[i,int_lb:int_ub+1])
                if one_sided:
                    difS = difS[difS>0]

                tm[i,k] = np.linalg.norm(difS, ord=pnorm)

                # Use the squared 2-norm if this is the case.
                if pnorm==2:
                    tm[i,k] = tm[i,k]**2

    return tm


def compute_statistics(sts, cs, simulation_pos, pos_exp, radius, rmax, pnorm, one_sided):
    """ Compute the given functional statistics.

    Args:
        sts (_type_): _description_
        cs (_type_): _description_
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
    if cs is None:
        cs = ComputeStatistics()

    stats_dict = dict()
    stats_dict['L'] = cs.compute_Lest
    stats_dict['F'] = cs.compute_Fest
    stats_dict['Frs'] = lambda a,b: cs.compute_Fest(a,b, correction='rs')
    stats_dict['Frs_vs'] = lambda a,b: cs.compute_Fest(a,b, correction='rs', var_stb=True)
    stats_dict['Fkm'] = lambda a,b: cs.compute_Fest(a,b, correction='km')
    stats_dict['Fkm_vs'] = lambda a,b: cs.compute_Fest(a,b, correction='km', var_stb=True)
    stats_dict['Fcs'] = lambda a,b: cs.compute_Fest(a,b, correction='cs')

    # Compute empirical S.
    Sexp, _ = stats_dict[sts](pos_exp, radius)

    # Compute the statistic Sm for the m-th realization of noise.
    for i, pos in enumerate(simulation_pos):
        Sm[i,:], _ = stats_dict[sts](pos, radius) 

    S0 = compute_S0(radius, Sexp, statistic = sts, Sm=Sm)
    # Compute the T statistic as T= Sm-S_0 where S_0 is the theoretic or average value.
    tm = compute_T_statistic(radius, rmax, Sm, S0, pnorm=pnorm,one_sided=one_sided)
    # sort values of t for null hypothesis
    tm = np.sort(tm, axis=0)[::-1, :]
    # Compute empirical statistic texp
    t_exp = compute_T_statistic(radius, rmax, Sexp, S0, pnorm=pnorm,one_sided=one_sided)

    return tm, t_exp.squeeze(), Sm, Sexp, S0


def compute_monte_carlo_sims(signal,
                    cs=None,
                    Nfft=None,
                    MC_reps = 199,
                    statistic='L',
                    pnorm = 2,
                    radius = None,
                    rmax = None,
                    one_sided=False):
    """Compute the Monte Carlo simulations.

    Args:
        signal (_type_): _description_
        cs (_type_, optional): _description_. Defaults to None.
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
        Nfft = 2*N
        
    if radius is None:
        # radius = np.linspace(0, 4.0, 50)
        radius = np.arange(0.0, 4.0, 0.01)

    if isinstance(rmax,float):
        rmax = (rmax,)

    if rmax is None:
        rmax = radius

    # Get round window and scale for normalization.    
    g, T = get_round_window(Nfft)

    # Compute empirical statistic Sexp:
    stf, _, _, _ = get_spectrogram(signal, window=g)
    pos_exp = find_zeros_of_spectrogram(stf)/T
    
    # Compute noise distribution of zeros.
    simulation_pos = list()
    # start = time.time() 
    for i in range(MC_reps):   
        pos = get_white_noise_zeros([N, Nfft])/T     
        simulation_pos.append(pos)
    # end = time.time()
    # print(end-start)
    
    output = dict()
    for sts in statistic:
        tm, t_exp, Sm, Sexp, S0 = compute_statistics(sts,
                                                    cs,
                                                    simulation_pos,
                                                    pos_exp,
                                                    radius,
                                                    rmax,
                                                    pnorm,
                                                    one_sided)
        output[sts] = (tm,t_exp, Sm, Sexp, S0)
   
    return output, radius


def compute_envelope_test(signal, 
                    cs=None, 
                    MC_reps = 199, 
                    alpha = 0.05, 
                    statistic='L',
                    pnorm = 2, 
                    radius=None, 
                    rmax=None,
                    one_sided=False,
                    return_values=False):
    """ Compute hypothesis tests based on Monte Carlo simulations.

    Args:
        signal (ndarray): Numpy ndarray with the signal.
        cs (ComputeStatistics, optional): This is an object of the ComputeStatistics
        class, that encapsulates the initialization of the spatstat-interface python
        package. This allows avoiding reinitialize the interface each time. 
        Defaults to None.
        MC_reps (int, optional): Repetitions of the Monte Carlo simulations. 
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

    output_dict, radius = compute_monte_carlo_sims(signal,
                                        cs,
                                        Nfft,
                                        MC_reps=MC_reps,
                                        statistic=statistic,
                                        pnorm=pnorm,
                                        radius=radius,
                                        rmax=rmax,
                                        one_sided=one_sided)
    
    for sts in statistic:
        tm, t_exp, Sm, Sexp, S0 = output_dict[sts]
        reject_H0 = np.zeros(tm.shape[1], dtype=bool)
        reject_H0[np.where(t_exp > tm[k])] = True

        if reject_H0.size == 1:
            reject_H0 = reject_H0[0]

        if return_values:
            output_dict[sts] = {'reject_H0': reject_H0,
                                'tm': tm,
                                't_exp': t_exp,
                                'Sm':Sm,
                                'Sexp':Sexp, 
                                'S0':S0,
                                'k': k,
                                'radius':radius}
        else:
            output_dict[sts] = reject_H0
    
    if len(statistic)>1:    
        return output_dict 
    else: # returns (tm, t_exp) if only one statistic was computed.
        return output_dict[statistic[0]] 


def generate_white_noise_zeros_pp(N, nsim, complex_noise=False, parallel=False):
    # STFT parameters.
    Nfft = 2*N
    T = np.sqrt(Nfft)

    # R packages.
    rbase = importr('base')
    spatstat = SpatstatInterface(update=False)
    spatstat.import_package("core", "geom", update=False)
    parallel_list = [[N,Nfft] for i in range(nsim)]

    white_noise_func = lambda params: get_white_noise_zeros(params, 
                                                        complex_noise=complex_noise)
    if parallel:
        nprocesses = 4
        pool = multiprocessing.Pool(processes=nprocesses) 
        list_of_zeros = pool.map(white_noise_func, parallel_list) 
        pool.close() 
        pool.join()
    else:
        list_of_zeros = map(white_noise_func,parallel_list) 

    list_ppp = rbase.list()
    for pos in list_of_zeros:
        pos = np.array(pos)/T
        u_r, v_r, b_u, b_v = compute_positions_and_bounds(pos)
        ppp_noise = spatstat.geom.ppp(u_r, v_r, b_u, b_v)
        list_ppp = rbase.c(list_ppp, spatstat.geom.as_solist(ppp_noise))

    list_ppp = spatstat.geom.as_solist(list_ppp)
    return list_ppp
    

def compute_rank_envelope_test(signal,
                                fun='Fest',
                                correction='none',
                                nsim=2499,
                                alpha=0.05,
                                rmin=0.0, 
                                rmax=1.2,
                                ptype = 'conservative',
                                ppp_sim=None,
                                return_dic = False,
                                **kwargs):
    """ Compute a ranked envelopes test.

    Args:
        signal (_type_): _description_
        fun (str, optional): _description_. Defaults to 'Fest'.
        correction (str, optional): _description_. Defaults to 'none'.
        nsim (int, optional): _description_. Defaults to 2499.
        alpha (float, optional): _description_. Defaults to 0.05.
        rmin (float, optional): _description_. Defaults to 0.0.
        rmax (float, optional): _description_. Defaults to 1.2.
        ptype (str, optional): _description_. Defaults to 'conservative'.
        ppp_sim (_type_, optional): _description_. Defaults to None.
        return_dic (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """

    # Initialize interface with R.
    spatstat = SpatstatInterface(update=False)
    spatstat.import_package("core", "geom", update=False)
    # spatstat_random = importr('spatstat.random')
    rbase = importr('base')
    package_GET = importr('GET')
    
    # STFT parameters.
    N = len(signal)
    Nfft = 2*N
    T = np.sqrt(Nfft)
    window = np.exp(-np.pi*np.arange(-Nfft//2,Nfft//2)**2/Nfft)

    # Generate empirical point-process:
    signal_aux = np.zeros((Nfft,))
    signal_aux[Nfft//4:Nfft//4+N] = np.random.randn(N)
    signal_aux[Nfft//4:Nfft//4+N]  = signal
    _,_,stft = sg.stft(signal_aux, window = window, nperseg = Nfft, noverlap=Nfft-1)
    stft = stft[:,Nfft//4:Nfft//4+N]
    pos = find_zeros_of_spectrogram(np.abs(stft)**2)

    # fig, ax = plt.subplots(1,1,figsize = (5,5))
    # ax.imshow(np.log10(abs(stft)), origin='lower')
    # ax.plot(pos[:,1],pos[:,0],'w+')
    # plt.show()

    pos = np.array(pos)/T
    u_r, v_r, b_u, b_v = compute_positions_and_bounds(pos)
    ppp_r = spatstat.geom.ppp(u_r, v_r, b_u, b_v)

    if ppp_sim is None:
        ppp_sim = generate_white_noise_zeros_pp(N,nsim)

    
    # If a transform is passed as an extra (optional) parameter, apply first the
    # expression parsing function from R.
    extra_args = kwargs.copy() # Better copying before modifying input parameters.
    if 'transform' in extra_args.keys():
        extra_args['transform'] = rbase.expression(extra_args['transform'])

    # Compute simulated envelopes:
    envelopes = spatstat.core.envelope(ppp_r, 
                                    fun=fun, 
                                    nsim=nsim, 
                                    savefuns=True,
                                    correction=correction, 
                                    simulate=ppp_sim, 
                                    verbose='FALSE',
                                    **extra_args)
    
    # Crop envelopes if required.
    envelopes = package_GET.crop_curves(envelopes, r_min=rmin, r_max=rmax)

    # Compute the actual test now.
    res = package_GET.global_envelope_test(envelopes, 
                                            alpha=alpha, 
                                            type='rank', 
                                            alternative = 'less') # <- We are doing 
                                                                  # one-sided tests.

    # Get the attributes from the results of the test.
    res_attr = rbase.attributes(res)

    # Get p-value interval: Liberal (first index) and Conservative (second index).
    numpy2ri.activate()
    pval_int = res_attr[13]
    if ptype == 'conservative':
        pval = pval_int[1]

    if ptype == 'liberal':
        pval = pval_int[0]

    # Get envelopes and r.
    r = res[0]
    obs = res[1]
    central = res[2]
    lo = res[3]
    hi = res[4]

    # fix, ax = plt.subplots(1,1)
    # ax.fill_between(r, lo, hi, color='g', alpha=.8)
    # ax.plot(r,obs,'r--')
    # plt.show()

    # Check rejection of H0:
    if pval < alpha:
        rejectH0 = True
    else:
        rejectH0 = False

    # If rejected, compute the maximum distance between the empirical statistic and the
    # envelopes (it might be useful as an approximate scale of interaction).
    r_max_dif = None
    ind_max_dif = None
    if rejectH0:
        ind_max_dif = np.argmax(lo-obs)
        r_max_dif = r[ind_max_dif]

    # Generate output dictionary:
    if return_dic:
        output_dic = {  'rejectH0':rejectH0,
                    'envelope_obs': obs,
                    'envelope_lo': lo,
                    'envelope_hi': hi,
                    'envelope_central':central,
                    'radi': r,
                    'r_max_dif': r_max_dif,
                    'ind_max_dif': ind_max_dif
                }
    
        return output_dic
    else:
        return rejectH0


def compute_scale(signal, **test_params):
    """_summary_

    Args:
        signal (_type_): _description_
        Nfft (_type_): _description_
        cs (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    
    # If the simulated ppp are present, use them, otherwise, generate the simulation and
    # save it so that it can be used later (this saves time, avoiding re-simulation).
    nsim = 2499
    N = len(signal)
    try:
        with open('ppp_simulations_N_{}_Nsim_{}.mcsim'.format(N,nsim), 'rb') as handle:
            ppp_simulation = pickle.load(handle)
    except:
        list_ppp = generate_white_noise_zeros_pp(N, nsim=nsim)
        ppp_simulation = {'list_ppp':list_ppp,
                    'N' : N,
                    'nsim' : nsim
                    }
        with open('ppp_simulations_N_{}_Nsim_{}.mcsim'.format(N,nsim), 'wb') as handle:
            pickle.dump(ppp_simulation, handle, protocol=pickle.HIGHEST_PROTOCOL)   


    output_dic = compute_rank_envelope_test(signal,
                                         return_dic=True, 
                                         ppp_sim=ppp_simulation['list_ppp'],
                                         **test_params)

    reject_H0 = output_dic['rejectH0']
  
    if not reject_H0:
        print('No detection.')
        radius_of_rejection = 0.8
    else:
        radius_of_rejection = output_dic['r_max_dif']
    
    return radius_of_rejection


def compute_global_mad_test(signal, 
                    cs=None, 
                    MC_reps = 199, 
                    alpha = 0.05, 
                    statistic='L',
                    pnorm = 2, 
                    radius=None, 
                    rmax=None,
                    one_sided=False,
                    return_values=False):
    """ Compute global MAD hypothesis tests based on Monte Carlo simulations.

    Args:
        signal (ndarray): Numpy ndarray with the signal.
        cs (ComputeStatistics, optional): This is an object of the ComputeStatistics
        class, that encapsulates the initialization of the spatstat-interface python
        package. This allows avoiding reinitialize the interface each time. 
        Defaults to None.
        MC_reps (int, optional): Repetitions of the Monte Carlo simulations. 
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

    output_dict, radius = compute_monte_carlo_sims(signal,
                                        cs,
                                        Nfft,
                                        MC_reps=MC_reps,
                                        statistic=statistic,
                                        pnorm=pnorm,
                                        radius=radius,
                                        rmax=rmax,
                                        one_sided=one_sided)
    
    for sts in statistic:
        _, _, Sm, Sexp, S0 = output_dict[sts]

        # Compute extreme values
        texp = np.max(np.abs(Sexp-S0))
        tsim = np.max(np.abs(Sm-S0), axis=1)
        tsim = np.sort(tsim, axis=0)[::-1]

        if texp > tsim[k]:
            reject_H0 = True
        else:
            reject_H0 = False

        if return_values:
            output_dict[sts] = {'reject_H0': reject_H0,
                                'Sm':Sm,
                                'Sexp':Sexp, 
                                'S0':S0,
                                'texp' : texp,
                                'tsim' : tsim,
                                'k': k,
                                'radius':radius}
        else:
            output_dict[sts] = reject_H0
    
    if len(statistic)>1:    
        return output_dict 
    else: # returns (tm, t_exp) if only one statistic was computed.
        return output_dict[statistic[0]]


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


