import numpy as np
from scipy.fft import fft, ifft
from numpy import pi as pi
import types
from scipy.spatial import KDTree

def is_minimum(F):
    F = F.reshape((F.shape[0]*F.shape[1],))
    ind_min = np.argmin(F)
    if ind_min == 4:
        return True
    else:
        return False


def is_maximum(F):
    F = F.reshape((F.shape[0]*F.shape[1],))
    ind_min = np.argmax(F)
    if ind_min == 4:
        return True
    else:
        return False


def zeros_finder(F):
    N = F.shape[0]
    M = F.shape[1]
    S = np.ones((N+2, M+2))*float("inf")
    S[1:N+1,1:M+1] = np.abs(F)
    zeros = np.zeros((1,2))

    skip_next_flag = False
    for i in range(1,F.shape[0]):
        for j in range(1, F.shape[1]):
            if skip_next_flag:
                # print("Me fui!")
                skip_next_flag = False
            else:
                aux = S[i-1:i+2,j-1:j+2]
                if is_minimum(aux):
                    new_zero = np.array([i,j]).reshape((1,2))
                    zeros = np.append(zeros,new_zero, axis=0)
                    skip_next_flag = True

    zeros = zeros[1::]
    return np.array(zeros)


def max_finder(F):
    N = F.shape[0]
    M = F.shape[1]
    S = np.zeros((N+2, M+2))
    S[1:N+1,1:M+1] = np.abs(F)
    maxima = np.zeros((1,2))

    skip_next_flag = False
    for i in range(1,F.shape[0]):
        for j in range(1, F.shape[1]):
            aux = S[i-1:i+2,j-1:j+2]
            if is_maximum(aux):
                new_zero = np.array([i,j]).reshape((1,2))
                maxima = np.append(maxima,new_zero, axis=0)

    maxima = maxima[1::]
    return np.array(maxima)


def my_fft(x,K=None):
    if K == None:
        K = len(x)
    return fft(x,n=K)/len(x)


def my_ifft(x):
    return ifft(x)


def my_window_temp(t, sigma, window='gaussian'):
    """
    :param t: domain of the window function
    :param sigma: extra parameter of the function, normally controlling the width.
    :param window: name of the window, or a function with firm fun((t,sigma)).
    :return: window function evaluated at t.
    """


    if isinstance(window, types.FunctionType):
        return window((t, sigma))
    else:
        # sigma = 1 / (2 * (sigma ** 2))
        windows = dict()
        windows['gaussian'] = lambda p: np.exp(-np.power(p[0], 2)*p[1])
        windows['gaussian_p'] = lambda p: -2*p[0]*p[1]*np.exp(-np.power(p[0], 2)*p[1])
        windows['gaussian_f'] = lambda p: np.sqrt(pi/p[1])*np.exp(-np.power(p[0]*pi,2)/p[1])
        windows['gaussian_fp'] = lambda p:  (2j*pi*p[0])*np.sqrt(pi/p[1])*np.exp(-np.power(p[0]*pi,2)/p[1])
        return windows[window]((t, sigma))


# STFT
def my_stft_temp(x, sigma, K=None, fmax=0.5, window='gaussian'):
    N = len(x)
    Npad = 0
    signal_pad = np.zeros(N+2*Npad)
    signal_pad[Npad:Npad+N] = x

    if K == None:
        K = N
    t = np.arange(0, N+2*Npad)
    f = np.arange(0, fmax, 1/K)
    F = np.zeros((len(f), N+2*Npad), dtype='complex')
    for q, i in enumerate(t):
        g = my_window_temp(t-i, sigma, window)
        F[:, q] = (my_fft(signal_pad*g,K)[0:len(f)])*np.exp(2j*pi*i*f)

    return F[:,Npad:Npad+N]


 # SST Frequency Operator
def sst_freq_op(x,sigma,K=None,fmax=0.5):
    eps = 0 #1E-10
    N = len(x)
    if K == None:
        K = N
    F = my_stft_temp(x, sigma, K=K) + eps
    Fprime = my_stft_temp(x, sigma, K=K, window='gaussian_p')
    a = np.arange(0, fmax, 1/K).reshape((-1,1))
    f = np.tile(a, (1, N))
    fhat = f - np.imag(Fprime/(F*2*pi))
    return fhat, F


def sst_temp_op(x,sigma,K=None,fmax=0.5):
    """_summary_

    Args:
        x (_type_): _description_
        sigma (_type_): _description_
        K (_type_, optional): _description_. Defaults to None.
        fmax (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    eps = 1E-6
    N = len(x)
    t = np.arange(0,N)
    if K == None:
        K = N
    F = my_stft_temp(x, sigma, K=K) + eps
    tg = lambda p: p[0]*my_window_temp(p[0], p[1], window='gaussian')
    Ftg = my_stft_temp(x, sigma, K=K, window=tg)
    tau = np.tile(t, (F.shape[0], 1))
    that = tau + np.real(Ftg/F)
    return that, F


def my_sst(x,T,K=None,fmax=0.5):
    N = len(x)
    if K == None:
        K = N
    fhat, F = sst_freq_op(x, T, K, fmax)
    ssF = np.zeros(F.shape, dtype='complex')
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            f_ind = int(np.round(fhat[i, j]*K))
            if 0 <= f_ind < F.shape[0]:
                ssF[f_ind, j] += F[i, j]
    return ssF, F


def my_rm(x,T,K=None,fmax=0.5):
    N = len(x)
    if K == None:
        K = N
    fhat, F = sst_freq_op(x, T, K, fmax)
    that, F = sst_temp_op(x, T, K, fmax)
    S = np.abs(F)**2
    S = S/np.sum(S)*np.sum(np.abs(x)**2)
    R = np.zeros(F.shape)
    reassignment_pos = np.zeros((F.size,2))
    k = 0

    Ex = np.mean(np.abs(x)**2)
    Threshold = 1.0e-4*Ex
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            if S[i,j] > Threshold:
                f_ind = int(np.round(fhat[i, j]*K))
                t_ind = int(np.round(that[i, j]))
                # print(t_ind)
                reassignment_pos[k] = [f_ind,t_ind]
                

                if 0 <= f_ind < F.shape[0]:
                    if 0 <= t_ind < F.shape[1]:
                        R[f_ind, t_ind] += S[i, j]
            else:
                reassignment_pos[k] = [i,j]

            k += 1
    return R, F, fhat, that, reassignment_pos


def compute_contours(x, T=None, Nfft=None, fmax=0.5):
    N = len(x)
    if T is None:
        T = pi/N

    if Nfft is None:
        Nfft = N

    t = np.arange(N)
    _, F, fhat, that, reassignment_pos = my_rm(x,T,Nfft,fmax)
    ceros = zeros_finder(F)
    ceros = ceros[np.where(ceros[:,0]<F.shape[0]-2)] #0<ceros[:,1]<F.shape[1]-1
    ceros = ceros[np.where(ceros[:,0]>0)]
    ceros = ceros[np.where(ceros[:,1]>0)]
    ceros = ceros[np.where(ceros[:,1]<F.shape[1]-1)]

    k = np.arange(0,fmax,1/Nfft)
    k.resize((k.shape[0],1))
    tcomp = that - np.tile(t,(that.shape[0], 1))
    tcomp[abs(tcomp)>N] = N
    fcomp = fhat - np.tile(k,(1, fhat.shape[1]))
    fcomp = fcomp*Nfft
    fcomp[abs(fcomp)>Nfft] = Nfft

    v = np.mod(np.arctan2(fcomp,tcomp),pi)
    vx = np.cos(v)
    vy = np.sin(v)
    indicator = np.zeros(v.shape)
    indicator2 = np.zeros(v.shape)

    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            indicator[i,j] = np.dot([tcomp[i,j],fcomp[i,j]],[vx[i,j], vy[i,j]])

    indicator[indicator>0] = 1
    indicator[indicator<0] = -1
    aux = np.concatenate((indicator, indicator[:,-1].reshape((indicator.shape[0],1))), axis = 1)
    indicator1 = np.diff(aux, axis = 1)
    indicator1[indicator1!=0] = 1
    aux = np.concatenate((indicator, indicator[-1,:].reshape((1,indicator.shape[1]))), axis = 0)
    indicator2 = np.diff(aux, axis = 0)
    indicator2[indicator2!=0] = 1

    indicator = indicator1 + indicator2
    indicator[indicator!=0] = 1

    for cero in ceros:
        cero = cero.astype(int)
        indicator[cero[0]-2:cero[0]+3:, cero[1]-2:cero[1]+3:] = 0

    return indicator,F, reassignment_pos


def get_contours_and_basins(indicator, reassignment_pos):
    ind_points_contours = np.arange(np.sum(indicator), dtype=int)
    points_contours = np.where(indicator!=0)
    points_contours = np.array(points_contours).T
    contour_assign = np.zeros(ind_points_contours.shape, dtype=int)
    contours = list()
    basins = list()
    kdpos = KDTree(points_contours)

    
    # Find the contours as paths in the indicator matrix.
    k = 0
    while len(ind_points_contours) > 0:
        flag = True
        points = [ind_points_contours[0],]
        while flag:
            results = kdpos.query_ball_point(points_contours[points], np.sqrt(2))
            new_points = list()
            for i in results:
                [new_points.append(j) for j in i]
            
            new_points = list(dict.fromkeys(new_points))
            if new_points == points:
                flag = False
            else:
                points = new_points

        contours.append(points_contours[points])
        contour_assign[points] = k
        k += 1
        for p in points:
            ind_points_contours = np.delete(ind_points_contours, np.where(ind_points_contours==p), axis=0)
    
    # Compute de basins
    idx_grid = [[i,j] for i in range(indicator.shape[0]) for j in range(indicator.shape[1])]
    # idx_contour = kdpos.query_ball_point(reassignment_pos, T)
    _, idx_contour = kdpos.query(reassignment_pos, k=1)
    idx_basin = contour_assign[idx_contour]
    idx_grid = np.array(idx_grid)
    for contour in range(len(contours)):
        basins.append(idx_grid[np.where(idx_basin == contour)])
    
    return contours, basins
  

def contours_filtering(signal, q = None, Nbasins = None, dict_output=False):
    
    if q is None and Nbasins is None:
        q = 0.9
    else:
        if q is not None and Nbasins is not None:
            q = 0.9
            Nbasins = None
            print('Please choose only one criterion for basins selection.')

    indicator, F, reassignment_pos = compute_contours(signal)
    contours, basins = get_contours_and_basins(indicator, reassignment_pos)

    S = np.abs(F)**2
    energy_basins = np.zeros((len(basins),)) 
    
    # for i,basin in enumerate(basins):
    #     energy_basins[i] = np.sum(S[basin[:,0],basin[:,1]])

    for i,contour in enumerate(contours):
        energy_basins[i] = np.sum(S[contour[:,0],contour[:,1]])


    if q is not None:
        ind_basins = np.where(energy_basins > np.quantile(energy_basins,q))
        basins = [basins[i] for i in ind_basins[0]]
        contours = [contours[i] for i in ind_basins[0]]

    if Nbasins is not None:
        if Nbasins == 'all':
            Nbasins = len(contours)
        
        order_energy_basins = np.argsort(energy_basins)[-1:0:-1]
        basins = [basins[i] for i in order_energy_basins[0:Nbasins]]
        contours = [contours[i] for i in order_energy_basins[0:Nbasins]]    

    mask = np.zeros(indicator.shape)
    for basin in basins:
        mask[basin[:,0],basin[:,1]] = 1 #np.random.randint(low = 1, high= 1500)

    F_hat = F*mask
    x_hat = 2*np.real(np.sum(F_hat, axis = 0))   

    if dict_output:
        return {'x_hat': x_hat, 'mask': mask, 'contours': contours,'basins': basins}
    else:
        return x_hat


def ridges_sst_filtering(signal,):
    N = len(signal)
    if T is None:
        T = pi/N

    if Nfft is None:
        Nfft = N
        
    ssF,F = my_sst(signal, T, Nfft, fmax=0.5)


# if __name__ == '__main__':
#     N = 512
#     K = N
#     fmax = 0.5
#     t = np.arange(N)
#     finst = 0.05*np.ones(t.shape) + 0.2*t/N
#     finst2 = 0.2*np.ones(t.shape) + 0.2*t/N 
#     chirp = (np.cos(2*pi*np.cumsum(finst))+np.cos(2*pi*np.cumsum(finst2)))*sg.windows.tukey(N,0.5)
#     x = chirp
#     x, n = add_snr(x,30)
#     indicator, F = compute_contours(x)
#     x_hat, mask, contours, basins = contours_filtering(x, Nbasins = 3) 
    
#     print(10*np.log10((np.sum(chirp**2))/(np.sum((chirp-x_hat)**2))))
    
#     fig, axs = plt.subplots(2,2)
#     axs[0,0].imshow(np.abs(F), origin = 'lower')
#     axs[0,0].imshow(-indicator, alpha = 0.5, origin = 'lower')
#     aux = np.zeros(indicator.shape)
#     for basin in basins:
#         aux[basin[:,0],basin[:,1]] = np.random.randint(low = 1, high= 1500)
    
#     axs[0,1].imshow(aux, origin = 'lower')
#     axs[1,0].imshow(mask, origin = 'lower')
#     axs[1,1].plot(chirp)
#     axs[1,1].plot(x_hat,'r--')
#     plt.show()
    