import numpy as np
import multiprocessing
import itertools
from neurotools import times_to_bin_multi


'''
Victor-Purpura spike time distance metric
Victor and Purpura, 1996, Journal of Neurophysiology
'''
def vp(st_one, st_two, cost):
    '''
    Calculates the "spike time" distance (Victor & Purpura, 1996) for a single
    cost.

    tli     - vector of spike times for first spike train
    tlj     - vector of spike times for second spike train
    cost    - cost per unit time to move a spike

    Translated to Python by Achilleas Koutsou from Matlab code by Daniel Reich.
    '''
    len_one = len(st_one)
    len_two = len(st_two)
    if cost == 0:
        dist = np.abs(len_one-len_two)
        return dist
    elif cost == float('inf'):
        dist = len_one+len_two
        return dist
    scr = np.zeros((len_one+1, len_two+1))
    scr[:,0] = np.arange(len_one+1)
    scr[0,:] = np.arange(len_two+1)
    if len_one and len_two:
        for i in range(1, len_one+1):
            for j in range(1, len_two+1):
                scr[i,j]=np.min((scr[i-1,j]+1,
                                scr[i,j-1]+1,
                                scr[i-1,j-1]+cost*np.abs(st_one[i-1]-st_two[j-1]))
                               )
    return scr[-1,-1]


def vp_pwise_mp(spiketrains, cost):
    '''
    Calculates the average pairwise distance between a set of spike trains.
    Uses Python's multiprocessing.Pool() to run each pairwise distance
    calculation in parallel.
    '''
    count = len(spiketrains)
    distances = []
    idx_all = range(count - 1)
    pool = multiprocessing.Pool()
    distances_nested = pool.map(_all_dist_to_end,
                                zip(idx_all, itertools.repeat(spiketrains),
                                    itertools.repeat(cost)))
    distances = []
    pool.close()
    pool.join()
    for dn in distances_nested:
        distances.extend(dn)
    return np.mean(distances)


def vp_pwise(all_spikes, cost):
    '''
    Calculates the average pairwise distance between a set of spike trains.
    '''
    count = len(all_spikes)
    distances = []
    for i in range(count - 1):
        for j in range(i + 1, count):
            dist = vp(all_spikes[i], all_spikes[j], cost)
            distances.append(dist)
    return np.mean(distances)


def _all_dist_to_end(args):
    '''
    Helper function for parallel pairwise distance calculations.
    '''
    idx = args[0]
    spiketrains = args[1]
    cost = args[2]
    num_spiketrains = len(spiketrains)
    distances = []
    for jdx in range(idx + 1, num_spiketrains):
        dist = vp(spiketrains[idx], spiketrains[jdx], cost)
        distances.append(dist)
    return distances


def interval_vp_st(inputspikes, outputspikes, cost, dt=0.0001, mp=True):
    '''
    Calculates the mean pairwise spike time distance in intervals defined
    by a separate spike train. This function is used to calculate the distance
    between *input* spike trains based on the interspike intervals of the
    *output* spike train. The result is therefore the distance between the
    input spikes that caused each response.

    inputspikes     - A set of spike trains whose pairwise distance will be
                    calculated

    outputspikes    - A single spike train to be used to calculate the
                    intervals

    cost            - The cost of moving a spike

    dt              - The simulation time step (default: 0.0001)

    mp              - Set to True to use the multiprocessing implementation
                    of the pairwise calculation function or False to use the
                    single process version (default: True)

    '''
    dt = float(dt)
    vpdists = []
    for prv, nxt in zip(outputspikes[:-1], outputspikes[1:]):
        interval_inputs = []
        for insp in inputspikes:
            interval_inputs.append(insp[(prv < insp) & (insp < nxt+dt)])
        if mp:
            vpd = vp_pwise_mp(interval_inputs, cost)
        else:
            vpd = vp_pwise(interval_inputs, cost)
        vpdists.append(vpd)
    return vpdists



'''
Kreuz SPIKE-distance
Kreuz, Chicharro, Greschner, Andrzejak, 2011, Journal of Neuroscience Methods.


Comment by Thomas Kreuz:

This Python code (including all further comments) was written by Jeremy Fix (see http://jeremy.fix.free.fr/),
based on Matlab code written by Thomas Kreuz.

The SPIKE-distance is described in this paper:

Kreuz T, Chicharro D, Houghton C, Andrzejak RG, Mormann F:
Monitoring spike train synchrony.
J Neurophysiol 109, 1457-1472 (2013).

The Matlab codes as well as more information can be found at http://www.fi.isc.cnr.it/users/thomas.kreuz/sourcecode.html.

'''
def find_corner_spikes(t, train, ibegin, ti, te):
    '''
    Return the times (t1,t2) of the spikes in train[ibegin:]
    such that t1 < t and t2 >= t
    '''
    if(ibegin == 0):
        tprev = ti
    else:
        tprev = train[ibegin-1]
    for idts, ts in enumerate(train[ibegin:]):
        if(ts >= t):
            return np.array([tprev, ts]), idts+ibegin
        tprev = ts
    return np.array([train[-1],te]), idts+ibegin


def bivariate_spike_distance(t1, t2, ti, te, N):
    '''Computes the bivariate SPIKE distance of Kreuz et al. (2012)
       t1 and t2 are 1D arrays with the spiking times of two neurones
       It returns the array of the values of the distance
       between time ti and te with N samples.
       The arrays t1, t2 and values ti, te are unit less '''
    t = np.linspace(ti+(te-ti)/N, te, N)
    d = np.zeros(t.shape)

    t1 = np.insert(t1, 0, ti)
    t1 = np.append(t1, te)
    t2 = np.insert(t2, 0, ti)
    t2 = np.append(t2, te)

    # We compute the corner spikes for all the time instants we consider
    # corner_spikes is a 4 column matrix [t, tp1, tf1, tp2, tf2]
    corner_spikes = np.zeros((N,5))

    ibegin_t1 = 0
    ibegin_t2 = 0
    corner_spikes[:,0] = t
    for itc, tc in enumerate(t):
       corner_spikes[itc,1:3], ibegin_t1 = find_corner_spikes(tc, t1,
                                                              ibegin_t1,
                                                              ti, te)
       corner_spikes[itc,3:5], ibegin_t2 = find_corner_spikes(tc, t2,
                                                              ibegin_t2,
                                                              ti, te)

    #print corner_spikes
    xisi = np.zeros((N,2))
    xisi[:,0] = corner_spikes[:,2] - corner_spikes[:,1]
    xisi[:,1] = corner_spikes[:,4] - corner_spikes[:,3]
    norm_xisi = np.sum(xisi,axis=1)**2.0

    # We now compute the smallest distance between the spikes in t2
    # and the corner spikes of t1
    # with np.tile(t2,(N,1)) we build a matrix :
    # np.tile(t2,(N,1)) = [t2] - np.tile(reshape(corner_spikes,(N,1)), t2.size) = [                        ]
    #                     [t2]                                                    [  corner  corner  corner]
    #                     [t2]                                                    [                        ]
    dp1 = np.min(np.fabs(np.tile(t2,(N,1))
                         - np.tile(np.reshape(corner_spikes[:,1],(N,1)),
                                   t2.size)),
                 axis=1)
    df1 = np.min(np.fabs(np.tile(t2,(N,1))
                         - np.tile(np.reshape(corner_spikes[:,2],(N,1)),
                                   t2.size)),
                 axis=1)
    # And the smallest distance between the spikes in t1 and the corner spikes of t2
    dp2 = np.min(np.fabs(np.tile(t1,(N,1))
                         - np.tile(np.reshape(corner_spikes[:,3],
                                              (N,1)),t1.size)),
                 axis=1)
    df2 = np.min(np.fabs(np.tile(t1,(N,1))
                         - np.tile(np.reshape(corner_spikes[:,4],(N,1)),
                                   t1.size)),
                 axis=1)

    xp1 = t - corner_spikes[:,1]
    xf1 = corner_spikes[:,2] - t
    xp2 = t - corner_spikes[:,3]
    xf2 = corner_spikes[:,4] - t

    S1 = (dp1 * xf1 + df1 * xp1)/xisi[:,0]
    S2 = (dp2 * xf2 + df2 * xp2)/xisi[:,1]

    d = (S1 * xisi[:,1] + S2 * xisi[:,0]) / (norm_xisi/2.0)

    return t,d


def multivariate_spike_distance(spike_trains, ti, te, N):
    '''
    t is an array of spike time arrays
    ti the initial time of the recordings
    te the end time of the recordings
    N the number of samples used to compute the distance
    spike_trains is a list of arrays of shape (N, T) with N spike trains
    The multivariate distance is the instantaneous average over all the
    pairwise distances
    '''
    d = np.zeros((N,))
    n_trains = len(spike_trains)
    t = 0
    for i, t1 in enumerate(spike_trains[:-1]):
        for t2 in spike_trains[i+1:]:
            tij, dij = bivariate_spike_distance(t1, t2, ti, te, N)
            if(i == 0):
                t = tij # The times are only dependent on ti, te, and N
            d = d + dij
    d = d / float(n_trains * (n_trains-1) /2)
    return t,d


def interval_kr(inputspikes, outputspikes, dt=0.0001):
    '''
    Calculates the mean pairwise SPIKE-distance in intervals defined
    by a separate spike train. This function is used to calculate the distance
    between *input* spike trains based on the interspike intervals of the
    *output* spike train. The result is therefore the distance between the
    input spikes that caused each response.

    inputspikes     - A set of spike trains whose pairwise distance will be
                    calculated

    outputspikes    - A single spike train to be used to calculate the
                    intervals

    dt              - The simulation time step (default: 0.0001)

    '''
    dt = float(dt)
    krdists = []
    for prv, nxt in zip(outputspikes[:-1], outputspikes[1:]):
        krd = multivariate_spike_distance(inputspikes, prv, nxt+dt, 1)
        krdists.append(krd[1])
    return krdists



'''
Correlation coefficient between binned spike trains.
'''
def corrcoef_spiketrains(spikes, b=0.001, duration=None):
    '''
    Calculates the mean correlation coefficient between a set of spike trains
    after being binned with bin width `b`.
    '''
    bintimes = times_to_bin_multi(spikes, b, duration)
    correlations = np.corrcoef(bintimes)
    return correlations


def interval_corr(inputspikes, outputspikes, b=0.001, duration=None):
    '''
    Calculates the mean pairwise correlation coefficient in intervals defined
    by a separate spike train. This function is used to calculate the
    correlation between *input* spike trains based on the interspike intervals
    of the *output* spike train. The result is therefore the distance between
    the input spikes that caused each response.

    inputspikes     - A set of spike trains whose pairwise distance will be
                    calculated

    outputspikes    - A single spike train to be used to calculate the
                    intervals

    dt              - The simulation time step (default: 0.0001)

    '''
    b = float(b)
    corrs = []
    for prv, nxt in zip(outputspikes[:-1], outputspikes[1:]):
        interval_inputs = []
        for insp in inputspikes:
            interval_spikes = insp[(prv < insp) & (insp <= nxt)]-prv
            if len(interval_spikes):
                interval_inputs.append(interval_spikes)
        corrs_i = np.mean(corrcoef_spiketrains(interval_inputs, b, duration))
        corrs.append(corrs_i)
    return corrs

