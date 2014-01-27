import numpy as np
import multiprocessing
import itertools
from neurotools import times_to_bin_multi


def vp(st_one, st_two, cost):
    '''
    Calculates the "spike time" distance (Victor & Purpura, 1996) for a single
    cost.

    tli: vector of spike times for first spike train
    tlj: vector of spike times for second spike train
    cost: cost per unit time to move a spike

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
    count = len(all_spikes)
    distances = []
    for i in range(count - 1):
        for j in range(i + 1, count):
            dist = vp(all_spikes[i], all_spikes[j], cost)
            distances.append(dist)
    return np.mean(distances)


def _all_dist_to_end(args):
    idx = args[0]
    spiketrains = args[1]
    cost = args[2]
    num_spiketrains = len(spiketrains)
    distances = []
    for jdx in range(idx + 1, num_spiketrains):
        dist = vp(spiketrains[idx], spiketrains[jdx], cost)
        distances.append(dist)
    return distances


def interval_vp_st(inputspikes, outputspikes, cost, dt=0.0001):
    dt = float(dt)
    vpdists = []
    for prv, nxt in zip(outputspikes[:-1], outputspikes[1:]):
        interval_inputs = []
        for insp in inputspikes:
            interval_inputs.append(insp[(prv < insp) & (insp < nxt+dt)])
        vpd = vp_pairwise_mean(interval_inputs, cost)
        vpdists.append(vpd)
    return vpdists


def interval_kr(inputspikes, outputspikes, dt=0.0001):
    dt = float(dt)
    krdists = []
    for prv, nxt in zip(outputspikes[:-1], outputspikes[1:]):
        krd = multivariate_spike_distance(inputspikes, prv, nxt+dt, 1)
        krdists.append(krd[1])
    return krdists


def interval_corr(inputspikes, outputspikes, b=0.001, duration=None):
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


def corrcoef_spiketrains(spikes, b=0.001, duration=None):
    bintimes = times_to_bin_multi(spikes, b, duration)
    correlations = np.corrcoef(bintimes)
    return correlations


