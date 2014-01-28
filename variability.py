"""
Variability metrics commonly used for measuring neural spiking variability
(CV, CV2, LV, IR).

TODO: List functions and classes and finish docstrings.
"""
import numpy as np


def CV(spiketrain):
    '''
    Calculates the coefficient of variation for a spike train or any supplied
    array of values

    Parameters
    ----------
    spiketrain : numpy array (or arraylike)
        A spike train characterised by an array of spike times

    Returns
    -------
    CV : float
        Coefficient of variation for the supplied values

    '''

    isi = np.diff(spiketrain)
    if len(isi) == 0:
        return 0.0

    avg_isi = np.mean(isi)
    std_isi = np.std(isi)
    return std_isi/avg_isi


def CV2(spiketrain):
    '''
    Calculates the localised coefficient of variation for a spike train or
    any supplied array of values

    Parameters
    ----------
    spiketrain : numpy array (or arraylike)
        A spike train characterised by an array of spike times

    Returns
    -------
    CV2 : float
        Localised coefficient of variation for the supplied values

    '''

    isi = np.diff(spiketrain)
    N = len(isi)
    if (N == 0):
        return 0

    mi_total = 0
    for i in range(N-1):
        mi_total = mi_total + abs(isi[i]-isi[i+1])/(isi[i]+isi[i+1])

    return mi_total*2/N


def IR(spiketrain):
    '''
    Calculates the IR measure for a spike train or any supplied array of values

    Parameters
    ----------
    spiketrain : numpy array (or arraylike)
        A spike train characterised by an array of spike times

    Returns
    -------
    IR : float
        IR measure for the supplied values

    '''



    isi = np.diff(spiketrain)
    N = len(isi)
    if (N == 0):
        return 0

    mi_total = 0
    for i in range(N-1):
        mi_total = mi_total + abs(np.log(isi[i]/isi[i+1]))

    return mi_total*1/(N*np.log(4))


def LV(spiketrain):
    '''
    Calculates the measure of local variation for a spike train or any
    supplied array of values

    Parameters
    ----------
    spiketrain : numpy array (or arraylike)
        A spike train characterised by an array of spike times

    Returns
    -------
    LV : float
        Measure of local variation for the supplied values

    '''


    isi = np.diff(spiketrain)
    N = len(isi)
    if (N == 0):
        return 0

    mi_total = 0
    for i in range(N-1):
        mi_total = mi_total + ((isi[i] - isi[i+1])/(isi[i] + isi[i+1]))**2

    return mi_total*3/N


def SI(spiketrain):
    '''
    Calculates the SI measure for a spike train or any supplied array of values

    Parameters
    ----------
    spiketrain : numpy array (or arraylike)
        A spike train characterised by an array of spike times

    Returns
    -------
    SI : float
        SI measure for the supplied values

    '''


    isi = np.diff(spiketrain)
    N = len(isi)
    if (N == 0):
        return 0

    mi_sum = 0
    for i in range(N-1):
        mi_sum = mi_sum + np.log(4*isi[i]*isi[i+1]/((isi[i]+isi[i+1])**2))

    return -1./(2*N*(1-np.log(2)))*mi_sum

