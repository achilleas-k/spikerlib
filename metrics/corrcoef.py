'''
Correlation coefficient between binned spike trains.
'''
import numpy as np
from ..tools import times_to_bin_multi


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

