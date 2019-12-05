__author__ = 'cvillforth'
"""
A container for useful and much-loved functions such as powerlaws.
"""
import numpy
import cmath


def powerlaw(x, a, alpha):
    """
    A powerlaw.
    @param x: x
    @param a: normalization
    @param alpha: alpha
    @return: the powerlaw
    """
    return a * numpy.power(x, alpha)


def gaussian(x, mu=0, sigma=1, scale=1, cuttail_low=False, cuttail_high=False, peakatone=False):
    """
    Evaluates a gaussian over x
    @param x:
    @param mu:
    @param sigma:
    @param scale:
    @return:
    """
    out = (1. * scale / (sigma * numpy.sqrt(2 * numpy.pi))) * numpy.exp(-((x - mu)** 2)/ (2 * sigma**2))
    out = out.ravel()
    if cuttail_low:
        out[(x < (mu - cuttail_low))] = 0
    if cuttail_high:
        out[(x > (mu + cuttail_high))] = 0
    if peakatone:
        out *= sigma * numpy.sqrt(2*numpy.pi)
    return out


def lognormal(x, mu=0, sigma=1, scale=1):
    out = scale * numpy.exp(-((x - mu)** 2)/ (2 * sigma**2)) / (x* sigma * numpy.sqrt(2 * numpy.pi))
    return out


def dampedharmonic(x, amp1, amp2, gamma, omega, delta=0):
    if gamma >= omega:
        out = amp1 * numpy.exp(x*(-gamma + numpy.sqrt(gamma**2 - omega**2)))\
              + amp2 * numpy.exp(x*(-gamma - numpy.sqrt(gamma**2 - omega**2)))
    else:
        out = amp1 * numpy.exp(-gamma*x) * numpy.cos(numpy.sqrt(x*(omega**2) - gamma**2) + delta)
    return out


def dampedharmonic_complex(x, amp1, amp2, gamma, omega):
    out = []
    for i, _x in enumerate(x):
        out.append(amp1 * cmath.exp(_x*(-gamma + cmath.sqrt(gamma**2 - omega**2)))\
            + amp2 * cmath.exp(_x*(-gamma - cmath.sqrt(gamma**2 - omega**2))))
    return out


def dampedharmonic_complex_func(amp1, amp2, gamma, omega):
    def outfunc(x):
        out = []
        for i, _x in enumerate(x):
            out.append(amp1 * cmath.exp(_x*(-gamma + cmath.sqrt(gamma**2 - omega**2)))\
                + amp2 * cmath.exp(_x*(-gamma - cmath.sqrt(gamma**2 - omega**2))))
        return out
    return outfunc

def dampedharmonic_func(amp1, amp2, gamma, omega, delta=0):
    if gamma >= omega:
        out = lambda x: amp1 * numpy.exp(x*(-gamma + numpy.sqrt(gamma**2 - omega**2)))\
              + amp2 * numpy.exp(x*(-gamma - numpy.sqrt(gamma**2 - omega**2)))
    else:
        out = lambda x: amp1 * numpy.exp(-gamma*x) * numpy.cos(numpy.sqrt(x*(omega**2) - gamma**2) + delta)
    return out
