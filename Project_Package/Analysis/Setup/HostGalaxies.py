__author__ = 'cvillforth'
import numpy


def msigma(mgal, a=8.192, b=1.13, x0=11, e=0.3):
    """
    Placeholder function until I can draw properly from msigma
    @param mgal:
    @param a:
    @param b:
    @param x0:
    @param e:
    @return:
    """
    if type(mgal) in [float, int]:
        mgal = numpy.array([mgal])
    _n = numpy.shape(mgal)[0]
    mbh = a + b * (numpy.log10(mgal) - x0) + numpy.random.standard_normal(_n) * e
    return mbh


def mgalsfr_brokenpl(mgal, a_low=0.94, a_high=0.14, b=1.11, scatter=0.3):
    """
    Draws a randomized sample of SFRs given an input galaxy mass in solar masses given a broken powerlaw
    prescription for the main sequence.
    The default values are from the lowest mass bin (0.5-1) in Whitaker et al. 2014, ApJ, 795, 104
    @param mgal:
    @param a_low:
    @param a_high:
    @param b:
    @param scatter:
    @return:
    """
    if type(mgal) in [float, int]:
        mgal = numpy.array([mgal])
    _n = numpy.shape(mgal)[0]
    a = numpy.zeros_like(mgal)
    for i, m in enumerate(mgal):
        if numpy.log10(m) < 10.2:
            a[i] = a_low
        else:
            a[i] = a_high
    sfr = numpy.power(10, (a * (numpy.log10(mgal)-10.2) + b + numpy.random.standard_normal(_n) * scatter))
    mean_sfr = numpy.power(10, (a * (numpy.log10(mgal)-10.2) + b))
    return sfr, mean_sfr


def mgalsfr_poly(mgal, a=-27.40, b=5.02, c=-0.22, scatter=0.3):
    """
    Draws a randomized sample of SFRs given an input galaxy mass in solar masses given a polynimial
    prescription for the main sequence.
    The default values are from the lowest mass bin (0.5-1) in Whitaker et al. 2014, ApJ, 795, 104
    @param mgal:
    @param a:
    @param b:
    @param c:
    @param scatter:
    @return:
    """
    if type(mgal) in [float, int]:
        mgal = numpy.array([mgal])
    _n = numpy.shape(mgal)[0]
    sfr = numpy.power(10, (a + b * numpy.log10(mgal) + c * numpy.log10(mgal)**2
                           + numpy.random.standard_normal(_n) * scatter))
    mean_sfr = numpy.power(10, (a + b * numpy.log10(mgal) + c * numpy.log10(mgal)**2))
    return sfr, mean_sfr


