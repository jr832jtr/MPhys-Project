# coding=utf-8
"""
@requires: numpy
@author: Carolin Villforth
@summary: Calculates Physical Properties of astrophysical objects, such as Eddington Luminosties and Rates and Similar
"""
import numpy
import scipy.constants


def EddingtonLuminosity(bhmass):
    """
    Calculates the Eddington Luminosity in erg/s given a BH mass in Solar Masses
    @param bhmass:
    @return:
    """
    l = 1.26e38 * bhmass
    return l


def AccretionRate(bhmass, edd_ratio=1, eta=0.1):
    """
    Calculates the Accretion Rate in solar masses per year given a BH mass, eddington ratio and eta
    @param bhmass:
    @param edd_ratio:
    @param eta:
    @return:
    """
    l = EddingtonLuminosity(bhmass)
    mdot = edd_ratio * (1.8e-3 * (l/1e44)) / eta
    return mdot


def AccretionRate_L(l, eta=0.1):
    """
    Calculates the Accretion Rate in solar masses per year given a BH mass, eddington ratio and eta
    @param bhmass:
    @param edd_ratio:
    @param eta:
    @return:
    """
    mdot = (1.8e-3 * (l/1e44)) / eta
    return mdot


def thindisk(mbh_8=1, mdot_edd=1, rin_s=3, rout_s=5000, step=0.5, wlmin=1e-11, wlmax=1e-7, n_wl=1000, wlspacing='log'):
    """

    @param mbh_8:
    @param mdot_edd:
    @param rin_s:
    @param rout_s:
    @param step:
    @param wlmin:
    @param wlmax:
    @param n_wl:
    @return:
    """
    print("THE UNITS ARE NOT RIGHT")
    r_array = numpy.arange(rin_s, rout_s, step)
    n_r = len(r_array)
    if wlspacing == 'log':
        wl_array = numpy.logspace(numpy.log10(wlmin), numpy.log10(wlmax), n_wl)
    elif wlspacing == 'lin':
        wl_array = numpy.linspace(wlmin, wlmax, n_wl)
    else:
        raise MyExcpetions.Hell("Not allowed")
    flux_array = numpy.zeros((n_r, n_wl))
    t_array = 6.3e5 * (mdot_edd**0.25) * (mbh_8**(-0.25)) * (r_array**(-0.75))
    for i, t, r in zip(range(n_r), t_array, r_array):
        flux_array[i] = planck(t, wl_array) * 2 * numpy.pi * r * step
    flux = numpy.sum(flux_array, 0)
    return {'flux': flux, 'flux_array': flux_array, 't': t_array, 'wl': wl_array}


def planck(t, wl):
    print("THE UNITS ARE NOT RIGHT")
    exp_value = (scipy.constants.h * scipy.constants.c) / (wl * t * scipy.constants.k)
    b = ((2 * scipy.constants.h * scipy.constants.c**2) / (wl**5)) / (numpy.exp(exp_value) - 1)
    return b


def gamma_from_beta(beta):
    """
    Lorentzfactor given beta
    :param beta:
    :return:
    """
    return 1./numpy.sqrt(1 - beta*2)


def beta_from_gamma(gamma):
    """
    beta given lorentz factor
    :param gamma:
    :return:
    """
    return numpy.sqrt(1-(1./gamma)**2)


def beta_transverse(beta, theta):
    """
    aparant transverse beta given input beta and angle to line of sigh
    :param beta:
    :param theta:
    :return:
    """
    return (beta * numpy.sin(theta))/(1-beta*numpy.cos(theta))
