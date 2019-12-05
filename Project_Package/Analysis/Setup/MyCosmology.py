__author__ = 'cvillforth'
from math import *
import numpy
import scipy.integrate
from scipy.stats import rv_continuous

astrodict = {'l_solar': 3.839 * 10 ** 33, 'Mpc_in_cm': 3.08568025 * 10 ** 24}


def cosmocal(z, h0=70, wm=0.3, wv=0.7, verbose=False, dictreturn=True):
    """
    Calculates cosmological parameters
    @type z: number
    @param z: redshift
    @type h0: number
    @param h0: Hubble Constant (Defaults to 70)
    @type wm: number
    @param wm: Matter Density (Defaults to 0.3)
    @type wv: number
    @param wv: Lambda Density (Defaults to 0.7)
    @type verbose: True/False
    @param verbose: True will print out information, False will return this information (Default False)
    @type dictreturn: True/False
    @param dictreturn: True will return dictionary, False will return list (Default True)
    @rtype: list/Dictionary/PrintOut Depending on Input
    @return: Years since the Big Bang, Comoving Radial Distance, Scale, Distance Module
    @note: This is Ned Wrights Cosmology Calculator
    (Input and output mildly modified) U{http://www.astro.ucla.edu/~wright/intro.html}
    """
    c = 299792.458  # velocity of light in km/sec
    tyr = 977.8  # coefficent for converting 1/H into Gyr
    h = h0 / 100.
    wr = 4.165E-5 / (h * h)  # includes 3 massless neutrino species, T0 = 2.72528
    wk = 1 - wm - wr - wv
    az = 1.0 / (1 + 1.0 * z)
    age = 0.
    n = 1000  # number of points in integrals
    for i in range(n):
        a = az * (i + 0.5) / n
        adot = sqrt(wk + (wm / a) + (wr / (a * a)) + (wv * a * a))
        age += 1. / adot
    zage = az * age / n
    zage_gyr = (tyr / h0) * zage
    dtt = 0.0
    dcmr = 0.0
    # do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
    for i in range(n):
        a = az + (1 - az) * (i + 0.5) / n
        adot = sqrt(wk + (wm / a) + (wr / (a * a)) + (wv * a * a))
        dtt += 1. / adot
        dcmr += 1. / (a * adot)

    dtt = (1. - az) * dtt / n
    dcmr = (1. - az) * dcmr / n
    age = dtt + zage
    age_gyr = age * (tyr / h0)
    dtt_gyr = (tyr / h0) * dtt
    dcmr_gyr = (tyr / h0) * dcmr
    dcmr_mpc = (c / h0) * dcmr
    # tangential comoving distance
    x = sqrt(abs(wk)) * dcmr
    if x > 0.1:
        if wk > 0:
            ratio = 0.5 * (exp(x) - exp(-x)) / x
        else:
            ratio = sin(x) / x
    else:
        y = x * x
        if wk < 0:
            y = -y
        ratio = 1. + y / 6. + y * y / 120.
    dcmt = ratio * dcmr
    da = az * dcmt
    da_mpc = (c / h0) * da
    kpc_da = da_mpc / 206.264806
    da_gyr = (tyr / h0) * da
    dl = da / (az * az)
    dl_mpc = (c / h0) * dl
    dl_gyr = (tyr / h0) * dl
    # comoving volume computation
    x = sqrt(abs(wk)) * dcmr
    if x > 0.1:
        if wk > 0:
            ratio = (0.125 * (exp(2. * x) - exp(-2. * x)) - x / 2.) / (x * x * x / 3.)
        else:
            ratio = (x / 2. - sin(2. * x) / 4.) / (x * x * x / 3.)
    else:
        y = x * x
        if wk < 0:
            y = -y
        ratio = 1. + y / 5. + (2. / 105.) * y * y
    vcm = ratio * dcmr * dcmr * dcmr / 3.
    v_gpc = 4. * pi * ((0.001 * c / h0) ** 3) * vcm
    dm_factor = 4 * numpy.pi * (dl_mpc * astrodict['Mpc_in_cm']) ** 2
    if verbose == 1:
        print('For H_o = ' + '%1.1f' % h0 + ', Omega_M = ' + '%1.2f' % wm + ', Omega_vac = ',)
        print('%1.2f' % wv + ', z = ' + '%1.3f' % z)
        print('It is now ' + '%1.1f' % age_gyr + ' Gyr since the Big Bang.')
        print('The age at redshift z was ' + '%1.1f' % zage_gyr + ' Gyr.')
        print('The light travel time was ' + '%1.1f' % dtt_gyr + ' Gyr.')
        print('The comoving radial distance, which goes into Hubbles law, is',)
        print('%1.1f' % dcmr_mpc + ' Mpc or ' + '%1.1f' % dcmr_gyr + ' Gly.')
        print('The comoving volume within redshift z is ' + '%1.1f' % v_gpc + ' Gpc^3.')
        print('The angular size distance D_A is ' + '%1.1f' % da_mpc + ' Mpc or',)
        print('%1.1f' % da_gyr + ' Gly.')
        print('This gives a scale of ' + '%.2f' % kpc_da + ' kpc/".')
        print('The luminosity distance D_L is ' + '%1.1f' % dl_mpc + ' Mpc or ' + '%1.1f' % dl_gyr + ' Gly.')
        print('The distance modulus, m-M, is ' + '%1.2f' % (5 * log10(dl_mpc * 1e6) - 5))
    if not dictreturn:
        return [zage_gyr, dcmr_mpc, kpc_da, (5 * log10(dl_mpc * 1e6) - 5)]
    else:
        return (
            {'ageAtZ': zage_gyr, 'ComovingDistanceMpc': dcmr_mpc, 'Scale': kpc_da, 'DM': (5 * log10(dl_mpc * 1e6) - 5),
             'D_a': da_mpc, 'dl_mpc': dl_mpc, 'dm_factor': dm_factor, 'comvol': v_gpc})


def schechter_l(l, phi, l_star, alpha):
    """
    Evaluates a given schechter function at a magnitude m.
    @param l: magnitude (can be array)
    @param phi: normalization of Schechter: phi_star
    @param l_star: knee of luminosity function, m_star
    @param alpha: shape of luminosity function, alpha
    @return: phi(m)
    """
    phil = phi * (l / l_star) ** alpha * numpy.exp(-l / l_star) / l_star
    return phil


def schechter_m(m, phi, m_star, alpha):
    """
    Evaluates a given schechter function at a magnitude m.
    @param m: magnitude (can be array)
    @param phi: normalization of Schechter: phi_star
    @param m_star: knee of luminosity function, m_star
    @param alpha: shape of luminosity function, alpha
    @return: phi(m)
    """
    phil = 0.4 * numpy.log(10) * phi * (10 ** (-0.4 * (m - m_star))) ** (alpha + 1) \
        * numpy.exp((-10 ** (-0.4 * (m - m_star))))
    return phil


def integrateschechter_l(l_start, l_end, phi, l_star, alpha):
    """
    Integrates a schechter luminosity function
    @param l_start: start of integration as M
    @param l_end: end of integration, as M
    @param phi: normalization of Schechter: phi_star
    @param l_star: knee of luminosity function, m_star
    @param alpha: shape of luminosity function, alpha
    @return: the integral
    """

    def _schechter(m):
        return phi * (m / l_star) ** alpha * numpy.exp(-m / l_star) / l_star

    out = scipy.integrate.quad(_schechter, l_start, l_end)
    return out[0]


def integrateschechter_m(m_start, m_end, phi, m_star, alpha):
    """
    Integrates a schechter luminosity function
    @param l_start: start of integration as M
    @param l_end: end of integration, as M
    @param phi: normalization of Schechter: phi_star
    @param l_star: knee of luminosity function, m_star
    @param alpha: shape of luminosity function, alpha
    @return: the integral
    """

    def _schechter(m):
        return 0.4 * numpy.log(10) * phi * (10 ** (-0.4 * (m - m_star))) ** (alpha + 1) \
            * numpy.exp((-10 ** (-0.4 * (m - m_star))))
    out = scipy.integrate.quad(_schechter, m_start, m_end)
    return out[0]


def random_schechter_l(n, l_star, alpha, l_start, l_end):
    """
    This is slow, and might do much better in log space.
    @param n:
    @param l_star:
    @param alpha:
    @param l_start:
    @param l_end:
    @return:
    """
    def _tmppdf(l):
        return (l / l_star) ** alpha * numpy.exp(-l / l_star) / l_star
    max = _tmppdf(l_start)
    out = []
    while len(out) < n:
        rand_x = numpy.random.uniform(l_start, l_end, 1)[0]
        rand_y = numpy.random.uniform(0, max, 1)[0]
        if _tmppdf(rand_x) > rand_y:
            out.append(rand_x)
    return out
