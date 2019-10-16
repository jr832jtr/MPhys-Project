__author__ = 'cvillforth'
import numpy
import scipy.stats
import copy
import pylab
import scipy.interpolate
import MyExceptions
import MyFunctions
import HostGalaxies
import MyCosmology
import MyAstroCalc

def lognorm_lightcurve(t_max, deltat, stretch=1, s=1, scale=1, loc=None, peakatone=False, norm=1):
    t = numpy.arange(0, t_max, deltat)
    if loc is None:
        loc = 0.5*t_max
    f = scipy.stats.lognorm.pdf(t / stretch, s=s, loc=loc/stretch, scale=scale)
        
    if peakatone:
        
        if (loc/stretch) >= ((t_max/stretch) - (deltat/stretch)): #ME
            f = numpy.zeros_like(f) #ME
        else: #ME
            f /= numpy.max(f)
        
    return t, f*norm


def burst_delta(t_max, deltat, burstlength, burstheight, bursttime):
    """
    A collection of delta bursts
    @param t_max: length of lightcurve
    @param deltat: timestep
    @param burstlength: length of individual burst
    @param burstheight: height of burst
    @param bursttime: times at which individual bursts appear
    @return: t, lightcurve
    """
    t = numpy.arange(0, t_max, deltat)
    f = numpy.zeros_like(t)
    if type(bursttime) in [float, int]:
        bursttime = [bursttime]
    if type(burstheight) in [float, int]:
        if len(bursttime) == 1:
            burstheight = [burstheight]
        else:
            burstheight = len(bursttime) * [burstheight]
    for b, h in zip(bursttime, burstheight):
        burstmask = (t >= b) & (t <= (b + burstlength))
        f[burstmask] = h
    return t, f

def burst_lognorm(t_max, deltat, burstheight, burstlength, bursttime, peakatone=True):
    """
    A single gaussian burst or a collection of gaussian bursts
    @param t_max: length of lightcurve
    @param deltat: timestep
    @param burstheight: height(s) of burst(s)
    @param burstlength: width(s) of burst(s)
    @param bursttime: time(s) of burst(s)
    @param peakatone: peak at one (rather than integrate to 1)
    @return: t, lightcurve
    """
    t = numpy.arange(0, t_max, deltat)
    f = numpy.zeros_like(t)
    if type(burstheight) in [float, int]:
        burstheight = [burstheight]
    if type(burstlength) in [float, int]:
        burstlength = [burstlength]
    if type(bursttime) in [float, int]:
        bursttime = [bursttime]
    for mu, sigma, scale in zip(bursttime, burstlength, burstheight):
        f += lognorm_lightcurve(t_max, deltat, loc=mu, stretch=sigma, norm=scale, peakatone=peakatone)[1]
    return t, f


def delay(t, f, t_delay=1e6, scale=1, baseline=0, randomheight=False, randdist='aird', randpars=None, scalecutoff=0.01):
    """
    Delay one lightcurve with respect to another
    @param scalecutoff:
    @param randpars:
    @param randdist:
    @param randomheight:
    @param t:
    @param f:
    @param t_delay:
    @param scale:
    @param baseline:
    @return:
    """
    if randpars is None:
        randpars = {}
    if randomheight is True and randdist not in ['norm', 'lorgnorm', 'aird']:
        raise MyExceptions.InputError("random distribution for heights must be norm, lognorm, or aird")
    if randomheight:
        if randdist == 'lognorm':
            curscale = numpy.random.lognormal(size=1, **randpars)
        elif randdist == 'norm':
            curscale = numpy.random.normal(size=1, **randpars)
            if curscale < scalecutoff:
                curscale = 0
        elif randdist == 'aird':
            curscale = randomledd_aird(1, **randpars)
        else:
            raise MyExceptions.Hell("one of those things that should not happen")
    else:
        curscale = 1.
    ind_start = numpy.where(t > t_delay)[0][0]
    f_out = numpy.zeros_like(f)
    if ind_start > 0:
        f_out[0:ind_start] = baseline
    else:
        print "Warning, delay is below resolution of data!"
    _n = len(t)
    #f_out = numpy.concatenate((numpy.empty(ind_start) * baseline, f[ind_start:]))
    for i in range(_n - ind_start):
        f_out[i + ind_start] = f[i] * scale
    return t, f_out * curscale


def random_burst(t_max, deltat, downtime, burstlength, burstheight, bursttype='gauss',
                 randomheight=False, randdist='lognorm', randpars=None, scalecutoff=0.01, peakatone=True):
    """
    A collection of random bursts.
    @param peakatone:
    @param scalecutoff:
    @param randomheight:
    @param randpars:
    @param randdist:
    @param bursttype:
    @param bursttype:
    @param t_max:
    @param deltat:
    @param downtime:
    @param burstlength:
    @param burstheight:
    @return:
    """
    if bursttype not in ['delta', 'gauss', 'lognorm']:
        raise MyExceptions.InputError("burst type must be delta or gauss")
    if randomheight is True and randdist not in ['norm', 'lognorm', 'aird']:
        raise MyExceptions.InputError("random distribution for heights must be norm, lognorm, or aird")
    if not randpars:
        randpars = {}
    n_burst = int(t_max / downtime)
    bursttimes = numpy.random.uniform(0, t_max, n_burst)
    burstlength = [burstlength] * len(bursttimes)
    if type(burstheight) is list and randomheight is True:
        raise MyExceptions.InputError("hey, you, not allowed")
    elif randomheight is True:
        if randdist == 'lognorm':
            randscale = numpy.random.lognormal(size=n_burst, **randpars)
        elif randdist == 'norm':
            randscale = numpy.random.normal(size=n_burst, **randpars)
        elif randdist == 'aird':
            randscale = randomledd_aird(1, **randpars)
        else:
            raise MyExceptions.Hell("one of those things that should not happen")
        cutoffmask = randscale < scalecutoff
        randscale[cutoffmask] = 0
        burstheight *= randscale
    else:
        burstheight = [burstheight] * len(bursttimes)
    if bursttype == 'delta':
        out = burst_delta(t_max, deltat, burstlength=burstlength, burstheight=burstheight, bursttime=bursttimes)
    elif bursttype == 'gauss':
        out = burst_gaussian(t_max, deltat, burstlength=burstlength, burstheight=burstheight, bursttime=bursttimes,
                             peakatone=peakatone)
    elif bursttype == 'lognorm':
        out = burst_lognorm(t_max, deltat, burstlength=burstlength, burstheight=burstheight, bursttime=bursttimes,
                            peakatone=peakatone)
    else:
        raise MyExceptions.Hell("This is one of those things: should not happen!")
    return out


def probabilistic_deltaburst(t, f, burstlength, burstheight, f_max=None,
                             randomheight=True, randdist='lognorm', randpars=None, scalecutoff=0.01):
    """
    A collection of bursts that appear randomly but with a probability that scales with the input flux
    @param scalecutoff:
    @param randpars:
    @param randdist:
    @param randomheight:
    @param t:
    @param f:
    @param burstlength:
    @param burstheight:
    @param f_max:
    @return:
    """
    if randpars is None:
        randpars = {}
    if randomheight is True and randdist not in ['norm', 'lognorm', 'aird']:
        raise MyExceptions.InputError("random distribution for heights must be norm, lognorm, or aird")
    f_out = numpy.zeros_like(f)
    agn_on = False
    agn_start = 0
    curscale = 1.
    _n = len(t)
    for _t, _f, i in zip(t, f, range(_n)):
        if agn_on is False:
            agn_on = randomburst_linear(_f, f_max=f_max)
            if agn_on:
                agn_start = _t
                if randomheight:
                    if randdist == 'lognorm':
                        curscale = numpy.random.lognormal(size=1, **randpars)
                    elif randdist == 'norm':
                        curscale = numpy.random.normal(size=1, **randpars)
                    elif randdist == 'aird':
                        curscale = randomledd_aird(1, **randpars)
                    else:
                        raise MyExceptions.Hell("One of those things that should not be happening")
                if curscale < scalecutoff:
                    curscale = 0
                else:
                    curscale = 1.
        if agn_on is True and _t < agn_start + burstlength:
            f_out[i] = burstheight * curscale
        elif agn_on is True and _t >= agn_start + burstlength:
            agn_on = False
    return t, f_out

def probabilistic_lognorm(t, f, burstheight, burstwidth, cuttail_low=None, f_max=None,
                                downtime=2, randomheight=True, randdist='lognorm', randpars=None, scalecutoff=0.01):
    """
    A collection of bursts that appear randomly but with a probability that scales with the input flux
    @param t:
    @param f:
    @param burstheight:
    @param burstwidth:
    @param cuttail_low:
    @param cuttail_high:
    @param f_max:
    @param downtime:
    @param randomheight:
    @param randdist:
    @param randpars:
    @param scalecutoff:
    @param peakatone:
    @return:
    """
    if randpars is None:
        randpars = {}
    if cuttail_low is None:
        cuttail_low = burstwidth
    if randomheight is True and randdist not in ['norm', 'lognorm', 'aird']:
        raise MyExceptions.InputError("random distribution for heights must be norm, lognorm, or aird")
    f_out = numpy.zeros_like(f)
    agn_on = False
    agn_start = 0
    _n = len(t)
    for _t, _f, i in zip(t, f, range(_n)):
        if agn_on is False:
            agn_on = randomburst_linear(_f, f_max=f_max)
            if agn_on:
                agn_start = _t
                if randomheight:
                    if randdist == 'lognorm':
                        curscale = numpy.random.lognormal(size=1, **randpars)
                    elif randdist == 'norm':
                        curscale = numpy.random.normal(size=1, **randpars)
                    elif randdist == 'aird':
                        curscale = randomledd_aird(1, **randpars)
                    else:
                        raise MyExceptions.Hell("one of these things that should not be happening")
                    if curscale < scalecutoff:
                        curscale = 0
                else:
                    curscale = 1.
                f_out += lognorm_lightcurve(max(t)+abs(t[1]-t[0]), abs(t[1]-t[0]), stretch=burstwidth,
                                                       norm= burstheight*curscale, loc = _t)[1]
        if agn_on is True and _t >= agn_start + downtime * burstwidth:
            agn_on = False
    return t, f_out



def damped_walk(t, f, delta_sfr, scale, weight_drw, sigma):
    N = len(t)
    f_agn = numpy.zeros_like(f)
    int_sfr = scipy.interpolate.interp1d(t, f, fill_value='extrapolate')
    for i in range(1, N):
        del_mean = -weight_drw*(f_agn[i-1] - scale * int_sfr(t[i] - delta_sfr))
        dL = (numpy.random.standard_normal(1) * sigma) + del_mean
        f_agn[i] = f_agn[i-1] + dL
    return f_agn



def truncate_agnlc(f, ll, setto=0):
    """
    Truncate a lioghtcurve and setting all values below a certain limit to a value (0)
    @param f:
    @param ll:
    @param setto:
    @return:
    """
    mask = numpy.where(f < ll)
    f_out = copy.copy(f)
    f_out[mask] = setto
    return f_out


def randomburst_linear(f, f_max=None, threshold=0):
    """
    Trigger a probabilistic burst
    @param threshold:
    @param f:
    @param f_max:
    @return:
    """
    if f_max is None:
        f_max = 1000
    p_cut = (f - threshold) / float(f_max - threshold)
    randn = numpy.random.uniform()
    if randn < p_cut:
        return True
    else:
        return False


def randomlightcurves(n=10000):
    """
    Basically noise
    @param n:
    @return:
    """
    t = numpy.arange(n)
    lagn = numpy.random.standard_normal(n)
    return t, lagn


def noisylightcurve(lc, scatter):
    """
    Noise up a lightcurve
    @param lc:
    @param scatter:
    @return:
    """
    n = numpy.shape(lc)
    noise = numpy.random.standard_normal(n) * scatter
    lognoisy = numpy.log10(lc) + noise
    out = numpy.power(10, lognoisy)
    return out

def simu_lx_sfr(n_gal, bursterror, tmax=1e9, deltat=1e6, galpoppars=None, mstype='poly', mspars=None, lctype='lognorm', lcpars=None,
                msigmapars=None, agnlctype='delay', agnlcpars=None, sbscale=10, sbbaseline=1, truncateedd=0.001,
                skiplc=False, cannedgalaxies=True, rs_exp=False): #bursterror argument is ME
    """
    All in one SFR-AGN simulation.
    @param n_gal:
    @param tmax:
    @param deltat:
    @param galpoppars:
    @param mstype:
    @param mspars:
    @param lctype:
    @param lcpars:
    @param msigmapars:
    @param agnlctype:
    @param agnlcpars:
    @param sbscale:
    @param sbbaseline:
    @param truncateedd:
    @return:
    """
    if cannedgalaxies is True and skiplc is True:
        raise MyExceptions.StupidError("This is stupid")
    if agnlcpars is None:
        agnlcpars = {}
    if mstype not in ['poly', 'pl']:
        raise MyExceptions.InputError("Main sequence type (mstype) must be poly or pl")
    if lctype not in ['exp', 'chi', 'norm', 'lognorm']:
        raise MyExceptions.InputError("lightcurve type (lctype) must be exp or chi")
    if agnlctype not in ['delay', 'random', 'prob_delta', 'prob_gauss', 'prob_lognorm']:
        raise MyExceptions.InputError("AGN lightcurve type (agnlctype) must be delay, random, prob_delta, prob_gauss")
    if galpoppars is None:
        galpoppars = {'l_star': 10e10, 'alpha': -1.25, 'l_start': 1e8, 'l_end': 1e12}
    if mspars is None:
        mspars = {}
    if lcpars is None:
        if lctype == 'exp':
            lcpars = {'tau': 2e8}
        elif lctype == 'chi':
            lcpars = {'df': 2, 'stretch': 1e8}
        elif lctype == 'norm':
            lcpars = {'pos': 1e8, 'width': 1e8}
        elif lctype == 'lognorm':
            lcpars = {'s': 1, 'stretch': 1e7}
        else:
            raise MyExceptions.Hell("Shouldn't happen")    
    if msigmapars is None:
        msigmapars = {}
    if agnlctype == 'delay':
        if 't_delay' not in agnlcpars.keys():
            agnlcpars['t_delay'] = 2e8
    elif agnlctype == 'random':
        tmax += bursterror #ME
        if 'downtime' not in agnlcpars.keys():
            agnlcpars['downtime'] = 1e7
            agnlcpars['scalecutoff'] = 0
        if 'burstlength' not in agnlcpars.keys():
            agnlcpars['burstlength'] = 1e7
    elif agnlctype == 'prob_delta':
        if 'burstlength' not in agnlcpars.keys():
            agnlcpars['burstlength'] = 1e7
        if 'f_max' not in agnlcpars.keys():
            agnlcpars['f_max'] = 100
    elif agnlctype == 'prob_gauss':
        if 'burstwidth' not in agnlcpars.keys():
            agnlcpars['burstwidth'] = 1e7
            agnlcpars['scalecutoff'] = 0
        if 'f_max' not in agnlcpars.keys():
            agnlcpars['f_max'] = 100
    elif agnlctype == 'prob_lognorm':
        if 'burstwidth' not in agnlcpars.keys():
            agnlcpars['burstwidth'] = 1e7
    else:
        raise MyExceptions.Hell("Should not be happening.")
    t = numpy.arange(0, tmax, deltat)
    if cannedgalaxies:
        getsample = db.fetch("""select m_gal, sfr, m_bh, l_edd, mean_ms
                             from galaxysample order by rand() limit %i""" % n_gal)
        m_gal = getsample[:, 0]
        sfrs = getsample[:, 1]
        bhmass = getsample[:, 2]
        ledd = getsample[:, 3]
        mean_sfrs = getsample[:, 4]
    else:
        # draw random galaxies
        print "Drawing galaxy sample....."
        m_gal = MyCosmology.random_schechter_l(n_gal, **galpoppars)
        # calculate sfrs
        print "Getting Main Sequence calibration...."
        if mstype == 'poly':
            sfrs, mean_sfrs = HostGalaxies.mgalsfr_poly(m_gal, **mspars)
        elif mstype == 'pl':
            sfrs, mean_sfrs = HostGalaxies.mgalsfr_brokenpl(m_gal, **mspars)
        else:
            raise MyExceptions.Hell("One of those things that should not happen.")
        # calculate BH masses
        # calculate Eddington Luminosity
        print "Getting black hole masses and Eddington luminosities....."
        bhmass = HostGalaxies.msigma(m_gal, **msigmapars)
        ledd = MyAstroCalc.EddingtonLuminosity(10 ** bhmass)
    if skiplc:
        return {'m_gal': m_gal, 'sfr': sfrs, 'bhmass': bhmass, 'ledd': ledd}
    print "Creating lightcurves...."
    # simulate lightcurves using sfr as scaling (there might be an issue here of that not turning out quite fine)
    # this is not elegant, use numpy.outer instead of those silly loops
    lightcurves_sfr = []
    if lctype == 'exp':
        for sfr in sfrs:
            tmp_lc = exponential_lightcurve(t_max=tmax, deltat=deltat, no=sfr * sbscale, **lcpars)[1]
            if rs_exp:
                tmp_lc = rescale_exp(tmp_lc, rs_exp)
            lightcurves_sfr.append(tmp_lc)
    elif lctype == 'chi':
        for sfr in sfrs:
            tmp_lc = chi_lightcurve(t_max=tmax, deltat=deltat, scale=sfr * sbscale, **lcpars)[1]
            if rs_exp:
                tmp_lc = rescale_exp(tmp_lc, rs_exp)
            lightcurves_sfr.append(tmp_lc)
    elif lctype == 'norm':
        for sfr in sfrs:
            tmp_lc = MyFunctions.gaussian(t, lcpars['pos'], lcpars['width'])
            tmp_lc *= (sfr * sbscale) / numpy.max(tmp_lc)
            if rs_exp:
                tmp_lc = rescale_exp(tmp_lc, rs_exp)
            lightcurves_sfr.append(tmp_lc)
    elif lctype == 'lognorm':
         for sfr in sfrs:
            if agnlctype == 'random':
                tmp_lc = lognorm_lightcurve(t_max=tmax, deltat=deltat, **lcpars)[1][int(bursterror/deltat):]
            else:
                tmp_lc = lognorm_lightcurve(t_max=tmax, deltat=deltat, **lcpars)[1]
            tmp_lc *= sfr * sbscale
             #tmp_lc = lognorm_lightcurve(t_max=tmax, deltat=deltat, scale=sfr * sbscale, **lcpars)[1]
             #if rs_exp:
             #    tmp_lc = rescale_exp(tmp_lc, rs_exp)
            lightcurves_sfr.append(tmp_lc)
    else:
        raise MyExceptions.Hell("One of those things that should not happen.")
    peak_sfr = t[numpy.argmax(lightcurves_sfr[0])]
    # simulate
    print "Creating AGN lightcurves...."
    lightcurves_agn = []
    if agnlctype == 'delay':
        for _ledd, _lcsfr, _sfr in zip(ledd, lightcurves_sfr, sfrs):
            lightcurves_agn.append(delay(t, _lcsfr / _sfr, scale=_ledd, **agnlcpars)[1])
    elif agnlctype == 'random':
        t = t[int(bursterror/deltat):] - bursterror #ME
        for _ledd in ledd:
            if agnlcpars['downtime'] < 1e7: #ME
                raise Exception('Downtime less than 1e7 may lead to crash: downtime is hardwired for \'agnlctype = random\'') #ME
            lightcurves_agn.append(random_burst(t_max=tmax, deltat=deltat, burstheight=_ledd, bursttype = 'lognorm', **agnlcpars)[1][int(bursterror/deltat):]) #ME
    elif agnlctype == 'prob_delta':
        for _ledd, _lcsfr, _sfr in zip(ledd, lightcurves_sfr, sfrs):
            lightcurves_agn.append(probabilistic_deltaburst(t, _lcsfr / _sfr, burstheight=_ledd,
                                                            **agnlcpars)[1])
    elif agnlctype == 'prob_gauss':
        for _ledd, _lcsfr, _sfr in zip(ledd, lightcurves_sfr, sfrs):
            lightcurves_agn.append(probabilistic_gaussianburst(t, _lcsfr / _sfr, burstheight=_ledd,
                                                               **agnlcpars)[1])
    elif agnlctype == 'prob_lognorm':
        for _ledd, _lcsfr, _sfr in zip(ledd, lightcurves_sfr, sfrs):
            lightcurves_agn.append(probabilistic_lognorm(t, _lcsfr / _sfr, burstheight=_ledd,
                                                               **agnlcpars)[1])
    else:
        raise MyExceptions.Hell("One of those things that should not happen.")
    print "Adding SF baseline..."
    for lc, sf in zip(lightcurves_sfr, sfrs):
        lc += sf * sbbaseline
    truncated = numpy.zeros_like(lightcurves_agn)
    if truncateedd is not False:
        print "Truncating AGN lightcurves....."
        for _lagn, _ledd, i in zip(lightcurves_agn, ledd, range(n_gal)):
            truncated[i] = truncate_agnlc(_lagn, _ledd * truncateedd)
    return {'m_gal': m_gal, 'sfr': sfrs, 'bhmass': bhmass, 'ledd': ledd, 'lc_sfr': numpy.array(lightcurves_sfr),
            'lc_agn_nottruncated': numpy.array(lightcurves_agn), 't': t, 'lc_agn': numpy.array(truncated),
            'peak_sfr': peak_sfr, 'mean_sfrs': mean_sfrs, 'n_obj': n_gal, 'n_lc': len(lightcurves_sfr[0])}
