import numpy as np
import pandas as pd
import Lightcurves as Lifetimes
import NewFunctions
import MPhysClasses
import bagpipes
import matplotlib.pyplot as plt

#parameters and list initialisation
#agnlcparsP = {'t_delay':2e8} # -DELAYED
agnlcparsP = {'downtime':500, 'burstwidth':5e8, 'f_max':2000} # -PROBABILISTIC
lcpars = {'stretch':2e8, 'loc':1e9}
no_gals = 200
sbscale = 100
obs_wavs = np.arange(3000.0, 6000.0, 0.1)
goodss_filt_list = np.loadtxt("filters/goodss_filt_list.txt", dtype="str")
thresh = 0.5
ind = 0
Column_Names = {0:'Universe Time'}
dblplaw_params = {'alpha':0.5, 'beta':10, 'metallicity':0.2}
lognorm_params = {'Tmax':11.4, 'Tmin':10, 'fwhm':0.2, 'metallicity':0.75}
Gal_Params = {'dblplaw':dblplaw_params, 'lognorm':lognorm_params, 'redshift':0.2}
TimesMasses, TriggerTimes, AGN_ON_Spectras = [], [], []

#generate galaxies
Prob_Sim = MPhysClasses.AGNSFR('prob_lognorm', tmax = 1e10, deltat = 1e7, agnlcpars = agnlcparsP, lcpars = lcpars, no_gals = no_gals, sbscale = sbscale, name = 'Prob Sim')

#Produce SFH and spectra for galaxies
SFHs = NewFunctions.Generate_SFHs([3000., 6000.], Prob_Sim, 0.5, Gal_Params)

#Generating agn activity from starburst
Time = SFHs['Time']
_galaxy = SFHs['Galaxy']
TimesMasses = SFHs['TimesMasses']
#bagpipes_df = NewFunctions.Generate_AGN('Delay', Time, SFHs['lognormlist'], SFHs['AccRates']) # -DELAYED
#TriggerTimes = [[agnlcparsP['t_delay']]]*len(SFHs['lognormlist']) # -DELAYED

bagpipes_df, TriggerTimes = NewFunctions.Generate_AGN('Probabilistic', Time, SFHs['lognormlist'], SFHs['AccRates'], BurstWidth = 2e7, DownTime = 4) # -PROBABILISTIC
AGN_ON = NewFunctions.AGN_Periods('Prob', bagpipes_df, TriggerTimes, ind, 25, 1, _galaxy.sfh.age_of_universe)

#Reproducing galaxies but ready to take spectra over time
for i in range(len(AGN_ON)):
    for j in range(len(AGN_ON[i])):
        _Z = NewFunctions.Z_Calc(AGN_ON[i][j]*10**-9)
        model_components = {}
        model_components["redshift"] = _Z

        dust = {}
        dust["type"] = "Calzetti"
        dust["Av"] = 0.2 

        dblplaw = {}
        dblplaw['alpha'] = 0.5
        dblplaw['beta'] = 10
        dblplaw['metallicity'] = 0.2
        model_components['dust'] = dust
    
        lognormal = {}
        lognormal['fwhm'] = 0.2
        lognormal['metallicity'] = 0.75
    
        dblplaw['tau'] = TimesMasses[ind]['PwrLawTime']
        lognormal['tmax'] = TimesMasses[ind]['BurstTime']

        lognormal['massformed'] = TimesMasses[ind]['BurstMass']
        dblplaw['massformed'] = TimesMasses[ind]['PwrLawMass']
        model_components['lognormal'] = lognormal
        model_components['dblplaw'] = dblplaw
        _galaxy = bagpipes.model_galaxy(model_components, filt_list=goodss_filt_list, spec_wavs=obs_wavs)
        AGN_ON_Spectras.append(_galaxy.spectrum)

Spectra_Sum = sum(AGN_ON_Spectras)
Spectra_Average = Spectra_Sum/len(AGN_ON_Spectras)

fig = plt.figure(figsize = (12, 4))
ax = plt.subplot()
x, y = zip(*Spectra_Average)
y = np.array(y)/10**-14
#x = np.array(x)/len(AGN_ON_Spectras)
plt.plot(x, y)
ax.set_ylabel("$\\mathrm{f_{\\lambda}}\\ \\mathrm{/\\ 10^{"+ str(-14)+ "}\\ erg\\ s^{-1}\\ cm^{-2}\\ \\AA^{-1}}$")
ax.set_xlabel("$\\lambda / \\mathrm{\\AA}$")
plt.show()
