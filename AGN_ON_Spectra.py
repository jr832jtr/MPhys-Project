import numpy as np
import pandas as pd
import Lightcurves as Lifetimes
import NewFunctions
import MPhysClasses
import bagpipes
import matplotlib.pyplot as plt

#parameters and list initialisation
agnlcparsP = {'t_delay':2e8}#{'downtime':500, 'burstwidth':5e8, 'f_max':2000}
lcpars = {'stretch':2e8, 'loc':1e9}
no_gals = 200
sbscale = 100
obs_wavs = np.arange(3000.0, 6000.0, 0.1)
goodss_filt_list = np.loadtxt("filters/goodss_filt_list.txt", dtype="str")
thresh = 0.5
ind = 0
Column_Names = {0:'Universe Time'}
lognormsfh, bheights = [], []
TimesMasses, TriggerTimes, AGN_ON_Spectras = [], [], []

#generate galaxies
Prob_Sim = MPhysClasses.AGNSFR('delay', tmax = 1e10, deltat = 1e7, agnlcpars = agnlcparsP, 
                               lcpars = lcpars, no_gals = no_gals, sbscale = sbscale, name = 'delay')
masses = Prob_Sim.data['m_gal']
ledds = Prob_Sim.data['ledd']

#Produce SFH and spectra for galaxies
for i in range(len(masses)):
    
    model_components = {}
    model_components["redshift"] = 0.2

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
    
    time_dblp = np.random.uniform(3, 6, 1)[0] #creates randomness in when galaxies formed
    dblplaw['tau'] = time_dblp
    
    time_lgnl = np.random.uniform(11.4, 10, 1)[0]
    lognormal['tmax'] = time_lgnl
    mass_fraction = np.random.uniform(5, 20, 1)[0] #randomness to the amount of mass in a starburst.
    lognormal['massformed'] = np.log10(masses[i]/mass_fraction)
    
    chance = np.random.uniform(0, 1, 1)[0]
    
    if chance > thresh: #Separating starburst galaxies from galaxies with out starbursts
        model_components['lognormal'] = lognormal #Only some galaxies will have starbursts.
        dblplaw['massformed'] = np.log10(masses[i] - (masses[i]/mass_fraction))
        model_components['dblplaw'] = dblplaw
        _galaxy = bagpipes.model_galaxy(model_components, filt_list=goodss_filt_list, spec_wavs=obs_wavs)
        TimesMasses.append({'BurstTime':time_lgnl, 'PwrLawTime':time_dblp, 
                            'BurstMass':lognormal['massformed'], 'PwrLawMass':dblplaw['massformed']})
        lognormsfh.append(_galaxy.sfh.component_sfrs['lognormal'])
        bheights.append(ledds[i])

#Generating agn activity from starburst
Time = (_galaxy.sfh.age_of_universe - _galaxy.sfh.ages)
#bagpipes_df = NewFunctions.AGN('Delay', Time, lognormsfh, bheights) # -DELAYED
#AGN_ON = np.array(bagpipes_df[bagpipes_df['Delayed AGN AR 0'] > 0.0]['Delayed AGN Time 0']) # -DELAYED

bagpipes_df, TriggerTimes = NewFunctions.Generate_AGN('Probabilistic', Time, lognormsfh, bheights) # -PROBABILISTIC
#AGN_Index = bagpipes_df['Universe Time'][np.array(bagpipes_df['Universe Time']) == TriggerTimes[ind]].index[0] # -PROBABILISTIC
AGN_ON = NewFunctions.AGN_Periods(bagpipes_df, TriggerTimes, ind, 25, 1, _galaxy.sfh.age_of_universe)#np.array(bagpipes_df.iloc[AGN_Index:, :]['Universe Time']) # -PROBABILISTIC

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
x, y = zip(*Spectra_Sum)
y = np.array(y)/10**-14
x = np.array(x)/len(AGN_ON_Spectras)
plt.plot(x, y)
ax.set_ylabel("$\\mathrm{f_{\\lambda}}\\ \\mathrm{/\\ 10^{"+ str(-14)+ "}\\ erg\\ s^{-1}\\ cm^{-2}\\ \\AA^{-1}}$")
ax.set_xlabel("$\\lambda / \\mathrm{\\AA}$")
plt.show()
