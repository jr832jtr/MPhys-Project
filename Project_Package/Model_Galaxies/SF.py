#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:51:20 2019

@author: jr832
"""

import numpy as np
import pandas as pd
import bagpipes
from ..Analysis.Setup import Lightcurves as Lifetimes
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import simps
from scipy.interpolate import interp1d

def MainSequenceData(SB_Data, SFH_Dic, Save = False):
    MASS = SB_Data['SB_mass']
    sfh_list = SB_Data['SB_sfh_list']
    sfhs = SB_Data['SB_sfhs']
    spectra = SB_Data['SB_spectra']
    bins = 4
    binlist = []
    model_components = SB_Data['Model_Components']
    lognormsfh = SFH_Dic['lognormlist']
    AoU = Lifetimes.MyCosmology.cosmocal(model_components["redshift"])['ageAtZ']

    for i in range(bins):
        binlist.append([i]*(int(len(MASS)/bins)))

    if (len(MASS) - len(np.ravel(binlist))) != 0:
        diff = (len(MASS) - len(np.ravel(binlist)))
        for i in range(1, diff + 1):
            MASS.remove(MASS[-i])
            sfh_list.remove(sfh_list[-i])
    
    dataf = pd.DataFrame(np.log10(MASS), np.log10(sfh_list)).reset_index()
    dataf = dataf.rename(columns = {'index':'SFRs', 0:'Mass'}).sort_values(by = 'Mass').reset_index().drop('index', axis = 1)
    
    dataf['Groups'] = np.ravel(binlist)
    samples = []

    for name, group in dataf.groupby(by = dataf['Groups']):
        samples.append([group.nlargest(3, 'SFRs', keep = 'all'), group.nsmallest(3, 'SFRs', keep = 'all')])

    size = 3*len(np.unique(binlist))
    sfh_df_max = pd.DataFrame(SB_Data['Time'], columns = ['Age of Universe [Gyr]'])
    sfh_df_min = pd.DataFrame(SB_Data['Time'], columns = ['Age of Universe [Gyr]'])
    Selected_Sfrs_M = []
    Selected_Mass_M = []
    Selected_Sfrs_L = []
    Selected_Mass_L = []
    T_Max, T_Min = [], []
    labels = []
    
    for i in range(size):
        ind_max = list(np.log10(MASS)).index(samples[int(i/3)][0].iloc[i%3, 1])
        ind_min = list(np.log10(MASS)).index(samples[int(i/3)][1].iloc[i%3, 1])
        
        Selected_Sfrs_M.append(samples[int(i/3)][0].iloc[i%3, 0])
        Selected_Mass_M.append(samples[int(i/3)][0].iloc[i%3, 1])
    
        Selected_Sfrs_L.append(samples[int(i/3)][1].iloc[i%3, 0])
        Selected_Mass_L.append(samples[int(i/3)][1].iloc[i%3, 1])
    
        sfh_df_max['Flux {} [M*/yr]'.format(i)] = sfhs[ind_max]
        sfh_df_min['Flux {} [M*/yr]'.format(i)] = sfhs[ind_min]
    
        m_t = SB_Data['Time'][lognormsfh[ind_max] > 1e-3]
        l_t = SB_Data['Time'][lognormsfh[ind_min] > 1e-3]
        T_Max.append(AoU - m_t[-1])
        T_Min.append(AoU - l_t[-1])
    
        if i == 0:
            Spectra_max = pd.DataFrame(spectra[ind_max], columns = ['Observed $\lambda$ [Angstrom]', 'Flux {} [M*/yr]'.format(i)])
            Spectra_min = pd.DataFrame(spectra[ind_min], columns = ['Observed $\lambda$ [Angstrom]', 'Flux {} [M*/yr]'.format(i)])
        else:
            _df1 = pd.DataFrame(spectra[ind_max], columns = ['Observed $\lambda$ [Angstrom]', 'Flux {} [M*/yr]'.format(i)])
            _df2 = pd.DataFrame(spectra[ind_min], columns = ['Observed $\lambda$ [Angstrom]', 'Flux {} [M*/yr]'.format(i)])
            Spectra_max = pd.merge(Spectra_max, _df1, left_on = 'Observed $\lambda$ [Angstrom]', right_on = 'Observed $\lambda$ [Angstrom]')
            Spectra_min = pd.merge(Spectra_min, _df2, left_on = 'Observed $\lambda$ [Angstrom]', right_on = 'Observed $\lambda$ [Angstrom]')
     
        labels.append('Flux {} [M*/yr]'.format(i))
    
    sfh_df_max = sfh_df_max[sfh_df_max['Age of Universe [Gyr]'] > 0.0]
    sfh_df_min = sfh_df_min[sfh_df_min['Age of Universe [Gyr]'] > 0.0]

    Sfrs_n_Masses = pd.DataFrame([Selected_Sfrs_M, Selected_Mass_M, T_Max, Selected_Sfrs_L, Selected_Mass_L, T_Min], columns = labels)
    Sfrs_n_Masses.rename(index = {0:'High SFR log(SFR [M*/yr])', 1:' High SFR log(M [M*])', 2:'High Delta T [Gyr]', 3:'Low SFR log(SFR [M*/yr])', 4:' Low SFR log(M [M*])', 5:'Low Delta T [Gyr]'}, inplace = True)


    Data = {'Max_SFHs':sfh_df_max, 'Min_SFHs':sfh_df_min, 'SFRsMassesTimes':Sfrs_n_Masses, 'Max_Spectra':Spectra_max, 
           'Min_Spectra':Spectra_min}
    
    if Save:
        sfh_df_min.to_csv('Min SF History z={}'.format(model_components['redshift']), sep = ',', index = False)
        sfh_df_max.to_csv('Max SF History z={}'.format(model_components['redshift']), sep = ',', index = False)
        Sfrs_n_Masses.to_csv('SFRs and Masses z={}'.format(model_components['redshift']), sep = ',')
        Spectra_max.to_csv('Max Spectral Data z={}'.format(model_components['redshift']), sep = ',', index = False)
        Spectra_min.to_csv('Min Spectral Data z={}'.format(model_components['redshift']), sep = ',', index = False)
        return None
    elif not Save:
        return Data
    
    

def Generate_SFHs(WL, AGN_df, SB_Prob, Gal_Params, SFH_Only = True, No_SB = False, Save_Data = False):
    
    obs_wavs = np.arange(WL[0], WL[1], 0.1)
    goodss_filt_list = np.loadtxt("filters/goodss_filt_list.txt", dtype="str")
    
    masses = AGN_df.data['m_gal']
    ledds = AGN_df.data['ledd']
    thresh = SB_Prob
    lognormsfh = []
    bheights = []
    TimesMasses = []
    
    if not SFH_Only:
        SB_sfh_list = []
        NSB_sfh_list = []
        SB_sfhs = []
        NSB_sfhs = []
        SB_spectra = []
        NSB_spectra = []
        SB_mass = []
        NSB_mass = []

    for i in range(len(masses)):
    
        lgnl_U, lgnl_L = Gal_Params['lognorm']['Tmax'], Gal_Params['lognorm']['Tmin']
        
        model_components = {}
        model_components["redshift"] = Gal_Params["redshift"]
    
        dust = {}
        dust["type"] = "Calzetti"
        dust["Av"] = 0.2 

        dblplaw = {}
        dblplaw['alpha'] = Gal_Params['dblplaw']['alpha']
        dblplaw['beta'] = Gal_Params['dblplaw']['beta']
        dblplaw['metallicity'] = Gal_Params['dblplaw']['metallicity']
        model_components['dust'] = dust
    
        lognormal = {}
        lognormal['fwhm'] = Gal_Params['lognorm']['fwhm']
        lognormal['metallicity'] = Gal_Params['lognorm']['metallicity']
    
        time_dblp = np.random.uniform(3, 6, 1)[0] #creates randomness in when galaxies formed
        dblplaw['tau'] = time_dblp
        
        time_lgnl = np.random.uniform(lgnl_U, lgnl_L, 1)[0]
        lognormal['tmax'] = time_lgnl
        mass_fraction = np.random.uniform(5, 20, 1)[0] #randomness to the amount of mass in a starburst.
        lognormal['massformed'] = np.log10(masses[i]/mass_fraction)
    
        chance = np.random.uniform(0, 1, 1)[0]
    
        if chance > thresh: #Separating starburst galaxies from galaxies with out starbursts
            model_components['lognormal'] = lognormal #Only some galaxies will have starbursts.
            dblplaw['massformed'] = np.log10(masses[i] - (masses[i]/mass_fraction))
            model_components['dblplaw'] = dblplaw
            _galaxy = bagpipes.model_galaxy(model_components, filt_list=goodss_filt_list, spec_wavs=obs_wavs)
            
            if not SFH_Only:
                SB_sfhs.append(_galaxy.sfh.sfh)
                SB_sfh_list.append(_galaxy.sfh.sfh[0])
                SB_spectra.append(_galaxy.spectrum)
                SB_mass.append(masses[i])
                
            lognormsfh.append(_galaxy.sfh.component_sfrs['lognormal'])
            bheights.append(ledds[i])
            TimesMasses.append({'BurstTime':time_lgnl, 'PwrLawTime':time_dblp, 
                                'BurstMass':lognormal['massformed'], 'PwrLawMass':dblplaw['massformed']})
        elif No_SB: #Galaxies without starbursts
            dblplaw['massformed'] = np.log10(masses[i])
            model_components['dblplaw'] = dblplaw
            _galaxy = bagpipes.model_galaxy(model_components, filt_list=goodss_filt_list, spec_wavs=obs_wavs)
            NSB_sfhs.append(_galaxy.sfh.sfh)
            NSB_sfh_list.append(_galaxy.sfh.sfh[0])
            NSB_spectra.append(_galaxy.spectrum)
            NSB_mass.append(masses[i])
    
    Time = (_galaxy.sfh.age_of_universe - _galaxy.sfh.ages)
    Time_Gyr = Time*10**-9
    SFH_Dic = {'Time':Time, 'Galaxy':_galaxy, 'lognormlist':lognormsfh, 'AccRates':bheights, 'TimesMasses':TimesMasses}
    
    if not SFH_Only:
        SB_Data = {'SB_mass':SB_mass, 'SB_sfh_list':SB_sfh_list, 'SB_sfhs':SB_sfhs, 'SB_spectra':SB_spectra, 'Time':Time_Gyr, 
                  'Model_Components':model_components}
    
    if SFH_Only:
        return SFH_Dic
    elif not SFH_Only:
        Data = MainSequenceData(SB_Data, SFH_Dic, Save = Save_Data)
        out = {'Data':Data, 'SB_Data':SB_Data, 'SFH_Data':SFH_Dic}
        return out




def Mass_Calculator(init_z, obs_z, sfh, number, n, T_min):
    cosmol = FlatLambdaCDM(H0=70., Om0=0.3)
    age = cosmol.age(init_z).value*1e9
    #point = MyCosmology.cosmocal(obs_z)['ageAtZ']*1e9
    Mass, Time = [], []
    SFHs = sfh['TimesMasses'][number]
    
    if T_min is None:
        peak_time = SFHs['BurstTime']*1e9
        tmin = peak_time - 4e8
        deltat = (age - tmin)/(n)
    elif T_min:
        tmin = T_min
        deltat = (age - tmin)/(n)
    
    for i in range(n):
        x = sfh['Time'][sfh['Time'] > (tmin + i*deltat)]
        y = sfh['lognormlist'][number][sfh['Time'] > (tmin + i*deltat)]
        Mass.append(simps(y, x)*-1)
        Time.append(tmin + i*deltat)
    
    _f = interp1d(Time, max(Mass) - Mass)
    
    return _f(obs_z)/max(Mass)