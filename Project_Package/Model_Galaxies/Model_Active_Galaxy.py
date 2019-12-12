#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:26:20 2019

@author: jr832
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import bagpipes
from . import AGN
from . import SF

class Active_Galaxy():
    
    def __init__(self, agnparams, Sim, gal_params, obs_wavs, gal_no, Res = 1000, fraction = 0.5, Thresh = 25, Scaling = 1e44, Show_Lc = False, IntSpec = False, Tmin = None):
        self.Agn_Type = agnparams['AGN_Type']
        self.Gal_Params = gal_params
        self.Galaxies = self.SFHs(obs_wavs, Sim, fraction, gal_params)
        agnparams['TimesMasses'] = self.Galaxies['TimesMasses']
        self.AGN_Params = agnparams
        
        if self.Agn_Type != 'Delay':
            self.Active_Galaxies, self.TriggerTimes = AGN.Generate_AGN(self.Galaxies['Time'], self.Galaxies['lognormlist'], self.Galaxies['AccRates'], self.AGN_Params)
        elif self.Agn_Type == 'Delay':
            self.Active_Galaxies = AGN.Generate_AGN(self.Galaxies['Time'], self.Galaxies['lognormlist'], self.Galaxies['AccRates'], self.AGN_Params)
            self.TriggerTimes = [[agnparams['Delay']]]*len(self.Galaxies['lognormlist'])
        else:
            raise Exception('Error: agnparams[\'AGN_Type\'] must be \'Delay\', \'Random\' or \'Probabilistic\'.')
        
        self.Filters = np.loadtxt("filters/goodss_filt_list.txt", dtype="str")
        self.Obs_Wavs = obs_wavs
        self.Wavelengths, self.Flux = self.Generate_Spectra(gal_no, Thresh = Thresh, Scaling = Scaling, show = Show_Lc, Res = Res, AGN_Off = IntSpec, Tmin = Tmin)
        
        
        
    def SFHs(self, Obs_Wavs, Sim, Fraction, Params):
        return SF.Generate_SFHs(Obs_Wavs, Sim, Fraction, Params)
    
    
    
    def Generate_Spectra(self, Number, Thresh, Scaling, show, Res, AGN_Off = False, Tmin = None):
        AGN_ON_WL, AGN_ON_Flux = [], []
        
        if AGN_Off:
            _df = self.Active_Galaxies
            minT =  self.Galaxies['TimesMasses'][Number]['PwrLawTime']*10**9 #min(_df['Universe Time'][_df['SFH {}'.format(Number)] > 0.0])
            if (not Tmin) or (Tmin < minT):
                raise Exception('Error: Tmin cannot be \'None\' value and must be greater than {}.'.format(minT))
            Tmax = max(_df['Universe Time'.format(Number)].dropna())
            DeltaT = Tmax/Res
            AGN_ON = [list(np.arange(Tmin, Tmax, DeltaT))]
        else:
            AGN_ON = AGN.AGN_Periods(self.Agn_Type, self.Active_Galaxies, self.TriggerTimes, Number, Thresh, Scaling, self.Galaxies['Galaxy'].sfh.age_of_universe, Res = Res, show = show)
            
        for i in range(len(AGN_ON)):
            for j in range(len(AGN_ON[i])):
                _Z = AGN.Z_Calc(AGN_ON[i][j]*10**-9)
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
    
                dblplaw['tau'] = self.Galaxies['TimesMasses'][Number]['PwrLawTime']
                lognormal['tmax'] = self.Galaxies['TimesMasses'][Number]['BurstTime']
                
                try:
                    if (AGN_Off) or (self.Agn_Type == 'Random'):
                        mass_frac_dblp = SF.Mass_Calculator(self.Gal_Params['redshift'], AGN_ON[i][j], self.Galaxies, Number, 100, T_min = Tmin, component = 'dblplaw', dblp = dblplaw)
                        if AGN_ON[i][j] > min(self.Active_Galaxies['Universe Time'][self.Active_Galaxies['SFH {}'.format(Number)] > 0.0]):
                            mass_frac = SF.Mass_Calculator(self.Gal_Params['redshift'], AGN_ON[i][j], self.Galaxies, Number, 100, T_min = AGN_ON[0][0] - 4e8)
                            if mass_frac != 0:
                                lognormal['massformed'] = np.log10(mass_frac*10**self.Galaxies['TimesMasses'][Number]['BurstMass'])
                                model_components['lognormal'] = lognormal
                    else:
                        mass_frac_dblp = 1
                        mass_frac = SF.Mass_Calculator(self.Gal_Params['redshift'], AGN_ON[i][j], self.Galaxies, Number, 100, T_min = AGN_ON[0][0] - 4e8)
                        if mass_frac != 0:
                            lognormal['massformed'] = np.log10(mass_frac*10**self.Galaxies['TimesMasses'][Number]['BurstMass'])
                            model_components['lognormal'] = lognormal      
                except ValueError:
                    print('BANG!')
                    break                  

                dblplaw['massformed'] = np.log10(mass_frac_dblp*10**self.Galaxies['TimesMasses'][Number]['PwrLawMass'])
                model_components['dblplaw'] = dblplaw
                _galaxy = bagpipes.model_galaxy(model_components, filt_list=self.Filters, spec_wavs=self.Obs_Wavs)
                
                log_wavs = np.log10(_galaxy.wavelengths)
                full_spectrum = _galaxy.spectrum_full*_galaxy.lum_flux*_galaxy.wavelengths
                wavs = (log_wavs > 2.75) & (log_wavs < 6.75)
                log_wavs = log_wavs[wavs]
                full_spectrum = full_spectrum[wavs]
                dat = pd.DataFrame([10**log_wavs, np.log10(full_spectrum)]).T.rename(columns = {0:'Log WL', 1:'Flux'})
                dat = dat[(dat['Log WL'] > 2000) & (dat['Log WL'] < 6000)]
                
                AGN_ON_WL.append(np.array(dat['Log WL']))
                AGN_ON_Flux.append(np.array(dat['Flux']))
                
        return AGN_ON_WL, AGN_ON_Flux
                
                
    def Plot_Average_Spectra(self, Number, Save_Data = False):
        mpl.rc('text', usetex=True)
        mpl.rcParams["text.usetex"] = True
        
        Spectra_Sum, Spectra_WL = sum(self.Flux), self.Wavelengths[0]
        Spectra_Average = Spectra_Sum/len(self.Flux)

        fig = plt.figure(figsize = (12, 4))
        ax = plt.subplot()
        x, y = Spectra_WL, Spectra_Average
        
        #scale = self.Decimal_Calculator(max(y))
        
        y = np.array(y)#/10**-scale
        
        plt.plot(x, y)
        ax.set_ylabel("$\\mathrm{log_{10}}\\big(\\mathrm{\\lambda L_{\\lambda}}\\ \\mathrm{/\\ erg" + "\\ s^{-1}}\\big)$")
        ax.set_xlabel("$\\lambda / \\mathrm{\\AA}$")
        
        if Save_Data:
            Spectral_Data = pd.DataFrame(data = {'Average Flux':Spectra_Average, 'Wavelengths':Spectra_WL})
            Spectral_Data.to_csv('{}AvgSpectralData{}'.format(self.Agn_Type, Number), sep = ',', index = False)
            plt.savefig('Avg{}Spec{}'.format(self.Agn_Type, Number))
        
        plt.show()
        
        
        
    def Plot_Summed_Spectra(self, Number, Save_Data = False):
        mpl.rc('text', usetex=True)
        mpl.rcParams["text.usetex"] = True
        
        Spectra_Sum, Spectra_WL = sum(self.Flux), self.Wavelengths[0]
        
        fig = plt.figure(figsize = (12, 4))
        ax = plt.subplot()
        x, y = Spectra_WL, Spectra_Sum
        
        #scale = self.Decimal_Calculator(max(y))
        
        y = np.array(y)#/10**-scale
        x = np.array(x)#/len(self.Spectra)
        
        plt.plot(x, y)
        ax.set_ylabel("$\\mathrm{log_{10}}\\big(\\mathrm{\\lambda L_{\\lambda}}\\ \\mathrm{/\\ erg" + "\\ s^{-1}}\\big)$")
        ax.set_xlabel("$\\lambda / \\mathrm{\\AA}$")
        
        if Save_Data:
            Spectral_Data = pd.DataFrame(data = {'Summed Flux':Spectra_Sum, 'Wavelengths':Spectra_WL})
            Spectral_Data.to_csv('{}SumSpectralData{}'.format(self.Agn_Type, Number), sep = ',', index = False)
            plt.savefig('Sum{}Spec{}'.format(self.Agn_Type, Number))
        
        plt.show()
        
        
    def Decimal_Calculator(self, Number):
        i, j = 1, 0
        
        while i > 0:
            if (Number > 1) and (Number < 10):
                i = 0
            else:
                Number *= 10
                j += 1
                
        return j
    
    