#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:26:20 2019

@author: jr832
"""

import numpy as np
import matplotlib.pyplot as plt
import bagpipes
from . import AGN
from . import SF

class Active_Galaxy():
    
    def __init__(self, agnparams, Sim, gal_params, obs_wavs, gal_no, fraction = 0.5, Thresh = 25, Scaling = 1e44, Show_Lc = False):
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
        self.Spectra = self.Generate_Spectra(gal_no, Thresh = Thresh, Scaling = Scaling, show = Show_Lc)
        
        
        
    def SFHs(self, Obs_Wavs, Sim, Fraction, Params):
        return SF.Generate_SFHs(Obs_Wavs, Sim, Fraction, Params)
    
    
    
    def Generate_Spectra(self, Number, Thresh, Scaling, show):
        AGN_ON_Spectras = []
        AGN_ON = AGN.AGN_Periods(self.Agn_Type, self.Active_Galaxies, self.TriggerTimes, Number, Thresh, Scaling, self.Galaxies['Galaxy'].sfh.age_of_universe, show = show)

        for i in range(len(AGN_ON)):
            for j in range(len(AGN_ON[i])):
                _Z = AGN.Z_Calc(AGN_ON[i][j]*10**-9)
                model_components = {}
                model_components["redshift"] = _Z

                try:
                    mass_frac = SF.Mass_Calculator(self.Gal_Params['redshift'], AGN_ON[i][j], self.Galaxies, Number, 100, T_min = AGN_ON[0][0] - 4e8)
                except ValueError:
                    break

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
        
                if mass_frac != 0:
                    lognormal['massformed'] = np.log10(mass_frac*10**self.Galaxies['TimesMasses'][Number]['BurstMass'])
                    model_components['lognormal'] = lognormal
            
                dblplaw['massformed'] = self.Galaxies['TimesMasses'][Number]['PwrLawMass']
                model_components['dblplaw'] = dblplaw
                _galaxy = bagpipes.model_galaxy(model_components, filt_list=self.Filters, spec_wavs=self.Obs_Wavs)
                AGN_ON_Spectras.append(_galaxy.spectrum)
                
                if j == 0:
                    _galaxy.sfh.plot()
                
        return AGN_ON_Spectras
                
                
    def Plot_Average_Spectra(self):
        Spectra_Sum = sum(self.Spectra)
        Spectra_Average = Spectra_Sum/len(self.Spectra)

        fig = plt.figure(figsize = (12, 4))
        ax = plt.subplot()
        x, y = zip(*Spectra_Average)
        
        scale = self.Decimal_Calculator(max(y))
        
        y = np.array(y)/10**-scale
        
        plt.plot(x, y)
        ax.set_ylabel("$\\mathrm{f_{\\lambda}}\\ \\mathrm{/\\ 10^{"+ str(-scale)+ "}\\ erg\\ s^{-1}\\ cm^{-2}\\ \\AA^{-1}}$")
        ax.set_xlabel("$\\lambda / \\mathrm{\\AA}$")
        plt.show()
        
        
        
    def Plot_Summed_Spectra(self):
        Spectra_Sum = sum(self.Spectra)
        
        fig = plt.figure(figsize = (12, 4))
        ax = plt.subplot()
        x, y = zip(*Spectra_Sum)
        
        scale = self.Decimal_Calculator(max(y))
        
        y = np.array(y)/10**-scale
        x = np.array(x)/len(self.Spectra)
        
        plt.plot(x, y)
        ax.set_ylabel("$\\mathrm{f_{\\lambda}}\\ \\mathrm{/\\ 10^{"+ str(-scale)+ "}\\ erg\\ s^{-1}\\ cm^{-2}\\ \\AA^{-1}}$")
        ax.set_xlabel("$\\lambda / \\mathrm{\\AA}$")
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
    
    