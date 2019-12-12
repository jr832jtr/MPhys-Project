#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:44:32 2019

@author: jr832
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..Analysis.Setup import Lightcurves as Lifetimes
from astropy.cosmology import FlatLambdaCDM
from scipy import optimize


def Z_Calc(Time):
    cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
    forcalc_x = []
    forcalc_y = []

    for i in range(100):
        forcalc_y.append(0.01*(i + 1))
        forcalc_x.append(cosmo.age(0.01*(i + 1)).value)
    
    curve = optimize.curve_fit(lambda t,a,b,c: a*t**-b + c,  forcalc_x,  forcalc_y, p0 = (2, 1,0.5))
    
    return curve[0][0]*Time**-curve[0][1] + curve[0][2]

def Generate_AGN(AoU, LognormSFHs, BurstHeights, AGN_Params):
    
    AGNs = ['Probabilistic', 'Delay', 'Random']
    if AGN_Params['AGN_Type'] not in AGNs:
        raise Exception("ERROR: AGN_Type should either be \'Probabilistic\', \'Delay\' or \'Random\'!")
    
    Time = AoU #raw time data from bagpipes
    bheights = BurstHeights
    T2 = Time[Time > 0][::-1] #ensuring time is non-negative and increasing
    bagpipes_df = pd.DataFrame({'Time':T2})
    Column_Names = {0:'Universe Time'}
    TriggerTimes = []
    
    if AGN_Params['AGN_Type'] == 'Probabilistic':
        for i in range(len(LognormSFHs)): #iterate through all starbursts
            _df = pd.DataFrame() #to handle data
            _df['Time'], _df['Flux'] = T2, LognormSFHs[i][Time > 0.0][::-1]
            CutOff = _df[_df['Flux'] > max(_df['Flux'])/1e4].iloc[0, :].name #cutoff stops unphysical agn activity occuring
            
            _df = _df.iloc[CutOff:, :]
            step = (_df['Time'].iloc[-1] - _df['Time'].iloc[0])/len(_df['Time'])
            T = np.arange(_df['Time'].iloc[0], max(_df['Time']), step) 
            #reproduce time data in non-log format so agn are not distorted
        
            if len(T) > len(_df['Flux']):
                T = T[:-1] #avoids mismatches in length due to rounding errors of np.arange. No harm done as bagpipes time is the dictator here. 

            _t, _agn, _triggertimes = Lifetimes.probabilistic_lognorm(np.array(_df['Time']), 
                                                                      np.array(_df['Flux']), bheights[i], 
                                                                      AGN_Params['BurstWidth'], 
                                                                      f_max = max(_df['Flux'])*AGN_Params['Fmax'], 
                                                                      downtime = AGN_Params['DownTime'], randomheight = False, 
                                                                      ttype = 2, Time = T)
        
            TriggerTimes.append(_triggertimes)
            _df2 = pd.DataFrame({'AGN Time': T, 'AGN Flux': _agn}).set_index(_df['Flux'].index)
            bagpipes_df = pd.concat([bagpipes_df, _df['Flux'], _df2], ignore_index = True, axis = 1)
    
            Column_Names[1 + i*3] = 'SFH {}'.format(i) #SFH history same as universal time ('LOG SCALE')
            Column_Names[2 + i*3] = 'AGN Time {}'.format(i) #AGN time NOT LOGSCALE, understand the point T (row 12)
            Column_Names[3 + i*3] = 'AGN AR {}'.format(i) #Hence AGN time needed, but not SFH time

        bagpipes_df.rename(columns = Column_Names, inplace = True)
        
        return bagpipes_df, TriggerTimes
        
    if AGN_Params['AGN_Type'] == 'Delay':
        for i in range(len(LognormSFHs)):
            _t, _f = Lifetimes.delay(T2, LognormSFHs[i][Time > 0.0][::-1], t_delay = AGN_Params['Delay'], scale = bheights[i])
            _df = pd.DataFrame(data = {'SFH':LognormSFHs[i][Time > 0.0][::-1], 
                                       'AGN Time':_t + AGN_Params['Delay'], 'AGN Rate':_f})
            bagpipes_df = pd.concat([bagpipes_df, _df], ignore_index = True, axis = 1)
            Column_Names[1 + i*3] = 'SFH {}'.format(i)
            Column_Names[2 + i*3] = 'AGN Time {}'.format(i)
            Column_Names[3 + i*3] = 'AGN AR {}'.format(i)
            
        bagpipes_df.rename(columns = Column_Names, inplace = True)
        
        return bagpipes_df
    
    if AGN_Params['AGN_Type'] == 'Random':
        for i in range(len(LognormSFHs)):
            _df = pd.DataFrame()
            _df['Time'], _df['Flux'] = T2, LognormSFHs[i][Time > 0.0][::-1]
            CutOff = _df[_df['Time'] > (AGN_Params['TimesMasses'][i]['BurstTime']*10**9) 
                         - AGN_Params['Delta T Min']].iloc[0, :].name #selects range for random bursts to occur
            _df = _df.iloc[CutOff:, :]
            
            _agn, _triggertimes = Lifetimes.random_burst(max(np.array(_df['Time'])), AGN_Params['Delta T'], 
                                                                 AGN_Params['DownTime'], AGN_Params['BurstLength'],
                                                                 bheights[i], bursttype = 'lognorm', 
                                                                 t_min = (AGN_Params['TimesMasses'][i]['BurstTime']*10**9)
                                                                 - AGN_Params['Delta T Min'], BurstTimes=True)
            TriggerTimes.append(_triggertimes)
            _df2 = pd.DataFrame({'AGN Time': _agn[0], 'AGN Flux': _agn[1]})
            bagpipes_df = pd.concat([bagpipes_df, _df['Flux'], _df2], ignore_index = True, axis = 1)
            
            Column_Names[1 + i*3] = 'SFH {}'.format(i)
            Column_Names[2 + i*3] = 'AGN Time {}'.format(i)
            Column_Names[3 + i*3] = 'AGN AR {}'.format(i)
            
        bagpipes_df.rename(columns = Column_Names, inplace = True)
        
        return bagpipes_df, TriggerTimes
        
        
def AGN_Periods(AGN_Type, SFHs_df, TriggerTimes, num, thresh, Scale, AoU, Res, show = False):
    
    TimeStep = AoU/Res
    
    if AGN_Type == 'Delay':
        
        
        ind = SFHs_df[SFHs_df['AGN AR {}'.format(num)] > max(SFHs_df['AGN AR {}'.format(num)])/100].index[0]
        SFHs_df = SFHs_df.iloc[ind:, :]
        
        if show:
            fig = plt.figure()
            ax = plt.subplot()
            ax.plot(SFHs_df['Universe Time'], SFHs_df['SFH {}'.format(num)], 
                SFHs_df['AGN Time {}'.format(num)], SFHs_df['AGN AR {}'.format(num)]/Scale)
            plt.show()
            
        AGN_on = min(SFHs_df['AGN Time {}'.format(num)].dropna())
        AGN_off = max(SFHs_df['AGN Time {}'.format(num)].dropna())
        No_Points = int((AGN_off - AGN_on)/TimeStep)
        DeltaT = (AGN_off - AGN_on)/No_Points
        
        Burst = np.arange(AGN_on, AGN_off, DeltaT)
        
        return [Burst]
    
    AGN_periods = []
    AGN_ends= []
    indstarts = []
    indon, indoff = [], []
    diff_filter = []
    Start = []
    j = 0
    
    #find all points which lie above some thresh val. allows to identify separate periods of agn activity
    num_arr = np.where((max(SFHs_df['AGN AR {}'.format(num)].dropna())/thresh) < 
                       SFHs_df['AGN AR {}'.format(num)].dropna())[0]
    
    if AGN_Type == 'Random':
        fig = plt.figure()
        ax = plt.subplot()
        
        if show:            
            ax.plot(SFHs_df['Universe Time'], SFHs_df['SFH {}'.format(num)], 
                    SFHs_df['AGN Time {}'.format(num)], 
                    SFHs_df['AGN AR {}'.format(num)]/Scale)
            ax.axhline(y = max(SFHs_df['AGN AR {}'.format(num)].dropna())/(thresh*Scale))
            
        for i in range(len(TriggerTimes[num])):
            diff_filter = []
            for j in range(len(SFHs_df.iloc[:, 2 + 3*num])):
                diff_filter.append(abs(SFHs_df.iloc[j, 2 + 3*num] - TriggerTimes[num][i]))
            ind = diff_filter.index(min(diff_filter))
            if (SFHs_df.iloc[ind, 3 + 3*num]) < (max(SFHs_df['AGN AR {}'.format(num)].dropna())/thresh):
                Start.append(TriggerTimes[num][i])
                indon.append(ind)
                if show:
                    ax.axvline(TriggerTimes[num][i])
                
        for i in range(len(num_arr) - 1):
            if (num_arr[i + 1] - num_arr[i]) > 1 or (num_arr[i] == num_arr[-2]):
                ind1 = SFHs_df['AGN AR {}'.format(num)].dropna().index[0] + num_arr[i]
                indoff.append(ind1)
                AGN_ends.append(SFHs_df.iloc[ind1, 2 + 3*num])
                if show:
                    ax.axvline(SFHs_df.iloc[ind1, 2 + 3*num])
                
        for i in range(len(Start)):
            diff_list1 = []
            for j in range(len(SFHs_df['Universe Time'])):
                diff_list1.append(abs(SFHs_df.iloc[j, 0] - Start[i]))
            indstarts.append(diff_list1.index(min(diff_list1)))

        indon.sort()
        indoff.sort()
        
        for i in range(len(indstarts)):
            AGN_on = SFHs_df.iloc[indon[i], 2 + 3*num]
            AGN_off = SFHs_df.iloc[indoff[i], 2 + 3*num]
            No_Points = int((AGN_off - AGN_on)/TimeStep)
            DeltaT = (AGN_off - AGN_on)/No_Points
            Bursts = np.arange(AGN_on, AGN_off, DeltaT)
            AGN_periods.append(Bursts)
                
            #AGN_periods.append(list(SFHs_df.iloc[indon[i]:indoff[i], 2 + 3*num]))
            
        
    if AGN_Type == 'Probabilistic':
        if show:
            fig = plt.figure()
            ax = plt.subplot()
        
            ax.plot(SFHs_df['Universe Time'], SFHs_df['SFH {}'.format(num)], 
                    SFHs_df['AGN Time {}'.format(num)], 
                    SFHs_df['AGN AR {}'.format(num)]/Scale)
            ax.axhline(y = max(SFHs_df['AGN AR {}'.format(num)].dropna())/(thresh*Scale))
            
        for i in range(len(TriggerTimes[num])):
            diff_list = []
            for j in range(len(SFHs_df['AGN Time {}'.format(num)])):
                diff_list.append(abs(SFHs_df.iloc[j, 2 + 3*num] - TriggerTimes[num][i]))
            AGNon = diff_list.index(np.nanmin(diff_list))
            Start.append(AGNon)
            
        j = 0
        
        for i in range(len(num_arr) - 1): #find indices of points where agn activity 'dies'
            if (num_arr[i + 1] - num_arr[i]) > 1 or (num_arr[i] == num_arr[-2]):
                ind = SFHs_df.iloc[:, 2 + 3*num].dropna().index[0] + num_arr[i] #rectifies the index displacement caused 'np.where'
                AGN_on = SFHs_df.iloc[Start[j], 2 + 3*num]
                AGN_off = SFHs_df.iloc[ind, 2 + 3*num]
                No_Points = int((AGN_off - AGN_on)/TimeStep)
                DeltaT = (AGN_off - AGN_on)/No_Points
                Bursts = np.arange(AGN_on, AGN_off, DeltaT)
                AGN_periods.append(Bursts)
                
                #finds number of steps between agn on and off using bagpipes source code- POSSIBLY UNNECESSARY
                #AGN_Start = SFHs_df['Universe Time'][np.array(SFHs_df['Universe Time']) 
                #                                     == TriggerTimes[num][j]].index[0]
                #AGN_End = AGN_Start + int((np.log10(AoU - SFHs_df.iloc[ind, 2 + 3*num]) 
                #                           - np.log10(AoU - TriggerTimes[num][j]))/-0.0025) 
                #AGN_periods.append(list(SFHs_df.iloc[AGN_Start:AGN_End, 0]))
                #AGN_periods.append(list(SFHs_df.iloc[Start[j]:ind, 2 + 3*num]))
            
                if show:
                    #ax.axvline(x = SFHs_df.iloc[AGN_Start:AGN_End, 0].iloc[0])
                    #ax.axvline(x = SFHs_df.iloc[AGN_Start:AGN_End, 0].iloc[-1])
                    ax.axvline(x = SFHs_df.iloc[Start[j]:ind, 2 + 3*num].iloc[0])
                    ax.axvline(x = SFHs_df.iloc[Start[j]:ind, 2 + 3*num].iloc[-1])

                j += 1
            
                try:
                    if SFHs_df.iloc[ind, 2 + 3*num] > TriggerTimes[num][j]:
                        j += 1
                except IndexError:
                    pass

    if show:
        plt.show()
        return AGN_periods
    else:
        return AGN_periods