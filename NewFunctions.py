import Lightcurves as Lifetimes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs
import bagpipes
from scipy.integrate import simps
from scipy.interpolate import interp1d
from scipy import optimize
from astropy.cosmology import FlatLambdaCDM

def plot_marker(point, index, c, SubPlots, Graph = True):
    
    if point == 0:
        return None
    if Graph == False:
        SubPlots[index].axvline(x = point, color = c)
    else:
        plt.axvline(x = point, color = c)
        
def delt(ngals, n, dn, dataset, corr, deltat = 1e6, tmax = 1e9, Print = True, log_y = True, FluxLog = True): #provides delta T plots and pearson coefficients
    
    if type(n) != int: #Exceptions, to avoid user input errors
        raise Exception('ERROR: n must be integer type')    
        
    if n > tmax/deltat:
        str1 = 'ERROR: n must be between 0 and {}'.format(int(tmax/deltat))
        raise Exception(str1)
        
    if n > dn:
        raise Exception('ERROR: dn must be greater n')
        
    if dn > tmax/deltat:
        str2 = 'ERROR: dn must be between 0 and {}'.format(int(tmax/deltat))
        raise Exception(str2)
    
    _datasetdf = pd.DataFrame(dataset['lc_agn']).T
    _datasetdf['Time'] = dataset['t']
        
    deltaT = (dataset['t'][n:(dn + 1)] - (deltat*n)) #slicing is n:dn + 1 not inclusive! 1:100 gives 1 to 99
    deltaF = (dataset['lc_agn'][:, n:(dn + 1)]) 

    if FluxLog == True:
        resultF = np.log10(deltaF[:, (dn - n)][deltaF[:, (dn - n)] != 0])
        resultSF = np.log10(dataset['lc_sfr'][:, dn][deltaF[:, (dn - n)] != 0])
    elif FluxLog == False:
        resultF = deltaF[:, (dn - n)][deltaF[:, (dn - n)] != 0]
        resultSF = dataset['lc_sfr'][:, dn][deltaF[:, (dn - n)] != 0]
        
    if Print:
        
        fig, axs = plt.subplots(1, 3, figsize = (15, 5))
        _df = pd.DataFrame(deltaF.T)
        _df['Time'] = deltaT
    
        _df.plot(x = 'Time', y = list(np.arange(0,ngals,1)), legend = False, linestyle = '', marker = '.', markersize = 2, logy = log_y, ax = axs[0], style = ['red', 'blue', 'green']*int(ngals/3))
        axs[0].set_xlabel('Delta T')
        axs[0].set_ylabel('Flux')
    
        _datasetdf.plot(x = 'Time', y = list(np.arange(0,ngals,1)), legend = False, linestyle = '', marker = '.', markersize = 2, logy = log_y, ax = axs[1], style = ['blue', 'red', 'green']*int(ngals/3))
        axs[1].set_ylabel('Flux')
    
        plt.subplot(1, 3, 3).scatter(x = [dataset['t'][dn]]*len(deltaF[:, (dn - n)]), y = deltaF[:, (dn - n)], color = 'black')
        plt.xlim(0, tmax)
        plt.ylim(axs[1].get_ylim())
        plt.gca().set_yscale((lambda x: 'log' if x else 'linear')(log_y))
        plt.gca().set_xlabel('Time')
        plt.gca().set_ylabel('Flux')
    
        axs[1].axvline(x = n*deltat, color = 'r')
        axs[1].axvline(x = dn*deltat, color = 'k')
    
        plt.subplots_adjust(wspace = 0.3)

        plt.show()
    
    if corr == 'Pearson':
        return [scs.pearsonr(np.ravel(dataset['lc_agn'][:, n:(dn+1)]), np.ravel(dataset['lc_sfr'][:, n:(dn+1)])), scs.pearsonr(resultF, resultSF), resultF, resultSF]
    elif corr == 'Spearman':
        return [scs.spearmanr(np.ravel(dataset['lc_agn'][:, n:(dn+1)]), np.ravel(dataset['lc_sfr'][:, n:(dn+1)])), scs.spearmanr(resultF, resultSF), resultF, resultSF]


def Average(bins, data, ngals, name, log_y, savefigure = False, Return = False):

    temp_lis = []
    for i in range(bins):
        temp_lis.append([i]*int(1000/bins))
    
    data[data == 0] = np.nan
    data['Groups'] = np.ravel(temp_lis)
    
    if not Return:
        fig, axez = plt.subplots(2, 2, figsize = (14, 10), sharex = True)
        
        data.groupby(data['Groups']).mean().plot(x = 'Time', y = list(np.arange(0,ngals,1)), ax = axez[0, 0], marker = '.', linestyle = '', logy = True, legend = False, style = ['blue', 'green']*int(ngals/2), title = 'Average Flux at {} bins for {}'.format(bins, name))

    temp_df = data.groupby(data['Groups']).mean()
    temp_df[temp_df > 0] = 1
    temp_df = temp_df.sum(axis = 1) - 1
    
    if Return:
        AVals = data.groupby(data['Groups']).mean().drop('Time', axis = 1).mean(axis = 1)
        Max = AVals[:9].max()#Max = temp_df.max()
        Ind = AVals[AVals == Max].index.tolist()#Ind = temp_df[temp_df == Max].index.tolist()
        T = round(data.groupby(data['Groups']).mean()['Time'][Ind[0]], -6)#T = round(data.groupby(data['Groups']).mean()['Time'][Ind[0]], -6)
        return T

    axez[1, 0].scatter(x = data.groupby(data['Groups']).mean()['Time'], y = temp_df, s = 24, marker = 'x', c = 'r');
    axez[0, 1].plot(data.groupby(data['Groups']).mean()['Time'], data.groupby(data['Groups']).mean().drop('Time', axis = 1).mean(axis = 1));
    
    data[data.isna()] = 0.0
    axez[1, 1].plot(data.groupby(data['Groups']).mean()['Time'], data.groupby(data['Groups']).mean().drop('Time', axis = 1).mean(axis = 1));
    
    axez[1, 0].set_title('Number of AGN with non-zero average flux for {}'.format(name));
    axez[1, 0].set_xlabel('Time');
    axez[1, 0].set_ylabel('Count');
    axez[0, 0].set_ylabel('Average Flux');
              
    axez[0, 1].set_title('Average of the Averages: zeroes excluded')
    axez[0, 1].set_xlabel('Time')
    axez[0, 1].set_ylabel('Average Flux')
    axez[0, 1].set_yscale((lambda x: 'log' if x else 'linear')(log_y))
    
    axez[1, 1].set_title('Average of the Averages: zeroes included')
    axez[1, 1].set_xlabel('Time')
    axez[1, 1].set_ylabel('Average Flux')
    axez[1, 1].set_yscale((lambda x: 'log' if x else 'linear')(log_y))
        
    plt.subplots_adjust(hspace = 0.2)
        
    if savefigure:
        plt.savefig('AvgFlux{}.png'.format(name.split(' ')[0]))
        
    plt.show()
                        
    return None      

    

def FindCorr(Coeffs, k, tscale, line = True):

    size = (tscale - 100)/len(Coeffs)
    
    spearlist = []
    meanlist = []

    for i in range(len(Coeffs)):
        spearlist.append(Coeffs[i])
    
    for i in range(len(spearlist) - (k + 1)):
        meanlist.append([scs.linregress(Coeffs[i:i + k])[0], scs.linregress(Coeffs[i + k:i + 2*k])[0], i + k + 1])
        
    _df = pd.DataFrame(meanlist, columns = ['Mean Below Point', 'Mean Above Point', 'Point'])
    _df.dropna(thresh = 3, inplace = True)
    _df = _df.sort_values(by = 'Mean Below Point', ascending = False).reset_index().drop('index', axis = 1)
    _df['Rank Max'] = _df.index + 1
    _df = _df.sort_values(by = 'Mean Above Point').reset_index().drop('index', axis = 1)
    _df['Rank Min'] = _df.index + 1
    _df['Rank Total'] = abs(_df['Rank Max'] + _df['Rank Min'])
    _df = _df.sort_values(by = 'Rank Total')
    
    vline = _df.iloc[0, 2]*size*1e7
    
    if line:
        return vline
    else:
        return _df
    
    
    
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
    T2 = Time[Time > 0][::-1] #ensuring time is non-negative, and increasing
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
        
        
def AGN_Periods(AGN_Type, SFHs_df, TriggerTimes, num, thresh, Scale, AoU, show = False):
    
    if AGN_Type == 'Delay':
        ind = SFHs_df[SFHs_df['AGN AR {}'.format(num)] > max(SFHs_df['AGN AR {}'.format(num)])/100].index[0]
        SFHs_df = SFHs_df.iloc[ind:, :]
        return [np.array(SFHs_df['AGN Time {}'.format(num)])]
    
    
    
    AGN_periods = []
    AGN_ends= []
    indstarts, indends = [], []
    indon, indoff = [], []
    diff_filter = []
    Start = []
    j = 0
    
    #find all points which lie above some thresh val. allows to identify separate periods of agn activity
    num_arr = np.where((max(SFHs_df['AGN AR {}'.format(num)].dropna())/thresh) < 
                       SFHs_df['AGN AR {}'.format(num)].dropna())[0]
    
    if AGN_Type == 'Random':
        if show:
            fig = plt.figure()
            ax = plt.subplot()
            
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
                ax.axvline(TriggerTimes[num][i])
                indon.append(ind)
                
        for i in range(len(num_arr) - 1):
            if (num_arr[i + 1] - num_arr[i]) > 1 or (num_arr[i] == num_arr[-2]):
                ind1 = SFHs_df['AGN AR {}'.format(num)].dropna().index[0] + num_arr[i]
                indoff.append(ind1)
                AGN_ends.append(SFHs_df.iloc[ind1, 2 + 3*num])
                ax.axvline(SFHs_df.iloc[ind1, 2 + 3*num])
                
        for i in range(len(Start)):
            diff_list1 = []
            diff_list2 = []
            for j in range(len(SFHs_df['Universe Time'])):
                diff_list1.append(abs(SFHs_df.iloc[j, 0] - Start[i]))
                diff_list2.append(abs(SFHs_df.iloc[j, 0] - AGN_ends[i]))
            indstarts.append(diff_list1.index(min(diff_list1)))
            indends.append(diff_list2.index(min(diff_list2)))
            
        indstarts.sort()
        indends.sort()
        indon.sort()
        indoff.sort()
        
        for i in range(len(indstarts)):
            AGN_periods.append(list(SFHs_df.iloc[indon[i]:indoff[i], 2 + 3*num]))
            
        
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
                
                #finds number of steps between agn on and off using bagpipes source code- POSSIBLY UNNECESSARY
                AGN_Start = SFHs_df['Universe Time'][np.array(SFHs_df['Universe Time']) 
                                                     == TriggerTimes[num][j]].index[0]
                AGN_End = AGN_Start + int((np.log10(AoU - SFHs_df.iloc[ind, 2 + 3*num]) 
                                           - np.log10(AoU - TriggerTimes[num][j]))/-0.0025) 
                #AGN_periods.append(list(SFHs_df.iloc[AGN_Start:AGN_End, 0]))
                AGN_periods.append(list(SFHs_df.iloc[Start[j]:ind, 2 + 3*num]))
            
                if show:
                    ax.axvline(x = SFHs_df.iloc[AGN_Start:AGN_End, 0].iloc[0])
                    ax.axvline(x = SFHs_df.iloc[AGN_Start:AGN_End, 0].iloc[-1])

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
        ind = len(np.unique(binlist))
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
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
