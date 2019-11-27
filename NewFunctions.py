import Lightcurves as Lifetimes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs
from scipy import optimize

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



def takeSecond(elem):
    return elem[1]

    

def FindCorr(Coeffs, k, tscale, line = True):

    size = (tscale - 100)/len(Coeffs)
    
    spearlist = []
    difflist = []
    meanlist = []
    maxlist = []
    minlist = []
    rankmax = []
    rankmin = []
    ranks = []

    for i in range(len(Coeffs)):
        spearlist.append(Coeffs[i])

    #for i in range(len(spearlist) - 1):
    #    difflist.append(abs(spearlist[i + 1] - spearlist[i]))
    
    for i in range(len(spearlist) - (k + 1)):
        #meanlist.append([np.mean(difflist[:i + k]), np.mean(difflist[i + k:]), i + k + 1])
        meanlist.append([scs.linregress(Coeffs[i:i + k])[0], scs.linregress(Coeffs[i + k:i + 2*k])[0], i + k + 1])
        
    _df = pd.DataFrame(meanlist, columns = ['Mean Below Point', 'Mean Above Point', 'Point'])
    _df.dropna(thresh = 3, inplace = True)
    max_val = _df.max(axis = 0)[0]
    min_val = _df.min(axis = 0)[1]
    indmax = _df.iloc[:, 0][_df.iloc[:, 0] == max_val].index.tolist()[0]
    indmin = _df.iloc[:, 1][_df.iloc[:, 1] == min_val].index.tolist()[0]
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
    forcalc_x = []
    forcalc_y = []

    for i in range(100):
        forcalc_y.append(0.01*(i + 1))
        forcalc_x.append(Lifetimes.MyCosmology.cosmocal(0.01*(i + 1))['ageAtZ'])
    
    curve = optimize.curve_fit(lambda t,a,b,c: a*t**-b + c,  forcalc_x,  forcalc_y, p0 = (2, 1,0.5))
    
    return curve[0][0]*Time**-curve[0][1] + curve[0][2]



def AGN(AGN_Type, AoU, LognormSFHs, BurstHeights, BurstWidth = 2e8, Fmax = 10, DownTime = 5, t_delay = 2e8):
    
    AGNs = ['Probabilistic', 'Delay', 'Random']
    if AGN_Type not in AGNs:
        raise Exception("ERROR: AGN_Type should either be \'Probabilistic\', \'Delay\' or \'Random\'!")
    
    Time = AoU #raw time data from bagpipes
    bheights = BurstHeights
    T2 = Time[Time > 0][::-1] #ensuring time is non-negative, and increasing
    bagpipes_df = pd.DataFrame({'Time':T2})
    Column_Names = {0:'Universe Time'}
    TriggerTimes = []
    
    if AGN_Type == 'Probabilistic':
        for i in range(len(LognormSFHs)): #iterate through all starbursts
            _df = pd.DataFrame() #to handle data
            _df['Time'], _df['Flux'] = T2, LognormSFHs[i][Time > 0.0][::-1]
            CutOff = _df[_df['Flux'] > 3e-4].iloc[0, :].name #cutoff stops unphysical agn activity occuring
            
            _df = _df.iloc[CutOff:, :]
            step = (_df['Time'].iloc[-1] - _df['Time'].iloc[0])/len(_df['Time'])
            T = np.arange(_df['Time'].iloc[0], max(_df['Time']), step) 
            #reproduce time data in non-log format so agn are not distorted
        
            if len(T) > len(_df['Flux']):
                T = T[:-1]

            _t, _agn, _triggertimes = Lifetimes.probabilistic_lognorm(np.array(_df['Time']), np.array(_df['Flux']), bheights[i], BurstWidth, f_max = max(_df['Flux'])*Fmax, downtime = DownTime, randomheight = False, ttype = 2, Time = T)
        
            TriggerTimes.append(_triggertimes)
            _df2 = pd.DataFrame({'AGN Time': T, 'AGN Flux': _agn}).set_index(_df['Flux'].index)
            bagpipes_df = pd.concat([bagpipes_df, _df['Flux'], _df2], ignore_index = True, axis = 1)
    
            Column_Names[1 + i*3] = 'SFH {}'.format(i) #SFH history same as universal time ('LOG SCALE')
            Column_Names[2 + i*3] = 'AGN Time {}'.format(i) #AGN time NOT LOGSCALE, understand the point T (row 12)
            Column_Names[3 + i*3] = 'AGN Flux {}'.format(i) #Hence AGN time needed, but not SFH time

        bagpipes_df.rename(columns = Column_Names, inplace = True)
        
        return bagpipes_df, TriggerTimes
        
    if AGN_Type == 'Delay':
        for i in range(len(LognormSFHs)):
            _t, _f = Lifetimes.delay(T2, LognormSFHs[i][Time > 0.0][::-1], t_delay = t_delay)
            _df = pd.DataFrame(data = {'SFH':LognormSFHs[i][Time > 0.0][::-1], 'AGN Time':_t + t_delay, 'AGN Rate':_f})
            bagpipes_df = pd.concat([bagpipes_df, _df], ignore_index = True, axis = 1)
            Column_Names[1 + i*3] = 'SFH {}'.format(i)
            Column_Names[2 + i*3] = 'Delayed AGN Time {}'.format(i)
            Column_Names[3 + i*3] = 'Delayed AGN AR {}'.format(i)
            
        bagpipes_df.rename(columns = Column_Names, inplace = True)
        
        return bagpipes_df
        
        
        
        
        
        
        
        
        
        