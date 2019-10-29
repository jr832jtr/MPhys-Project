import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs

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
        
        data.groupby(data['Groups']).mean().plot(x = 'Time', y = list(np.arange(0,ngals,1)), ax = axez[0, 0], marker = '.', linestyle = '', logy = True, legend = False, style = ['blue', 'green']*(ngals/2), title = 'Average Flux at {} bins for {}'.format(bins, name))

    temp_df = data.groupby(data['Groups']).mean()
    temp_df[temp_df > 0] = 1
    temp_df = temp_df.sum(axis = 1) - 1
    
    if Return:
        Max = temp_df.max()
        Ind = temp_df[temp_df == Max].index.tolist()
        T = round(data.groupby(data['Groups']).mean()['Time'][Ind[0]], -6)
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
