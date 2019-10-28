import numpy as np
import pandas as pd
import scipy.stats as scs
import Lightcurves as Lifetimes
import matplotlib.pyplot as plt
import NewFunctions

class AGNSFR:
    
    def __init__(self, agntype, agnlcpars, lcpars, no_gals, sbscale, name):
        self.agntype = agntype
        self.agnlcpars = agnlcpars
        self.no_gals = no_gals
        self.sbscale = sbscale
        self.lcpars = lcpars
        self.name = name
        self.data = Lifetimes.simu_lx_sfr(no_gals, sbscale = sbscale, lcpars = lcpars, agnlcpars = agnlcpars, cannedgalaxies = False, agnlctype = agntype)
        
        
        
    def msigma(self):
        SFRs = self.data['m_gal']
        AGNs = self.data['bhmass']

        fig = plt.figure() #create figure objectburstwidth

        plt.scatter(SFRs, AGNs, s = 2)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Galaxy Mass');
        plt.ylabel('BH Mass');

        return scs.pearsonr(np.log10(SFRs), np.log10(AGNs))
    
        
        
    def Coefficients(self, coefftype, LogBool, no_vals, plot = True):
        
        fig = plt.figure()
        coeffs = []
        
        if self.agntype == 'delay':
            start = int((no_vals*2)/9) + 1
        else:
            start = 1
        
        for i in range(start, no_vals, 1):
            coeffs.append((int(900/no_vals)*i, NewFunctions.delt(self.no_gals, 100, 100 + (int(900/no_vals)*i), self.data, corr = coefftype, Print = False, FluxLog = LogBool)[1][0]))

        coeff_arr = np.reshape(np.array(np.ravel(coeffs)), (len(range(start, no_vals, 1)), 2))
        
        if self.agntype == 'delay':
            Coeff = ['N/A']
        else:
            Coeff = (lambda x: scs.spearmanr(coeff_arr[:, 0], coeff_arr[:, 1]) if x == 'Spearman' else scs.pearsonsr(coeff_arr[:, 0], coeff_arr[:, 1]))(coefftype)
        
        if plot:
            plt.scatter(*zip(*coeffs))
            plt.gca().set_xlabel('Delta T')
            plt.gca().set_ylabel('{} Coefficient'.format(coefftype))
            plt.gca().set_title('{}, {}: {}'.format(self.name, coefftype, Coeff[0]))
            plt.gca().set_xlim(0, 900)
        
        return coeffs, Coeff
    
    
    
    def SimPlot(self, plot = True):
        
        fig = plt.figure()
        
        df = pd.DataFrame(self.data['lc_agn']).T
        df['Time'] = self.data['t']
        
        if plot:
            df.plot(x = 'Time', y = list(np.arange(0,self.no_gals,1)), logy = True, legend = False, markersize = 2, marker = '.', linestyle = '', style = ['red', 'blue', 'green']*int(self.no_gals/3), title = self.name)
            plt.gca().set_ylabel('Flux')
        
        return df
    
    
    
    def Count(self, bins, plot = True):
        
        data = self.SimPlot(plot = False)
        
        temp_lis = []
        for i in range(bins):
            temp_lis.append([i]*int(1000/bins))
    
        data[data == 0] = np.nan
        data['Groups'] = np.ravel(temp_lis)
        
        temp_df = data.groupby(data['Groups']).mean()
        temp_df[temp_df > 0] = 1
        temp_df = temp_df.sum(axis = 1) - 1
        
        
        if plot:
            plt.scatter(x = data.groupby(data['Groups']).mean()['Time'], y = temp_df, s = 24, marker = 'x', c = 'r');
            plt.gca().set_xlabel('Time')
            plt.gca().set_ylabel('Count')
            plt.gca().set_title('No. of AGN with non-zero observed Flux')    
    
        return temp_df, data
    
    
    
    def AllPlots(self, coefftype, LogBool, no_vals, bins):
        
        coeffs, Coeff = self.Coefficients(coefftype, LogBool, no_vals, plot = False)
        df = self.SimPlot(plot = False)
        count, data = self.Count(bins, plot = False)
        
        fig, axs = plt.subplots(1, 3, figsize = (15, 4))
        
        df = pd.DataFrame(self.data['lc_agn']).T
        df['Time'] = self.data['t']
        df.plot(x = 'Time', y = list(np.arange(0,self.no_gals,1)), ax = axs[0], logy = True, legend = False, markersize = 2, marker = '.', linestyle = '', style = ['red', 'blue', 'green']*int(self.no_gals/3), title = self.name)
        axs[0].set_ylabel('Flux')

        axs[1].scatter(*zip(*coeffs))
        axs[1].set_xlabel('Delta T')
        axs[1].set_ylabel('{} Coefficient'.format(coefftype))
        axs[1].set_title('{}, {}: {}'.format(self.name, coefftype, Coeff[0]))
        axs[1].set_ylim(0, 1)
        axs[1].set_xlim(0, 900)
        
        axs[2].scatter(x = data.groupby(data['Groups']).mean()['Time'], y = count, s = 24, marker = 'x', c = 'r');
        axs[2].set_xlabel('Time')
        axs[2].set_ylabel('Count')
        axs[2].set_title('No. of AGN with non-zero observed Flux')    
    
    
    
    def GridPlot(self, size, Corr, LogBool):
        
        fig, axs = plt.subplots(size[0], size[1], sharex = 'col', sharey = 'row', gridspec_kw={'hspace': 0, 'wspace': 0}, figsize = (12,10))
        
        a = max(size)
        b = min(size)
        
        for i in range(size[0]*size[1]):
            
            if a == 1 or b == 1:
                axs[i].scatter(x = NewFunctions.delt(self.no_gals, 100, 100 + (899/(size[0]*size[1]))*(i+1), self.data, corr = Corr, Print = False, FluxLog = LogBool)[2], y = NewFunctions.delt(self.no_gals, 100, 100 + (899/(size[0]*size[1]))*(i+1), self.data, corr = Corr, Print = False, FluxLog = LogBool)[3])
                
            
            elif size[1] >= size[0]:
                axs[int(i/a), i%a].scatter(x = NewFunctions.delt(self.no_gals, 100, 100 + (899/(size[0]*size[1]))*(i+1), self.data, corr = Corr, Print = False, FluxLog = LogBool)[2], y = NewFunctions.delt(self.no_gals, 100, 100 + (899/(size[0]*size[1]))*(i+1), self.data, corr = Corr, Print = False, FluxLog = LogBool)[3])
                
                axs[int(i/a), i%a].set_xscale((lambda x: 'linear' if x else 'log')(LogBool))
                axs[int(i/a), i%a].set_yscale((lambda x: 'linear' if x else 'log')(LogBool))
                axs[int(i/a), i%a].text(0.08, 0.9, '$\Delta$t = {0:.2f}e8'.format((8.99/(size[0]*size[1]))*(i+1)), transform = axs[int(i/a), i%a].transAxes, fontsize = 12)
                
                if i%a == 0:
                    axs[int(i/a), i%a].set_ylabel((lambda x: 'log(SFR Flux)' if x else 'SFR Flux')(LogBool))
                    axs[(b - 1), i%a].set_xlabel(((lambda x: 'log(AGN Flux)' if x else 'AGN Flux')(LogBool)))
                elif i < a:
                    axs[(b - 1), i].set_xlabel(((lambda x: 'log(AGN Flux)' if x else 'AGN Flux')(LogBool)))
                                
            elif size[0] > size[1]:
                axs[int(i/b), i%b].scatter(x = NewFunctions.delt(self.no_gals, 100, 100 + (899/(size[0]*size[1]))*(i+1), self.data, corr = Corr, Print = False, FluxLog = LogBool)[2], y = NewFunctions.delt(self.no_gals, 100, 100 + (899/(size[0]*size[1]))*(i+1), self.data, corr = Corr, Print = False, FluxLog = LogBool)[3])
                axs[int(i/b), i%b].set_xscale((lambda x: 'linear' if x else 'log')(LogBool))
                
    
        '''if not LogBool:
            axs[i%3, int(i/3)].set_xlim(1e39, 1e50)
        
        axs[i%3, int(i/3)].set_yscale((lambda x: 'linear' if x else 'log')(LogBool))
    
        if i%3 == 0:
            axs[int(i/3), i%3].set_ylabel((lambda x: 'log(SFR Flux)' if x else 'SFR Flux')(LogBool))
            axs[(i%3) + 2, int(i/3)].set_xlabel(((lambda x: 'log(AGN Flux)' if x else 'AGN Flux')(LogBool)))
            axs[(i%3), int(i/3)].set_title(Titles[int(i/3)])
            axs[int(i/3), i%3].text(0.08, 0.9, '$\Delta$t = {}e8'.format(3.5 + 2*int(i/3)), transform = axs[int(i/3), i%3].transAxes, fontsize = 12)'''

        
    
    
    
    
    
    
    
    