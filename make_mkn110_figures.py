#
# Create plots from the MKN 110 data
#

import pandas as pd
import numpy as np
import os.path
from astropy.time import Time
from extinction import fitzpatrick99 as f99
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy import stats
import scipy.constants as spc

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.patches as patches
import matplotlib.colors as mcl
from matplotlib import gridspec
from astropy.convolution import convolve, Box1DKernel


class FluxData:
    '''
    Class used to hold MKN 110 flux data. The stored data
    and methods are intended to be used in generating the 
    plots, specified in the PlotCreator class. 
    
    The methods load and structure the data, normalise it,
    and apply the various required corrections, as described
    in the paper. In addition, there are several methods to
    fit the data with pre-specified models and to calculate 
    the uncertainties on the data and the fits.
    
    Attributes:
    data        Pandas DataFrame; contains the flux data for the 
                continuum and lines.
    fn          string; name of the flux file to be loaded.
    z           float; redhshift of MKN 110
    flux_cols   list of (str,str); names of the flux columns,
                grouped by continuum range/line species.
    rm lags     dict; measured Reverberation Mapping lags
                for several line species.
                
    Methods:
    load_data(filename=None,separator='\t')
        Load data from file into Pandas DataFrame
    
    correct_and_normalise() 
        Apply correction to flux data and normalise.
        Store to file (if no file exists).
        

    fit_flat_inv(x,*p)
        Fit two-component function to data.

    run_fitting_function(line)
        Perform the iterative fitting described in Section 6.3,
        for the line species specified by 'line'.

    find_chi_sq(ydat,ysig,model)
        Return chi^2 for given data and mode.
        
        
    calc_chisq_contour_error(line,N_grid=10,print_res=False,
                             chi2_levels = [2.3,6.17,11.8])
        Estimate error on linear fit on data for the line
        species defined by 'line'


    def run_linear_fit_epochs(line,print_res=False)
        Run linear regression on subsets of the data (by epoch),
        for the line species defined by 'line'.
    '''

    def __init__(self):
        '''
        Constructor for FluxData class.
        '''
        self.data = None
        self.fn = 'mkn110_flux_data_mr.tsv'
        self.z = 0.0355
        self.flux_cols = [('F5100','F5100err'),
                          ('F4265','F4265err'),
                          ('F3750','F3750err'),
                          ('Halpha','Halphaerr'),
                          ('Hbeta','Hbetaerr'),
                          ('HeII4686','HeII4686err'),
                          ('HeI4471','HeI4471err'),
                          ('HeI5876','HeI5876err')]
        self.rm_lags = {'Halpha':32.3,'Hbeta':24.2,
                        'HeII4686':3.9,'HeI4471':11.1,
                        'HeI5876':10.7} # RM Lags from K01 (in days)
        

    def load_data(self,filename=None,separator='\t'):
        '''
        Load the MKN 110 data into a Pandas dataframe
        
        Keyword arguments:
        filename   string; path of the flux datafile. If none is
                   specified, the default filename will be used.
        separator  string; separator in the input file. Default
                   is a tab.
        '''
        if filename:
            self.fn = filename
        self.data = pd.read_csv(self.fn,sep=separator,
                                header=0,index_col=False)


    def __corr_heii_flux(self):
        '''
        Correction to the flux of He II 4686, using a fitted value 
        for the narrow line component, based on the SDSS spectrum 
        _before normalisation_ (Section 5.2). This value was measured
        as 6*10^15 erg cm^-2 s^-1.
        '''
        heii_corr = 6 # fluxes are in units of 10^15 erg cm^-2 s^-1
        self.data.HeII4686 = self.data.HeII4686 - heii_corr
        

    def __corr_rm_lags(self,rmdates=None):
        '''
        Adjust the line fluxes (where possible) using the Reverberation
        Mapping lags from K01, such that each line flux best matches the 
        approriate 5100AA continuum flux (Section 5.2).
        
        Keyword arguments:
        rmdates   list of 2-tuples; list of date-pairs (MJD) which mark
                  the ranges where the RM correction is applied. This is 
                  to exclude regions where the gap between observations 
                  is too large for this correction to work. Default values
                  match the K01, FAST-RM1, and FAST-RM2 data-sets.
        '''
        if rmdates:
            rmd = rmdates
        else:
            rmd = [(51495,51679),
                   (52500,52800),
                   (52900,53140)]
        for dates in rmd:
            dfc = self.data.loc[(self.data.mjd>dates[0]) & (self.data.mjd<dates[1])].copy(deep=True)
            # shift dates for lines and find matches
            for c in self.rm_lags.keys():
                dfc['shifted'] = dfc.mjd + self.rm_lags[c]
                matchidx = []
                for _,row in dfc.iterrows():
                    matchidx += [ (dfc.mjd-row['shifted']).abs().idxmin() ]
                # set values in original DataFrame
                self.data.loc[dfc.index,c] = self.data[c].iloc[matchidx].values

        
    def __normalise_flux(self):
        '''
        Normalise the flux data to the values of MJD 47574
        (Section 5.2).
        '''
        ref_ind = self.data[self.data.mjd==47574].index.values[0]
        for c in self.flux_cols:
            self.data[c[0]+'_norm'] = self.data[c[0]]/self.data[c[0]][ref_ind]
            self.data[c[1]+'_norm'] = self.data[c[1]]/self.data[c[0]][ref_ind]

    
    def __corr_norm_flux(self):
        '''
        Apply corrections for the non-variable component to
        the normalised fluxes of Halpha, Hbeta, and He I, using
        the shifts to the normalised fluxes (Section 6.2). 
        ---
        A copy of the normalised data before the application of
        these shifts is stored in the self.data DataFrame under
        the column header with suffix '_norm_ps' (pre-shift), as
        these data are required for the generation of Figure 4. 
        '''
        shifts = {'Halpha':0.45,'Hbeta':0.35,'HeI5876':0.2}
        for line in shifts.keys():
            self.data[line+'_norm_ps'] = self.data[line+'_norm'].copy()
            self.data[line+'_norm'] = self.data[line+'_norm'] - shifts[line]

                
    def correct_and_normalise(self):
        '''
        Wrapper function that creates a normalised flux data-set and
        applies the appropriate corrections before and after the
        normalisation. A copy of the fully corrected normalised data
        is saved to file.
        '''
        self.__corr_heii_flux()
        self.__corr_rm_lags()
        self.__normalise_flux()
        self.__corr_norm_flux()
        if not os.path.isfile('./mkn110_normflux_data_mr.tsv'):
            col_names = ['mjd']+[e+'_norm' for e in sum(self.flux_cols,())]+['dataset']
            self.data.to_csv('./mkn110_normflux_data_mr.tsv',
                             columns=col_names,
                             index=False,header=True,
                             sep='\t',float_format='%.5f')
     
                             
    def fit_flat_inv(self, x,*p):
        '''
        Fitting function for the normalised flux data,
        as defined in Section 6.3. This is a two-component
        function, consisting of a sloped line joined onto
        a flat line. The fitting parameters are the slope
        a and the height of the flat line y_s (the saturation
        level).
        
        Positional arguments:
        x    list of floats; the input values for the x-axis.
             These will be the normalised He II 4686 fluxes.
        p    input values for the function: a, y_sat
        
        Returns:
        f    NumPy array; y_values, f(x).
        '''
        if len(p)!=1:
            y_s,a = p
        else:
            y_s,a = p[0]
        if type(x) == np.ndarray:
            f = []
            for el in x:
                if y_s > el*a:
                    f+=[float(el*a)]
                else:
                    f+=[float(y_s)]
        else:
            if y_s > x*a:
                f = [float(x*a)]
            else:
                f = [float(y_s)]
        return np.array(f)

        
    def __do_fit(self, xdat, ydat, **kwargs):
        '''
        Perform a least-squares fit given input data 
        (xdat and ydat), using the two-component fitting
        function defined under fit_flat_inv().
        
        Positional arguments:
        xdat   NumPy array of floats; input x-values.
        ydat   NumPy array of floats; input y-values.
        
        Keyword arguments:
        sig    NumPy array of floats; uncertainties on ydat.
        bnds   2-tuple of 2-tuples; lower-upper bound
               pairs for 'y_s' and 'a' in fit_flat_inv(),
               to be applied during the fit.
               
        Returns:
        par    NumPy array; best-fit values.
        pcov   Numpy array; covariance matrix for fit.
        '''
        if 'bnds' in kwargs:
            par, pcov = curve_fit(self.fit_flat_inv,xdat,ydat,p0=[1,1],
                                  sigma=kwargs['sig'],bounds=kwargs['bnds'])
        else:
            par, pcov = curve_fit(self.fit_flat_inv,xdat,ydat,p0=[1,1],
                                  sigma=kwargs['sig'])
        return par,pcov

    
    def __calc_uncertainty(self,xdat,ydat,par):
        '''
        Calulate difference between fit and data 
        (to estimate overall error).
        
        Positional arguments:
        xdat
        ydat
        par
        
        Returns:
        err       float; stanard deviation of y-residuals
        errArray  NumPy array; list of err of length ydat
        '''
        dif = ydat-self.fit_flat_inv(xdat,*par)
        err = np.std(dif)
        return err, np.full(len(ydat),err)


    def run_fitting_function(self,line):
        '''
        Perform the iterative fitting described in Section 6.3,
        for the normalised flux data for a given line species.
        
        Positional arguments:
        line      string; name of the line species.
        
        Returns:
        est_err   float; error estimated from the data,
                  based on the first iteration of fitting. 
        par       NumPy array; best-fit values.
        pcov      Numpy array; covariance matrix for fit.
        '''
        # Datasets
        idx = self.data.index[self.data[line].notna()]
        xdat = self.data.iloc[idx].HeII4686_norm.to_numpy()
        ydat = self.data.iloc[idx][line+'_norm'].to_numpy()
        ysig = self.data.iloc[idx][line+'err_norm'].to_numpy()
        
        # First run (estimate scatter & error)
        par,cov = self.__do_fit(xdat,ydat,sig=ysig)
        
        # Calculate error
        est_err, ysig = self.__calc_uncertainty(xdat,ydat,par)
        
        # Second run (best-fit parameters)
        par,cov = self.__do_fit(xdat,ydat,sig=ysig)
        return est_err,par,cov
        

    def find_chi_sq(self,ydat,ysig,model):
        '''
        Calculate the chi^2 for a given model and data.
        
        Positional arguments:
        ydat    NumPy array of float; data.
        ysig    NumPy array of float; uncertainty.
        model   NumPy array of float; uncertainty.

        Returns:
        chisq   float; chi^2 value.
        '''
        return np.sum(np.power((ydat-model)/ysig,2))
        
        
    def calc_chisq_contour_error(self,line,N_grid=10,print_res=False,chi2_levels = [2.3,6.17,11.8]):
        '''
        Calculate the uncertainties on fitted parameters
        based on the chi^2 contours of the fits.
        
        Positional argument:
        line          string; name of line species.
        
        Keyword arguments:
        N_grid       integer; size of side of NxN grid.
        print_res    boolean; print the results of the error
                     estimation (based on the contours) to
                     screen.
        chisq_levels list of floats; chi^2 levels to use in error
                     calculation (the first element in the list will
                     be used).
        
        Returns:
        Sat_err   2-list of floats; upper and lower uncertainty
                  (1-sigma) on the saturation level.
        Slope_err 2-list of floats; upper and lower uncertainty
                  (1-sigma) on the initial slope.
        plots     list; objects required for plotting.
            z         NxN list of floats; chi^2 squared values over
                      the parameter grid (for plotting).
            zlims     2x2 tuple; corners of the bounding box used
                      to determine the uncertaintes (for plotting).
            rng       list of NumPy arrays; xrange, yrange.
            grid      NumPy Meshgrid; grid for values
            bd        2-tuple; best-fit values of ysat and a.
        '''
        # Dataset
        idx = self.data.index[self.data[line].notna()]
        xdat = self.data.iloc[idx].HeII4686_norm.to_numpy()
        ydat = self.data.iloc[idx][line+'_norm'].to_numpy()
        ysig = self.data.iloc[idx][line+'err_norm'].to_numpy()
        
        # Run fit to get best-fit parameters
        err,par,cov = self.run_fitting_function(line)
        ysig = np.full(len(ydat),err)
        chi_min = self.find_chi_sq(ydat,ysig,self.fit_flat_inv(xdat,par))
        xc, yc = par # central values; min_chi^2
        
        # Define grid in parameter space and fill it with chi^2 values
        xrl, yrl = 30, 25 # range of plot (scaled in % of best-fit values)
        xr = np.linspace(xc-(xc/100)*xrl,xc+(xc/100)*xrl,N_grid)
        yr = np.linspace(yc-(yc/100)*yrl,yc+(yc/100)*yrl,N_grid)
        xv,yv = np.meshgrid(xr,yr)
        z = []
        for jj in range(len(xr)):
            zrow = []
            for kk in range(len(yr)):
                model = self.fit_flat_inv(xdat,[ xv[jj][kk],yv[jj][kk] ])
                zrow += [ self.find_chi_sq(ydat,ysig,model)-chi_min ]
            z += [zrow]

        # Search for the limits of the chi-squared contours ('bounding box')
        # 1) The error on the saturation level
        
        Sat_err, Slope_err = [[],[]], [[],[]]
        zlow_sa,zhigh_sa = 0,0
        for jj in range(len(xr)):
            for kk in range(len(yr)):
                if zlow_sa==0 and z[kk][jj]<chi2_levels[1]:
                    zlow_sa = (jj,kk)
                    break
            if zlow_sa!=0: break
        for jj in reversed(range(len(xr))):
            for kk in reversed(range(len(yr))):
                if zhigh_sa==0 and z[kk][jj]<chi2_levels[1]:
                    zhigh_sa = (jj,kk)
                    break
            if zhigh_sa!=0: break
        Sat_err[0] += [ np.abs(xr[zlow_sa[0]]-xc) ]
        Sat_err[1] += [ np.abs(xr[zhigh_sa[0]]-xc) ]

        # 2) The error on the initial slope
        zlow_sl,zhigh_sl = 0,0
        for jj in range(len(yr)):
            for kk in range(len(xr)):
                if zlow_sl==0 and z[jj][kk]<chi2_levels[1]:
                    zlow_sl = (kk,jj)
                    break
            if zlow_sl!=0: break
        for jj in reversed(range(len(yr))):
            for kk in reversed(range(len(xr))):
                if zhigh_sl==0 and z[jj][kk]<chi2_levels[1]:
                    zhigh_sl = (kk,jj)
                    break
            if zhigh_sl!=0: break
        Slope_err[0] += [ np.abs(yr[zlow_sl[1]]-yc) ]
        Slope_err[1] += [ np.abs(yr[zhigh_sl[1]]-yc) ]
        
        # Objects for plotting
        zlims = ((zlow_sa,zhigh_sa), (zlow_sl,zhigh_sl))
        rng = (xr,yr)
        grid = (xv,yv)
        bf = (xc,yc)
        plots = [z,zlims,rng,grid,bf]
        
        # Print results if required
        if print_res:
            print ('##### {} #####'.format(line))
            print ('est. uncertainty = {:.3f}'.format(err))
            print ('y_sat = {} - {} + {}'.format(xc,Sat_err[0],Sat_err[1]))
            print ('slope = {} - {} + {}'.format(yc,Slope_err[0],Slope_err[1]))
            print ('##############')
            
        return Sat_err, Slope_err, plots


    def run_linear_fit_epochs(self,line,print_res=False):
        '''
        Perform a linear fit on the normalised flux data
        for each of the three epochs defined in Section 6.4.
        
        Positional arguments:
        line       string; name of the line species to fit.
        
        Keyword arguments:
        print_res  boolean; print fit results to screen.
        
        Returns:
        res        tuple; contains three dictionaries, one for
                   each of the epochs. The dictionaries are the 
                   standard output for the SciPy stats method 
                   linregress
        '''
        # Epoch I
        cnd = ( ((self.data.dataset=='BK99') | (self.data.dataset=='WHP98')) &
                (self.data[line+'_norm'].notna()) )
        x = self.data.loc[cnd].HeII4686_norm
        y = self.data.loc[cnd,line+'_norm']
        res1 = stats.linregress(x,y)
        # Epoch II
        cnd = ( ((self.data.dataset=='K01') | (self.data.dataset=='SDSS') | 
                 (self.data.dataset=='FAST-RM1') | (self.data.dataset=='FAST-RM2')) &
                 (self.data[line+'_norm'].notna()) )
        x = self.data.loc[cnd].HeII4686_norm
        y = self.data.loc[cnd,line+'_norm']
        res2 = stats.linregress(x,y)
        # Epoch III
        cnd = ( ((self.data.dataset=='FAST-Landt') | (self.data.dataset=='WHT') | 
                 (self.data.dataset=='FAST-New')) & (self.data[line+'_norm'].notna()) )
        x = self.data.loc[cnd].HeII4686_norm
        y = self.data.loc[cnd,line+'_norm']
        res3 = stats.linregress(x,y)
        
        # print results
        if print_res:
            for ii,r in enumerate((res1,res2,res3)):
                print('Epoch',ii+1)
                print('slope:',r.slope,'+/-',r.stderr)
                print('offset:',r.intercept,'+/-',r.intercept_stderr)
                print('R²:',r.rvalue**2)
        
        return (res1, res2, res3)
        

class SpecData:
    '''
    Class to hold the data for optical spectroscopic observations
    of MKN 110. The stored data and methods are intended to be 
    used in generating the plots, specified in the PlotCreator
    class.
    
    Atributes:
    fileloc   string; path/directory name where to locate the
              spectral files.
    spectra   dict; dictionary of NumPy arrays containing the
              loaded spectra.
    z         float; redshift of MKN 110.
    
    Methods:
    load_spectra(self,specfiles=None,specnames=None)
        Load the spectra for the specified file names.
        
    apply_greyshift(self,gsfile='./spectra/FAST_greyshifts_ymd',datelist=None)
        Shift the spectra by a pre-defined grey correction
        based on fits to the [OIII] lines.
        
    correct_for_redshift(self,z=None)
        De-redshift the spectra.
    '''

    def __init__(self,fileloc='./spectra'):
        '''
        Constructor for the SpecData class.
        
        Keyword arguments:
        fileloc   string; directory name for the location
                  of the spectral data for MKN 110. Default
                  is './spectra'
        '''
        self.fileloc = fileloc
        self.spectra = None
        self.z = 0.0355


    def load_spectra(self,specfiles=None,specnames=None):
        '''
        Load the spectra into Numpy arrays.
        
        Keyword arguments:
        specfiles   list of strings; filenames of the spectra to
                    be loaded. Default are the spectra needed to
                    create Figure 3.
        specnames   list of strings; names by which the spectra
                    can be identified. The default names will be
                    the filenames.
        '''
        if specfiles:
            spf = specfiles
        else:
            spf = ['sdss_mkn110_20011209.spec',
                   'fast_20021103.spec',
                   'fast_landt_20060106.spec',
                   'fast_landt_20070122.spec',
                   'wht_mkn110_20160616.spec',
                   'wht_mkn110_20170221.spec']
        if specnames and len(specnames) == len(specfiles):
            names = specnames
        else:
            specnames = [s.split('.')[0] for s in spf]
        spf = [os.path.join(self.fileloc,s) for s in spf]
        spectra = []
        for f in spf:
            spectra += [np.loadtxt(f).T]
        self.spectra = dict(zip(specnames,spectra))
        
        
    def apply_greyshift(self,gsfile='./spectra/FAST_greyshifts_ymd',datelist=None):
        '''
        Load the greyshifts for the FAST spectra (Section 3.2)
        and apply them to the relevant spectra. The greyshifts 
        were determined by matching the integrated [OIII] fluxes
        to those of the SDSS spectrum.
        
        Keyword arguments:
        gsfile    string; full path of the file containing the 
                  greyshifts. Default location is in the 'spectra'
                  directory.
        datelist  list of strings; list of dates (format=YYYYMMDD)
                  to match spectra. Default is to take the dates
                  from the filenames of the spectra. If specifying
                  dates, the order should match the order of 
                  'specfiles' in the load_spectra method.
        '''
        gsl = pd.read_csv(gsfile,delim_whitespace=True,
                         names=('date','gshift'))
        gs = dict(zip(gsl.date,gsl.gshift))
        if datelist:
            dlst = dict(zip(self.spectra.keys(),datelist))
        else:
            dlst = dict(zip(self.spectra.keys(),
                      [s.split('_')[-1] for s in self.spectra.keys()]
                      ) )
            for k in self.spectra.keys():
                date = int(dlst[k])
                if date in gs.keys():
                    self.spectra[k][1] = self.spectra[k][1]/gs[date]
                    
                    
    def correct_for_redshift(self,z=None):
        '''
        Correct the spectra for the redshift of MKN 110.
        
        Keyword arguments:
        z   float; redshift (to avoid using the default
            value of 0.0355)
        '''
        if z:
            corr = 1+z
        else:
            corr = 1+self.z
        for sp in self.spectra.values():
            sp[0] = sp[0]/corr
        

class ProfileFittingData:
    '''
    Class to hold the data for results of fits to the available
    MKN 110 spectra. The stored data and methods are intended to
    be used in generating the plots, specified in the PlotCreator
    class.
    
    Attributes:
    fn      string; name of the file containing the data.
    fn_st   string; name of the file containing the data for
            the stacked spectrum.
    data    Pandas DataFrame; contains profile data once loaded.
    data_st Pandas DataFrame; contains profile data for 
            stacked spectra once loaded.
    
    Methods:
    load_data(filename=None,filename_st=None,separator='\t')
        Load profile data from file.
        
    find_p_pearson(arrin1,arrin2,print_res=False)
        Calculate p-value associated with Pearson coefficient.
        
    find_rs (arrin1,arrin2,print_res=False)
        Calculate Spearman ranked correlation coefficient.
        
    gravz_estimate(v,a)
        Modelled line width for given gravitational line redshift.
    '''

    def __init__(self,fileloc='./'):
        '''
        Constructor for the ProfileFittingData class.
        
        Keyword arguments:
        fileloc   string; directory name for the location
                  of the data based on fitting the available
                  MKN 110 spectra. Default is './'.
        '''
        self.fn = 'mkn110_profile_data.tsv'
        self.fn_st = 'mkn110_profile_data_stacked.tsv'
        self.data = None
        self.data_st = None

    def load_data(self,filename=None,filename_st=None,separator='\t'):
        '''
        Load the fitted MKN 110 data into a Pandas DataFrame.
        
        Keyword arguments:
        filename     string; path of the datafile. If none is
                     specified, the default filename will be used.
        filename_st  string; path of the datafile containing the
                     results of fitting the stacked data. If none
                     is specified, the default filename will be used.
        separator    string; separator in the input file. Default
                     is a tab.
        '''
        if filename:
            self.fn = filename
        self.data = pd.read_csv(self.fn,sep=separator,
                                header=0,index_col=False)
        if filename_st:
            self.fn_st = filename_st
        self.data_st = pd.read_csv(self.fn_st,sep=separator,
                                   header=0,index_col=False)

    
    def find_p_pearson (self,arrin1,arrin2,print_res=False):
        '''
        Function for correlation metric, calculating
        the p-value for t-statistic derived from 
        Pearson correlation coefficient, using a 
        two-tailed test.
        
        Positional arguments:
        arrin1     list or Numpy array; x-data. 
        arrin2     list or Numpy array; y-data.
        
        Keywword arguments:
        print_res  boolean; print results to screen.
        
        Returns:
        pval       float; p-value (two-tailed test)
        '''
        r = stats.pearsonr(arrin1,arrin2)
        pcc = r[0]
        bn = len(arrin1)
        Tpcc = pcc*np.sqrt((bn-2)/(1-pcc*pcc))
        pval = stats.t.sf(np.absolute(Tpcc), bn-2)
        if print_res:
            print ('Pearson coefficient: {}; pvalue (two-tailed ='.format(r,pval*2))
        return pval*2.
        

    def find_rs (self,arrin1,arrin2,print_res=False):
        '''
        Function for correlation metric, calculating
        the Spearman ranked correlation efficient.
        
        Positional arguments:
        arrin1     list or Numpy array; x-data. 
        arrin2     list or Numpy array; y-data.
        
        Keywword arguments:
        print_res  boolean; print results to screen.
        
        Returns:
        spcc       float; Spearman coefficient.
        '''
        r1 = stats.rankdata(arrin1)
        r2 = stats.rankdata(arrin2)
        bn = len(arrin1)
        Spcc = 1 - 6./(bn*(bn*bn-1))*sum(np.power(r2-r1,2))
        if print_res:
            print ('Spearman coefficient = {}'.format(Spcc))
        return Spcc


    def gravz_estimate(self,v,a):
        '''
        Expected relation between the line width and
        the broad-narrow line offset (in km/s), under the
        assumption that the offset is caused solely by
        the gravitational redshift.
        
        Positional arguments:
        v    float; The offset-velocity (km/s) 
        a    float; sin(i)/f
        '''
        lw = np.sqrt(spc.c*1e-3*v)*a
        return lw


class PlotCreator:
    '''
    Class to create the figures 2--11 in the Paper. Contains 
    methods for individual plots, as well as a wrapper function
    to generate all plots at once.
    
    Attributes:
    fd          FluxData instance.
    sd          SpecData instance.
    pdf         ProfileFittingData instance.
    fluxnames   dict; strings containing names of lines
                for printing in plots.
    fluxclr     dict; colours associated with line species
                for consistent use in plots.
    fluxmrk     dict; markes for lines species, for 
                consistent use.
    datasets    list of strings; names of the included data-sets.
    dataclr     dict; colours to be used per data-set, for
                consistent use.
    datamrk     dict; markers for data-set, for consistent use.
    epochs      dict; mapping from epochs (I,II,III)) to data-sets.
    
    Methods:
    make_figures()
        Convenience function to create all plots.
    
    make_figure2(outname='Figure2_f5100_lc.pdf',show_plot=False)
        Create plot of F5100 light curve.
        
    make_figure3(outname='Figure3_spec_example.pdf',show_plot=False,labels=None)
        Create plot showing example spectra.
        
    make_figure4(outname='Figure4_lines_vs_f5100.pdf',show_plot=False)
        Create plot showing line fluxes plotted against F5100 continuum.
        
    make_figure5a(outname='Figure5a_HaHb_vs_f5100.pdf',show_plot=False)
        Create plot of Halpha/Hbeta ration against F5100 continuum,
        compared with a theoretical curve.
        
    make_figure5b(outname='Figure5b_HeIHa_vs_f5100.pdf',show_plot=False)
        Create plot of He I 5876/Halpha against F5100 continuum,
        compared with a theoretical curve.
        
    make_figure6(outname='Figure6_norm_flux_fit.pdf',show_plot=False)
        Create plot of normalised line fluxes against He II 4686. This
        plot includes fitting results.
        
    make_chisq_plots(self,outname='Figure_chisq_contour.pdf',
    			         show_plot=False,print_res=False,ngrid=1000)
        Create plot that is not in the paper, which illustrates the method
        used to calculate errors on the fitting results.
        
    make_figure7(outname='Figure7_ysat_vs_rmlag.pdf',
                 show_plot=False,ysat_err=None)
        Create plot of the fitted saturation level against the RM lags.
        
    make_figure9(outname='Figure9_offset_vs_flux.pdf',show_plot=False)
        Create plot of the line offsets against He II 4686 flux.
        
    make_figure10(outname='Figure10_width_vs_flux.pdf',show_plot=False)
        Create plot of the line widths (sigma) against the He II 4686 flux.
        
    make_figure11(outname='Figure11_width_vs_offset.pdf',show_plot=False)
        Create plot of the line widhts (sigma) against the line offsets.
    '''

    def __init__(self,fluxdata=None,specdata=None,profiledata=None):
        '''
        Constructor for the PlotCreator class.
        
        Keyword arguments:
        fluxdata     FluxData instance; object containing a 
                     Pandas DataFrame with the MKN 110 flux data.
        specdata     SpecData instance; object containing the 
                     NumPy arrays containing MKN 110 spectral data.
        profiledata  ProfileFittingData instance; object containing
                     the Pandas DataFrames containing MKN 110 spectral 
                     line-profile data.
        '''
        self.fd = fluxdata
        self.sd = specdata
        self.pfd = profiledata
        self.fluxnames = {'Halpha':r'H$\alpha$',
                          'Hbeta':r'H$\beta$',
                          'HeII4686':'HeII$\lambda$4686',
                          'HeI5876':'HeI$\lambda$5876'}
        self.fluxclr = {'Halpha':'g','Hbeta':'b',
                        'HeII4686':'c','HeI5876':'r'}
        self.fluxmrk = {'Halpha':'H','Hbeta':'s',
                        'HeII4686':'o','HeI5876':'v'}         
        self.datasets = ['WHP98', 'BK99',
                         'K01', 'SDSS',
                         'FAST-RM1','FAST-RM2',
                         'FAST-Landt','WHT',
                         'FAST-New']
        self.dataclr = dict(zip(self.datasets,
                                ['y','k','g','b',
                                 'r','orange','cyan',
                                 'orchid','limegreen']
                            ) )
        self.datamrk = dict(zip(self.datasets,
                                ['x','H','^','s',
                                 'D','*','p','v','2']
                            ) )
        self.epochs = dict(zip(self.datasets,
                               ['I','I','II','II','II',
                                'II','III','III','III']
                            ) )
        
                    
    def make_figure2(self,outname='Figure2_f5100_lc.pdf',show_plot=False):
        '''
        Create plot of log(F5100) over time, also showing
        the definition of the epochs and the contribution
        of the different data-sets.
        
        Keyword arguments:
        outname    string; name of the output file.
        show_plot  boolean; show resulting plot as well as 
                   saving it to file.
        '''
        fig, ax = plt.subplots(1,figsize=(9,5))
        mclr = ['y']+7*['k']+['limegreen']
        mjd_lims = [46500,58600]
        df = self.fd.data.copy() # shorthand for convenience
        
        # Plot data
        df.sort_values(by=['mjd'],inplace=True)
        ax.plot(df.mjd,np.log10(df.F5100),c='k',linestyle='dashed')
        for d,mec in zip(self.datasets,mclr):
            dfs = df.loc[df.dataset==d]
            ax.errorbar(dfs.mjd, np.log10(dfs.F5100),
                        yerr = 1./(np.log(10)*dfs.F5100)*dfs.F5100err,
                        marker = self.datamrk[d], markersize = 10,
                        color = self.dataclr[d], ecolor = self.dataclr[d],
                        markeredgecolor = mec, elinewidth = 2.4,
                        linestyle = 'None', capsize = 4,
                        label = d)
        
        # Set plot labels etc.
        ax.set_xlabel('MJD',fontsize=14,fontweight='bold')
        ax.set_ylabel(r'log($f_{5100}$) (10$^{-15}$ erg cm$^{-2}$ s$^{-1}$ $\rm\AA^{-1}$)',
                      fontsize=14,fontweight='bold')
        handles, labels = ax.get_legend_handles_labels()
        l_order = [1,0,2,3,4,5,6,7,8]
        handles = [handles[ii] for ii in l_order]; labels = [labels[ii] for ii in l_order]
        handles = [h[0] for h in handles]
        ax.legend(handles,labels,loc='lower left',edgecolor='k',ncol=2)
        ax.minorticks_on()
        ax.set_xlim(mjd_lims)
        ax.set_ylim(-0.25,0.84)

        # Upper x-axis
        new_ticks = [str(i)+'-01-01' for i in range(1988,2023,5)]
        ax2 = ax.twiny()
        ax2.set_xlim(mjd_lims)
        ax2.set_xticks(Time(new_ticks).mjd)
        ax2.set_xticklabels(s[:4] for s in new_ticks)

        # Show regions for 'epochs'
        ax.vlines(50900,-10,10,ls='--',color='grey',linewidth=2,alpha=0.8)
        ax.vlines(53500,0.35,10,ls='--',color='grey',linewidth=2,alpha=0.8)
        ax.text(0.23,0.9,'I',fontsize=21,fontweight='normal',
                bbox=dict(facecolor='none', edgecolor='k'),
                transform=ax.transAxes)
        ax.text(0.46,0.9,'II',fontsize=21,fontweight='normal',
                bbox=dict(facecolor='none', edgecolor='k'),
                transform=ax.transAxes)
        ax.text(0.77,0.9,'III',fontsize=21,fontweight='normal',
                bbox=dict(facecolor='none', edgecolor='k'),
                transform=ax.transAxes)

        # Inset with close-up of RM data
        axins = inset_axes(ax,width='40%',height='50%',loc=4)
        axins.plot(df.mjd,np.log10(df.F5100),
                   c='k',linestyle='dashed')
        for d,mm in zip(['K01','SDSS','FAST-RM1','FAST-RM2'],[8,9,7,9]):
            dfs = df.loc[df.dataset==d]
            axins.errorbar(dfs.mjd, np.log10(dfs.F5100),
                           yerr=1./(np.log(10)*dfs.F5100)*dfs.F5100err,
                           marker=self.datamrk[d],markersize=mm,
                           color=self.dataclr[d],ecolor=self.dataclr[d],
                           markeredgecolor='k',elinewidth=2.4,
                           linestyle='None',capsize=4,
                           label=d)
        axins.set_xticks([52000],minor=False)
        axins.tick_params(which='both',direction='in',
                          labelbottom=0,length=6)
        axins.set_xlim(51400,53180)
        axins.set_ylim(-0.25,0.6)
        
        # Save and show output
        plt.savefig(outname)
        if show_plot:
            plt.show()
            
        
    def make_figure3(self,outname='Figure3_spec_example.pdf',show_plot=False,labels=None):
        '''
        Create plot of several example optical spectra (Figure 3).
        
        Keyword arguments:
        outname    string; name of the output file.
        show_plot  boolean; show resulting plot as well as 
                   saving it to file.
        labels     list of strings; list of labels for the
                   spectra,if not using the default spectra.
                   The length must match the number of spectra
                   stored in the SpecData object.
        '''
        fig, ax = plt.subplots(1, figsize=(14,7))
        lbl = ['52252-SDSS','52975-FAST','53741-FAST',
                    '57539-WHT','57801-WHT','58521-FAST']
        clr = ['k','c','limegreen','r',
               'y','grey','#19c10e','g',
               'orange','m','b','pink']
        lines = {'Ha':[6565,r'H$\alpha$',0],'Hb':[4861,r'H$\beta$',0],
                 'Hg':[4340,r'H$\gamma$',0],'Hd':[4102,r'H$\delta$',0],
                 'He':[3970,r'H$\epsilon$',0],'Hz':[3889,r'H$\zeta$/HeI$\lambda$3889',0],
                 'HeII':[4686,r'HeII$\lambda$4686',0],'HeI5':[5876,r'HeI$\lambda$5876',0],
                 'OII':[3727,r'OII$\lambda$3727',0],'OIIIa':[4959,'',0],
                 'OIIIb':[5007,r'OIII$\lambda$5007',0],'OI':[6300,r'OI$\lambda$6300',0]}
        xmin,xmax = 3600,7200
        ymin,ymax = 0,2300
        
        # Plot spectra
        for k,lb,cl in zip(self.sd.spectra.keys(),lbl,clr):
            ax.plot(self.sd.spectra[k][0],convolve(self.sd.spectra[k][1],Box1DKernel(6))*1e17,
                    label=lb,c=cl)
            # Determine label positions for emission line identifications    
            for ll in lines.keys():
                # Wavelength range around emission line peak
                lam_rng = np.isclose(self.sd.spectra[k][0],lines[ll][0],rtol=5e-4)
                # Max flux value around line peak
                lm = np.amax(self.sd.spectra[k][1][lam_rng])*1e17
                if lines[ll][2]<lm:
                    lines[ll][2]=lm
            lines['Hz'][2]=1520
        
        # Plot emission lines identifications
        for ll in lines.values():
            if ll[2]>ymax: ll[2]=ymax/1.1
            ax.vlines(ll[0], ymin, ymax, linestyles='--', colors='#4A4A4A')
            ax.text(ll[0]+16,ll[2]*1.05,ll[1],
                    transform=ax.transData,color='k',fontsize=14,
                    bbox=dict(facecolor='white', edgecolor='none',alpha=0.6))
            
        # Set labels etc.    
        ax.axis([xmin,xmax,ymin,ymax])
        ax.set_xlabel(r'Wavelength in QSO Restframe ($\rm \AA$)',
                      fontsize=20, fontweight='bold')
        ax.set_ylabel(r'f$_\lambda$ (10$^{-17}$ erg cm$^{-2}$ s$^{-1}$ $\rm\AA^{-1}$)',
                      fontsize=20, fontweight='bold')
        legend = ax.legend(fontsize=12, loc='upper right')
        ax.minorticks_on()
        ax.tick_params(axis='both', which='major', labelsize=13)

        # Save and show output
        plt.tight_layout()
        plt.savefig(outname)
        if show_plot:
            plt.show()
            

    def make_figure4(self,outname='Figure4_lines_vs_f5100.pdf',show_plot=False):
        '''
        Create plot of normalised line fluxes for Halpha,
        Hbeta, He I 5876, and He II 4686, plotted against
        the normalised F5100 flux.
        
        Keyword arguments:
        outname    string; name of the output file.
        show_plot  boolean; show resulting plot as well as 
                   saving it to file.
        '''
        fig, ax = plt.subplots(1, figsize=(6,4))
        xmin,xmax=0,2.3
        ymin,ymax=0,4.5
        lines = ['Halpha','Hbeta','HeII4686','HeI5876']
        suffix = ['_norm_ps','_norm_ps','_norm','_norm_ps']

        # Plot data and line showing 1:1 correspondence
        for line,sfx in zip(lines,suffix):
            ax.scatter(self.fd.data['F5100_norm'], self.fd.data[line+sfx],
                       c=self.fluxclr[line],marker=self.fluxmrk[line],
                       edgecolor='k', lw=0.5, s=40,
                       label=self.fluxnames[line])
        ax.plot([0,3],[0,3],color='r',ls='--')

        # Set labels etc.
        ax.axis([xmin,xmax,ymin,ymax])
        ax.set_xlabel(r'$f_{5100}$ (normalised)',fontweight='bold')
        ax.set_ylabel('Normalised Flux',fontweight='bold')
        legend = ax.legend(scatterpoints=1, fontsize=12, loc='upper left',edgecolor='k')
        legend.set_zorder(20)

        # Save and show output
        plt.tight_layout()
        plt.savefig(outname)
        if show_plot:
            plt.show()
            

    def __calc_reddening_line_ratio(self,line1,line2,Frng):
        '''
        Using the reddening curve of Fitzpatrick (1999),
        calculate the flux ratio for two line species,
        assuming it is only altered by dust extinction.
        
        The change in Av is calculated based on the F5100
        flux, as: 

        Delta_Av = -2.5 log (F_new/F_ref)
        
        Here F_ref is the flux at a given reference point
        (this only functions as normalisation).
        
        The predicted change in the line flux ratio is then 
        calculated as:
        
        Delta(mag1-mag2) = (A_1/E(B-V) - (A_2(E(B-V)) * Delta_Av/3.1
        
        Here the subscripts 1 and 2 refer to the 1st and 2nd line
        in the line ratio. A_1 and A_2 are the value of the
        extinction, calculated at the line centres.
        
        The change in line flux ratio is then plotted scaled to 
        the reference values (same observation date as the F5100
        reference value).
        ---        
        Positional arguments:
        line1   string; name of the first line (numerator)
        line2   string; name of the second line (denominator)
        -> Acceptable line names: Halpha, Hbeta, HeI5876
        Frng    list of floats; the range of the F5100 flux
                to calculate the line ratio over. Default is
                (0.05 - 6) * 10^15 erg cm^-2 s^-1.
        
        Returns:
        R       numpy array; the predicted line flux ratios, 
                assuming the change in F5100 is caused by
                reddening only.
        ''' 
        line_wl = {'Halpha': np.array([6563]),
                   'Hbeta': np.array([4861]),
                   'HeI5876': np.array([5876])}
        # Filter data to those epochs were both line fluxes are available
        df = self.fd.data.loc[(self.fd.data[line1].notna()) & (self.fd.data[line2].notna())]
        # Reference point for the reddening curve
        ref_idx = 12
        F0 = df.F5100.iloc[ref_idx]
        R0 = df[line1].iloc[ref_idx]/df[line2].iloc[ref_idx]
        M0 = -2.5*np.log10(R0)
        # Reddening factor based on the line ratio in the reference epoch
        # This factor represents (A_1/E(B-V) - A_2/E(B-V)) /3.1
        cf = (f99(line_wl[line1],1.,3.1)-f99(line_wl[line2],1.,3.1))/3.1

        F_range = np.array(Frng)
        d_Av = -2.5*np.log10(F_range/F0) # Delta_Av
        R = np.power(10,-.4*(M0+cf*d_Av))
        return R


    def make_figure5a(self,outname='Figure5a_HaHb_vs_f5100.pdf',show_plot=False):
        '''
        Create one of the subplots of Figure 5, comparing the
        Halpha/Hbeta ratio, plotted against F5100, with a 
        theoretical prediction, based on reddening only
        (Section 5.4). The reddening curve is from Fitzpatrick
        (1999).
        
        Keyword arguments:
        outname    string; name of the output file.
        show_plot  boolean; show resulting plot as well as 
                   saving it to file.
        '''
        fig, ax = plt.subplots(1,figsize=(8,5.5))
        xmin,xmax = 0,6
        ymin,ymax = 2,6.5

        # Plot data
        ax.scatter(self.fd.data.F5100,
                   self.fd.data.Halpha/self.fd.data.Hbeta,
                   marker='o',s=100,c='r',edgecolor='k',
                   label='Flux Data')
        F_range = np.arange(0.05,6,0.01)
        R = self.__calc_reddening_line_ratio('Halpha','Hbeta',F_range)
        ax.plot(F_range,R,'-',c='k',label='F99 Extinction')

        # Labels etc.
        ax.axis([xmin,xmax,ymin,ymax])
        ax.set_xlabel(r'$f_{5100}$ ($10^{-15}$ erg cm$^{-2}$ s$^{-1}$)',
                      fontsize=18,fontweight='bold')
        ax.set_ylabel(r'H$\alpha$/H$\beta$',fontsize=18,fontweight='bold')
        ax.legend(scatterpoints=1,fontsize=14,loc='upper right',edgecolor='k')
        ax.tick_params(axis='both', which='major', labelsize=14)

        # Save and show output
        plt.tight_layout()
        plt.savefig(outname)
        if show_plot:
            plt.show()
            
            
    def make_figure5b(self,outname='Figure5b_HeIHa_vs_f5100.pdf',show_plot=False):
        '''
        Create one of the subplots of Figure 5, comparing the
        HeI5876/Halpha ratio, plotted against F5100, with a 
        theoretical prediction, based on reddening only
        (Section 5.4). The reddening curve is from Fitzpatrick
        (1999).
        
        Keyword arguments:
        outname    string; name of the output file.
        show_plot  boolean; show resulting plot as well as 
                   saving it to file.
        '''
        fig, ax = plt.subplots(1,figsize=(8,5.5))
        xmin,xmax = 0,6
        ymin,ymax = 0,.08

        # Plot data
        ax.scatter(self.fd.data.F5100,
                   self.fd.data.HeI5876/self.fd.data.Halpha,
                   marker='o',s=100,c='r',edgecolor='k',
                   label='Flux Data')
        F_range = np.arange(0.05,6,0.01)
        R = self.__calc_reddening_line_ratio('HeI5876','Halpha',F_range)
        ax.plot(F_range,R,'-',c='k',label='F99 Extinction')

        # Labels etc.
        ax.axis([xmin,xmax,ymin,ymax])
        ax.set_xlabel(r'$f_{5100}$ ($10^{-15}$ erg cm$^{-2}$ s$^{-1}$)',
                      fontsize=18,fontweight='bold')
        ax.set_ylabel(r'HeI$\lambda$5876/H$\alpha$',fontsize=18,fontweight='bold')
        ax.legend(scatterpoints=1,fontsize=14,loc='upper right',edgecolor='k')
        ax.tick_params(axis='both', which='major', labelsize=14)

        # Save and show output
        plt.tight_layout()
        plt.savefig(outname)
        if show_plot:
            plt.show()


    def make_figure6(self,outname='Figure6_norm_flux_fit.pdf',show_plot=False):
        '''
        Create plot showing the normalised fluxes for Halpha, Hbeta,
        and He I 5876, plotted against the normalised He II 4686 flux.
        Also included are the results of the fitting functions.
        
        Keyword arguments:
        outname    string; name of the output file.
        show_plot  boolean; show resulting plot as well as 
                   saving it to file.        
        '''
        fig, ax = plt.subplots(1,figsize=(10,6))
        xmin,xmax = 0,4.5
        ymin,ymax = -0.2,2.5
        lines = ['Halpha','Hbeta','HeI5876']
        
        # Plot data
        heii_rng = np.arange(0,5,0.01)
        for line in lines:
            idx = self.fd.data.index[self.fd.data[line].notna()]
            err,par,cov = self.fd.run_fitting_function(line)
            ax.scatter(self.fd.data.HeII4686_norm,self.fd.data[line+'_norm'],
                       c=self.fluxclr[line],marker=self.fluxmrk[line],
                       edgecolor='k',s=80,
                       label=self.fluxnames[line])
            ax.plot(heii_rng,self.fd.fit_flat_inv(heii_rng,par[0],par[1]),
                    c=self.fluxclr[line])
            if line == 'Halpha':
                est_err = err

        # Add an indication of the size of the estimated errors
        ax.add_patch(patches.Rectangle((.25,1.4),0.4,0.4,fill=False))
        ax.errorbar(0.45,1.6,xerr=est_err,yerr=est_err,
                    ecolor='k',elinewidth=1.4,zorder=1,
                    capsize=3,fmt='None')
        ax.scatter(0.45,1.6,marker='o',c='g',
                   edgecolor='k',s=80,zorder=2)
        ax.tick_params(labelsize=13)
        
        # Labels etc.
        ax.axis([xmin,xmax,ymin,ymax])
        ax.set_xlabel(r'HeII$\lambda$4686 (Ionising Flux)',
                      fontsize=15,fontweight='bold')
        ax.set_ylabel('Normalised Flux',fontsize=15,fontweight='bold')
        ax.legend(scatterpoints=1, fontsize=16, loc='upper left',
                  framealpha=1, edgecolor='k')

        # Save and show output
        plt.tight_layout()
        plt.savefig(outname)
        if show_plot:
            plt.show()


    def make_chisq_plots(self,outname='Figure_chisq_contour.pdf',
    			         show_plot=False,print_res=False,ngrid=1000):
        '''
        Create plots showing the chi^2 contours for the
        fitted two-component functions (Section 6.3) to the 
        data of Halpha, Hbeta, and He I 5876. The resulting
        errors can also be printed to screen. These are the
        uncertainties included in Table 5.
        
        Keyword arguments:
        outname    string; name of the output file.
        show_plot  boolean; show resulting plot as well as 
                   saving it to file.
        print_res  boolean; print the results of the error
                   estimation (based on the contours) to
                   screen.
        ngrid      integer; size of grid (ngrid x ngrid) to be
                   passed to calc_chisq_contour_error method
                   of FluxData class.
        '''
        fig, axs = plt.subplots(1,3,figsize=(18,5))
        chisq = [2.3,6.17,11.8]
        lines = ['Halpha','Hbeta','HeI5876']
    
        for line,ax in zip(lines,axs):
            # Get data
            Sat_err,Slope_err,plot_things = self.fd.calc_chisq_contour_error(line,N_grid=ngrid,
                                                                             print_res=print_res,
                                                                             chi2_levels = chisq)
            z, zlims, rng, grid, bf = plot_things
            xr,yr = rng
            xv,yv = grid
            xc, yc = bf
            zlow_sa, zhigh_sa = zlims[0]
            zlow_sl, zhigh_sl = zlims[1]

            #Plot
            ax.vlines(xc,yr[0],yr[-1],linestyles='dotted',
                      colors=self.fluxclr[line], lw=1.)
            ax.hlines(yc,xr[0],xr[-1],linestyles='dotted',
                      colors=self.fluxclr[line], lw=1.)
            ax.vlines(xr[zlow_sa[0]],yr[0],yr[-1],colors='k',lw=1.1)
            ax.vlines(xr[zhigh_sa[0]],yr[0],yr[-1],colors='k',lw=1.1)
            ax.hlines(yr[zlow_sl[1]],xr[0],xr[-1],colors='k',lw=1.1)
            ax.hlines(yr[zhigh_sl[1]],xr[0],xr[-1],colors='k',lw=1.1)
            contour = ax.contour(xv,yv,z,chisq,colors=self.fluxclr[line])
            ax.clabel(contour,colors=self.fluxclr[line],inline=1,fontsize=11)

            ax.set_xlabel('Saturation Level',fontweight='bold',fontsize=16)
            ax.set_ylabel('Slope',fontweight='bold',fontsize=16)
            ax.tick_params(labelsize=12)
            ax.text(0.67,0.9,self.fluxnames[line],
                    color=self.fluxclr[line],fontsize=16,
                    transform=ax.transAxes,
                    bbox=dict(facecolor='w',edgecolor='w'))

        # Save and show output
        plt.tight_layout()
        plt.savefig(outname)
        if show_plot:
            plt.show()
            

    def make_figure7(self,outname='Figure7_ysat_vs_rmlag.pdf',
                     show_plot=False,ysat_err=None):
        '''
        Create plot of the fitted parameter y_sat (Section 6.3),
        plotted against the RM lags reported in K01. The errors
        on y_sat can be calculated using the FluxData method  
        calc_chisq_contour_error. To speed of the procedure, the
        errors can also be passed to this function, if they have
        already been calculated (e.g. in make_chisq_plots).
        
        Keyword arguments:
        outname    string; name of the output file.
        show_plot  boolean; show resulting plot as well as 
                   saving it to file.
        ysat_err   2-list of 3-tuples; list of lower errors
                   and upper errors for the three lines (in order
                   Halpha, Hbeta, He I 5876. If this argument
                   is not given, the errors will be calculated.
        '''
        fig, ax = plt.subplots(1,figsize=(3,3))
        lines = ['Halpha','Hbeta','HeI5876']
        Lags = [32.3,24.2,10.7]
        Lag_err = [[4.9,3.3,6.0],[4.3,3.7,8.0]]
        xmin, xmax = 0, 40
        ymin, ymax = 0., 1.7
        
        # The saturation level (y_sat) and the measurement uncertainties
        ysat = []
        if ysat_err:
            Sat_err = ysat_err
            for line in lines:
                err,par,cov = self.fd.run_fitting_function(line)
                ysat += [par[0]]
        else:
            chisq = [2.3,6.17,11.8]
            Sat_err = []
            for line in lines:
                sat_err,slope_err,plots = self.fd.calc_chisq_contour_error(line,N_grid=100,
                                                                           chi2_levels = chisq)
                Sat_err += [sat_err]
                ysat += [plots[4][0]] # The best-fit values
            Sat_err = np.array(Sat_err).T[0] # Re-order for use in Matplotlib's errorbar method
        
        # Plot data
        ax.errorbar(Lags,ysat,xerr=Lag_err,yerr=Sat_err,
                    ecolor='k',elinewidth=1.2,capsize=4,fmt='o')

        # Set labels and plot details
        ax.axis([xmin,xmax,ymin,ymax])
        ax.set_xlabel('RM Lag (days)',fontsize=12,fontweight='bold')
        ax.set_ylabel('Saturation Level',fontsize=12,fontweight='bold')
        for line,loc in zip(lines,[(0.68,0.25),(0.45,0.4),(0.02,0.65)]):
            ax.text(*loc,self.fluxnames[line],
                    transform=ax.transAxes,fontsize=13)

        # Save and show output
        plt.tight_layout()
        plt.savefig(outname)
        if show_plot:
            plt.show()
            

    def make_figure8(self,outname='Figure8_hbeta_epoch_fit.pdf',
                     show_plot=False,print_res=False,mrk_epoch=True):
        '''
        Create plot showing the normalised flux of Hbeta
        plotted against the normalised flux of He II 4686.
        The plot splits the data by orgin of the data-set. 
        Furthermore, it shows the results of a linear regression
        for each of the three subepochs (Section 6.4).
        
        Keyword arguments:
        outname    string; name of the output file.
        show_plot  boolean; show resulting plot as well as 
                   saving it to file.
        print_res  boolean; print the results of linear
                   regression to screen.
        mrk_epoch  boolean; switch to alternative format of
                   plot, marking the datapoints by data-set,
                   instead of epoch (default=True, use epochs).
        '''
        # Show results in a plot
        fig, ax = plt.subplots(1,figsize=(9,5.5))
        xmin, xmax = 0, 4.3
        mclr = [None]+7*['k']+[None]
        epoch_mrk = {'I':'s','II':'o','III':'v'}
        epoch_clr = {'I':'r','II':'g','III':'b'}
        epoch_alp = {'I':0.8,'II':0.6,'III':0.4}
        epoch_cnd = { 'I':  ( ((self.fd.data.dataset=='BK99') | (self.fd.data.dataset=='WHP98')) ),
                      'II': ( ((self.fd.data.dataset=='K01') | (self.fd.data.dataset=='SDSS') | 
                               (self.fd.data.dataset=='FAST-RM1') | (self.fd.data.dataset=='FAST-RM2')) ),
                      'III':( ((self.fd.data.dataset=='FAST-Landt') | (self.fd.data.dataset=='WHT') | 
                               (self.fd.data.dataset=='FAST-New')) ) }

        # Run the linear regressions
        res1, res2, res3 = self.fd.run_linear_fit_epochs('Hbeta',print_res=print_res)
        
        # Plot data
        if mrk_epoch:
            for ep in ('I','II','III'):
                x = self.fd.data[epoch_cnd[ep]].HeII4686_norm
                y = self.fd.data[epoch_cnd[ep]].Hbeta_norm
                ax.scatter(x,y,marker=epoch_mrk[ep],
                           c=epoch_clr[ep],edgecolor='k',
                           alpha=epoch_alp[ep],
                           s=140,label='Epoch '+ep)
        else: 
            for ds,mc in zip(self.datasets,mclr):
                x = self.fd.data[self.fd.data.dataset==ds].HeII4686_norm
                y = self.fd.data[self.fd.data.dataset==ds].Hbeta_norm
                ax.scatter(x,y,marker=self.datamrk[ds],
                           c=self.dataclr[ds],edgecolor=mc,
                           s=140,label=ds)

        # Plot fitting results
        xr = np.linspace(xmin,xmax,1000)
        ax.plot(xr,res1.intercept+res1.slope*xr,c='k',ls='-',lw=2,label='Epoch I')
        ax.plot(xr,res2.intercept+res2.slope*xr,c='k',ls=':',lw=2,label='Epoch II')
        ax.plot(xr,res3.intercept+res3.slope*xr,c='k',ls='-.',lw=2,label='Epoch III')
        ax.plot(xr,self.fd.fit_flat_inv(xr,(0.90,0.64)),ls='--',c='g',alpha=0.5,lw=2)
            
        # Set plot details
        ax.set_xlabel(r'HeII$\lambda$4686 (normalised)',fontsize=18,fontweight='bold')
        ax.set_ylabel(r'H$\beta$ (normalised)',fontsize=20,fontweight='bold')
        handles,labels = ax.get_legend_handles_labels()
        if mrk_epoch:
            leg1 = plt.legend(loc='lower right',edgecolor='k',
                              borderaxespad=2,
                              handles=handles[3:],labels=labels[3:],
                              fontsize=13.5)
        else:
            labels = [self.epochs[ds]+': '+ds for ds in self.datasets]
            leg1 = plt.legend(loc='lower right', edgecolor='k',
                              borderaxespad=2,
                              handles=handles[3:],labels=labels,
                              fontsize=13.5)
        leg2 = plt.legend(loc='upper left', edgecolor='k',
                          borderaxespad=1, handles=handles[:3],
                          labels=('Epoch I','Epoch II','Epoch III'),fontsize=13.5)
        ax.add_artist(leg1)
        ax.add_artist(leg2)
        ax.tick_params(labelsize=12.5)
        ax.minorticks_on()
        ax.set_ylim((-0.2,1.76))
        ax.set_xlim((xmin,xmax))
        plt.tight_layout()

        # Save and show output
        plt.tight_layout()
        plt.savefig(outname)
        if show_plot:
            plt.show()
            
    
    def __plot_offset_flux_width(self,xd,yd,xlbl,ylbl,
                                 yrng,splitdate=53471,txtbox_loc=(0.64,0.8)):
        '''
        Create a plot showing two of the following plotted against eachother:
        1) the offset (broad-line component to narrow-line component)
        2) the broad He II 4686 flux
        3) the broad line width
        for Hbeta and He II 4686, in a joint plot (Figures 9,10, and 11).
        
        Positional arguments:
        xd         2-list of strings; column names to use for
                   the x-axis (He II 4686, Hbeta).
        yd         2-list of strings; column names to use for
                   the y-axis (He II 4686, Hbeta).
        xlbl       2-list of strings; labels for x-axis
                   (He II 4686, Hbeta).
        ylbl       2-list of strings; labels for y-axis
                   (He II 4686, Hbeta).
        yrng       2D-list of floats; limits on the y-axis. The
                   first sublist for He II 4686, the next one 
                   for Hbeta.
        
        Keyword arguments:
        splitdate  integer; MJD to mark the split between two 
                   RM-campaigns (subsets of the data). One will
                   be plotted in red, the other in blue.
        txtbox_loc 2-tuple; position of textbox with correlation
                   metrics. Position measured as a fraction of the
                   length x-axis and y-axis, respectively.
                   
        Returns:
        fig,ax     Figure and 2D-Axes; the Matplotlib objects
                   containing the plots.
        '''
        fig, ax = plt.subplots(1,2,figsize=(8,4))
        rdrng = mcl.LinearSegmentedColormap.from_list('rdrng', colors=['#FF0000','#000000'])
        blrng = mcl.LinearSegmentedColormap.from_list('blrng', colors=['#8DEEEE','#00008B'])
        df1 = self.pfd.data[self.pfd.data.mjd<splitdate]
        df2 = self.pfd.data[self.pfd.data.mjd>=splitdate]

        for ii,xnm,ynm in zip([0,1],xd,yd):
            # First date range
            im1 = ax[ii].scatter(df1[xnm],df1[ynm],
                                 c=df1.mjd,s=50,cmap=rdrng,
                                 vmin=df1.mjd.min(),vmax=df1.mjd.max(),
                                 marker='o',edgecolor='k',zorder=1)
            ax[ii].errorbar(df1[xnm],df1[ynm],
                            xerr=df1[xnm+'_err'],yerr=df1[ynm+'_err'],
                            fmt='None',ecolor='k',elinewidth=1,
                            capsize=3,zorder=0)
            # Second date range
            im2 = ax[ii].scatter(df2[xnm],df2[ynm],
                                 c=df2.mjd,s=50,cmap=blrng,
                                 vmin=df2.mjd.min(),vmax=df2.mjd.max(),
                                 marker='s',edgecolor='k',zorder=1)
            ax[ii].errorbar(df2[xnm],df2[ynm],
                            xerr=df2[xnm+'_err'],yerr=df2[ynm+'_err'],
                            fmt='None',ecolor='k',elinewidth=1,
                            capsize=3,zorder=0)
            
            # data point for stacked spectra
            ax[ii].errorbar(self.pfd.data_st.iloc[0][xnm],self.pfd.data_st.iloc[0][ynm],
                            xerr=self.pfd.data_st.iloc[0][xnm+'_err'],
                            yerr=self.pfd.data_st.iloc[0][ynm+'_err'],
                            fmt='p',markersize=8,color='lime',markeredgecolor='k',
                            ecolor='k',elinewidth=1,capsize=3)

            # Plot details
            Title = [r'HeII$\lambda$4686',r'H$\beta$']
            ax[ii].axis([0,1.1*np.amax(self.pfd.data[xnm]),
                         yrng[ii][0],yrng[ii][1]])
            ax[ii].set_xlabel(xlbl[ii],fontweight='bold',fontsize=12)
            ax[ii].set_ylabel(ylbl[ii],fontweight='bold',fontsize=12)
            ax[ii].set_title(Title[ii], fontsize=14, fontweight='bold')
            plt.subplots_adjust(wspace = .5)
            
            # Include text box with statistic metrics
            rs = self.pfd.find_rs(self.pfd.data[xnm],self.pfd.data[ynm])
            pval = self.pfd.find_p_pearson(self.pfd.data[xnm],self.pfd.data[ynm])
            ax[ii].text(txtbox_loc[0]*np.amax(self.pfd.data[xnm]),
                        txtbox_loc[1]*(yrng[ii][1]-yrng[ii][0])+yrng[ii][0],
                        r'$r_s$='+'{:.2}'.format(rs)+'\np ='+'{:.2}'.format(pval),
                        color='grey',bbox=dict(facecolor='w',edgecolor='k'))
        
        # Add colour legend
        cax1 = fig.add_axes([0.45, 0.05, 0.02, 0.4])
        cb1 = fig.colorbar(im1, cax1, orientation='vertical', 
                           ticks=np.arange(df1.mjd.min(),df1.mjd.max(),50))
        cb1.ax.tick_params(labelsize=8)
        cax2 = fig.add_axes([0.45, 0.5, 0.02, 0.4])
        cb2 = fig.colorbar(im2, cax2, orientation='vertical',
                           ticks=np.arange(df2.mjd.min(),df2.mjd.max(),1000))
        cb2.ax.tick_params(labelsize=8)
        ax[0].text(1.175, 1.04, 'MJD', fontsize=10, fontweight='bold',
                   transform=ax[0].transAxes)
                   
        return fig,ax
            

    def make_figure9(self,outname='Figure9_offset_vs_flux.pdf',show_plot=False):
        '''
        Create a plot of the broad-narrow line offsets
        against the He II 4686 flux.
        
        Keyword arguments:
        outname    string; name of the output file.
        show_plot  boolean; show resulting plot as well as 
                   saving it to file.
        '''
        xd = ('heii_flux','heii_flux')
        yd = ('heii_offset','hb_offset')
        xlbl = [r'Broad HeII$\lambda$4686 flux',r'Broad HeII$\lambda$4686 flux']
        ylbl = ['Broad-Line Offset (km/s)','']
        yrng = [[0,1800],[0,500]]
        fig,ax = self.__plot_offset_flux_width(xd,yd,xlbl,ylbl,yrng)
       
        # Save and show output
        plt.savefig(outname)
        if show_plot:
            plt.show()


    def make_figure10(self,outname='Figure10_width_vs_flux.pdf',show_plot=False):
        '''
        Create a plot of the broad-line widths against
        the He II 4686 flux.
        
        Keyword arguments:
        outname    string; name of the output file.
        show_plot  boolean; show resulting plot as well as 
                   saving it to file.
        '''
        xd = ('heii_flux','heii_flux')
        yd = ('heii_width','hb_width')
        xlbl = [r'Broad HeII$\lambda$4686 flux',r'Broad HeII$\lambda$4686 flux']
        ylbl = [r'$\sigma_{\mathrm{Broad}}$ (km s$^{-1}$)','']
        yrng = [[1000,2800],[700,1000]]
        fig,ax = self.__plot_offset_flux_width(xd,yd,xlbl,ylbl,yrng)
       
        # Save and show output
        plt.savefig(outname)
        if show_plot:
            plt.show()


    def make_figure11(self,outname='Figure11_width_vs_offset.pdf',show_plot=False):
        '''
        Create a plot of the broad-line widths against
        the broad-narrow line offsets for He II 4686 and
        Hbeta. In addition, the figure includes the results
        of a 
        
        Keyword arguments:
        outname    string; name of the output file.
        show_plot  boolean; show resulting plot as well as 
                   saving it to file.
        '''
        xd = ('heii_offset','hb_offset')
        yd = ('heii_width','hb_width')
        xlbl = [r'Offset HeII$\lambda$4686 (km s$^{-1}$)',r'Offset H$\beta$ (km s$^{-1}$)']
        ylbl = [r'$\sigma_{\mathrm{Broad}}$ (km s$^{-1}$)','']
        yrng = [[0,3100],[0,1300]]
        fig,ax = self.__plot_offset_flux_width(xd,yd,xlbl,ylbl,yrng,txtbox_loc=(0.64,0.1))
        
        # Add grav-z prediction (using f=4.3 and the 21 deg. inclination from K01)
        a_vals = {}
        f = 4.3
        a_vals['heii_amin'] = np.sin(16*np.pi/180)/f
        a_vals['heii_amax'] = np.sin(26*np.pi/180)/f
        a_vals['heii_fit'] = np.sin(21*np.pi/180)/f
        a_vals['hb_amin'] = np.sin(16*np.pi/180)/f
        a_vals['hb_amax'] = np.sin(26*np.pi/180)/f
        a_vals['hb_fit'] = np.sin(21*np.pi/180)/f
        
        for ii,ll in zip([0,1],['heii','hb']):
            xrng = ax[ii].get_xlim()
            xvals = np.arange(xrng[0],xrng[1],1)
            ax[ii].fill_between(xvals,
                                self.pfd.gravz_estimate(xvals,a_vals[f'{ll}_amin']),
                                self.pfd.gravz_estimate(xvals,a_vals[f'{ll}_amax']),
                                color = 'k', alpha=0.15)
            ax[ii].plot(xvals,self.pfd.gravz_estimate(xvals,a_vals[f'{ll}_fit']),c='k')        
       
        # Save and show output
        plt.savefig(outname)
        if show_plot:
            plt.show()

    def make_figures(self):
        '''
        Convenience function to generate the Figures 2-11
        from the paper.
        '''
        self.make_figure2(show_plot=True)
        self.make_figure3(show_plot=True)
        self.make_figure4(show_plot=True)
        self.make_figure5a(show_plot=True)
        self.make_figure5b(show_plot=True)
        self.make_figure6(show_plot=True)
        self.make_chisq_plots(show_plot=True,print_res=True,ngrid=2000)
        self.make_figure7(show_plot=True,ysat_err=[(0.076,0.057,0.151),(0.089,0.096,0.162)])
        self.make_figure8(show_plot=True,print_res=False)
        #self.make_figure8(show_plot=True,print_res=True,mrk_epoch=False)
        self.make_figure9(show_plot=True)
        self.make_figure10(show_plot=True)
        self.make_figure11(show_plot=True)


###################
## Main Function ##
###################

def generate_plots():
    '''
    Main function: load data and generate plots.
    '''
    # Load and normalise flux data
    fd = FluxData()
    fd.load_data()
    fd.correct_and_normalise()
    
    # Load and correct the spectra
    sd = SpecData()
    sd.load_spectra()
    sd.apply_greyshift()
    sd.correct_for_redshift()
    
    # Load the results of the spectral fitting
    pfd = ProfileFittingData()
    pfd.load_data()
    
    # Create plots
    pc = PlotCreator(fluxdata=fd,specdata=sd,profiledata=pfd)
    pc.make_figures()
    

if __name__ == '__main__':

    generate_plots()
    
    
    
