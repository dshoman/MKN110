# Generating Figures on MKN 110 BLR variability

This repository contains the optical fluxes and spectra discussed in the paper [The Long-term broad-line responsivity in MKN 110](ArXiv URL). In addition to these data-sets, the script *make_mkn110_figures.py* produces the figures included in the paper (as well as a few additional ones).

## The Data

The fluxes are included in the tab-separated file *mkn110_flux_data_mr.tsv*. The columns represent the MJD of the observation (except for the stacked data), the relevant emission line, the data-set to which the epoch belongs (data-sets as defined in [the paper](ArXiv URL)). The flux data are comprised of fluxes first published by [Bischoff & Kollatschny (1999)](https://ui.adsabs.harvard.edu/abs/1999A%26A...345...49B/abstract)[^1] and [Kollatschny et al. (2001)](https://ui.adsabs.harvard.edu/abs/2001A%26A...379..125K/abstract)[^2], fluxes derived from the available spectra (listed below), and fluxes first published here.

The spectral data are stored in the **spectra** directory. The spectra are in ASCII format containing two columns: wavelength (Angstrom) and flux density (erg/cm^-2/s^-1/AA). The spectra are labelled by provenance and date:
- The spectra labelled *fast_YYYYMMDD* were dowloaded from the [FAST public archive](http://tdc-www.harvard.edu/instruments/fast/) and flux calibrated as detailed in the paper.
- The spectrum labelled *sdss_mkn110_20011209* was downloaded from the [SDSS public data archive](https://dr16.sdss.org/optical/spectrum/search).
- The spectra labelled *fast_landt_YYYYMMDD* were first published in [Landt et al. (2008)](https://ui.adsabs.harvard.edu/abs/2008ApJS..174..282L/abstract)[^3] and [Landt et al. (2011)](https://ui.adsabs.harvard.edu/abs/2011MNRAS.414..218L/abstract)[^4].

The results of the spectral fitting are included in the tab-separated files *mkn110_profile_data.tsv* and *mkn110_profile_data_stacked.tsv*. The columns represent the data of the spectral observation (does not apply to the stacked data), and for He II 4686 and Hbeta the normalised line flux, narrow-to-broad line component offset in km/s, and the broad line width in km/s. The files *mkn110_profile_data_original.tsv* and *mkn110_profile_data_original_stacked.tsv* contain the same information, however the flux is not normalised (erg/cm^-2/s^-1) and the offsets and line widths are in units of Angstrom.

Further details on the data-sets provided here can be found in [the paper](ArXiv URL).

## Creating the Plots

To run the Python script, open a terminal, navigate to the directory containing the contents of this repository, and simply enter
```
python make_mkn110_figures.py
```

When run, the script makes use of the convenience function *generate_plots*, which will produce each of the figures in turn. The code is structured around the classes **FluxData**, **SpecData**, and **ProfileFittingData**, and **PlotCreator**, all of which are documented through docstrings. The template provided by *generate_plots* can also be followed as an example for the use of these classes, for any interested user. 

Comments, questions, and suggestions are of course always welcome.

### Requirements

The script has been tested using Python 3.8.5 and requires the following packages to be installed (version numbers are those with which this script was tested):
- numpy (1.20.2)
- scipy (1.8.0)
- pandas (1.2.3)
- astropy (5.0.3)
- extinction (0.4.6)

[^1] Bischoff & Kollatschny, 1999, A&A, 345, 49. 
[^2] Kollatschny, 2001, A&A, 379, 125.
[^3] Landt H. et al., 2008, ApJS, 174, 282.
[^4] Landt H. et al., 2011, MNRAS, 414, 218.

