# SMART (Spectral Modeling Analysis and RV Tool)
The `SMART` is a Markov Chain Monte Carlo (MCMC) forward-modeling framework for spectroscopic data, currently working for the Keck/NIRSPEC, SDSS/APOGEE, and IGRINS high-resolution near-infrared spectrometers. A slightly modified version that can model the Keck/OSIRIS medium-resolution near-infrared spectrometer is available [here](https://github.com/ctheissen/osiris_fmp).

For NIRSPEC users, required adjustments need to be made before reducing private data using [NIRSPEC-Data-Reduction-Pipeline(NSDRP)](https://github.com/Keck-DataReductionPipelines/NIRSPEC-Data-Reduction-Pipeline), to perform telluric wavelength calibrations, and to forward model spectral data. The code is currently being developed.

Authors:
* Dino Chih-Chun Hsu (Northwestern, UCSD)
* Adam Burgasser, PI (UCSD)
* Chris Theissen (UCSD, BU)
* Jessica Birky (UW, UCSD)

## Code Setup:
Dependencies. The `SMART` has been tested under the following environments:
* Python 3.6.10/3.7.3
* astropy 3.0.4/3.0.5
* numpy 1.12.1/1.13.3/1.18.1
* scipy 0.19.0/1.4.1
* matplotlib 2.2.3/3.1.3
* pandas 0.20.1/0.23.4/1.0.1
* emcee 3.0.2/3.0.3.dev4+gc14b212
* corner 2.0.1
* openpyxl 3.0.10
* wavelets (for defringeflat)

Download the `SMART` and the forked and modified version of the NSDRP to your computer.

There are two ways of setting up the code:

<br/>(i) In the terminal under the `SMART` folder, 

```
python setup.py install
```

<br/>(ii) Set up the environment variables in the `.bashrc` or `.bash_profile`

```
export PYTHONPATH="/path/to/smart:${PYTHONPATH}"
```

To model the SDSS/APOGEE spectra, you will also need to put the associated APOGEE LSF and wavelength fits files under `forward_model/apogee/`:

[apLSF-a-05440020.fits](https://dr13.sdss.org/sas/dr13/apogee/spectro/redux/r6/cal/lsf/apLSF-a-05440020.fits)<br/>
[apLSF-b-05440020.fits](https://dr13.sdss.org/sas/dr13/apogee/spectro/redux/r6/cal/lsf/apLSF-b-05440020.fits)<br/>
[apLSF-c-05440020.fits](https://dr13.sdss.org/sas/dr13/apogee/spectro/redux/r6/cal/lsf/apLSF-c-05440020.fits)<br/>
[apWave-a-02420038.fits](https://dr13.sdss.org/sas/dr13/apogee/spectro/redux/r6/cal/wave/apWave-a-02420038.fits)<br/>
[apWave-b-02420038.fits](https://dr13.sdss.org/sas/dr13/apogee/spectro/redux/r6/cal/wave/apWave-b-02420038.fits)<br/>
[apWave-c-02420038.fits](https://dr13.sdss.org/sas/dr13/apogee/spectro/redux/r6/cal/wave/apWave-c-02420038.fits)<br/>

The codes under the apogee folder are from Jo Bovy's [apogee](https://github.com/jobovy/apogee) package.

## Defringe Flats:
The algorithm follows Rojo & Harrington (2006) to remove fringe patterns from flat files. The example and sample outputs are upder the example folder.

```
>>> import smart
>>> smart.defringeflatAll(data_folder_path, diagnostic=False)
```

## Reducing the data using NSDRP:
To reduce the data for the whole folder, use the forked [NSDRP](https://github.com/ctheissen/NIRSPEC-Data-Reduction-Pipeline) on the terminal:

```
$ python ~/path/to/NIRSPEC-Data-Reduction-Pipeline/nsdrp.py rawData/ reducedData/ -oh_filename ~/path/to/NIRSPEC-Data-Reduction-Pipeline/ir_ohlines.dat -spatial_jump_override -verbose -debug
```

, where the directory rawData/ is the path to the raw data is, and reducedData/ is the path where you want to store the reduce data.

To reduce the data for the for two nodding files with a single flat file, you can run the following command:

```
$ python ~/path/to/NIRSPEC-Data-Reduction-Pipeline/nsdrp.py flat_file.fits nod_A_file.fits -b nod_B_file.fits -oh_filename ~/path/to/NIRSPEC-Data-Reduction-Pipeline/ir_ohlines.dat -spatial_jump_override -verbose -debug
```

## Wavelength Calibration using Telluric Standard Spectra:
To calibrate the most precise wavelength solutions with a given a data, we rely on the telluric spectrum using the algorithm that follows [Blake at el. (2010)](https://ui.adsabs.harvard.edu/abs/2010ApJ...723..684B/abstract) and [Burgasser et al. (2016)](https://ui.adsabs.harvard.edu/abs/2016ApJ...827...25B/abstract) to cross-correlate the [ESO atmospheric model](https://ui.adsabs.harvard.edu/abs/2014A%26A...568A...9M/abstract) and an observed telluric spectrum, fit the residual, and iterate the process until the standard deviation of the residual reaches a mininum.

```
>>> import smart
>>> smart.run_wave_cal(data_name, data_path, order_list, save_to_path, test=False, save=True, applymask=True, pwv='0.5')
```

The parameters `data_name`, `data_path`, `save_to_path` have the type str, while `order_list` has the type list. Optional parameter `applymask` provides a simple sigma-clipping mask to remove cosmic rays in the telluric spectrum. Optional parameter `pwv` is the precipitable water vapor parameter for users to adjust. Typically, 0.5 mm to 3.0 mm can fit the most cases under different weather conditions. 

## Forward Modeling Science and Telluric Spectra:
To fit the science data, `SMART` provies various self-consistent synthetic modeling grids to forward-model the data that are available to download [here](https://drive.google.com/drive/folders/1P-NrlxdyX3nphRgN4R-4oS86BKZ4Z9th). Users will need to place the downloaded files to `smart/libraries`.

We perform the MCMC forward-modeling fitting to the high-resolution near-infrared spectroscopic data for both the telluric and science files.

The telluric spectrum after calibrated with its wavelength solution is modeled with the equation:

<img src="https://render.githubusercontent.com/render/math?math=D[p] = C[p(\lambda)] \times \Big[ T \big[ p^*(\lambda) \big] \otimes \kappa_G (\Delta \nu_{inst}(p)) \Big] + C_{flux}.">

, where <img src="https://render.githubusercontent.com/render/math?math=D[p]"> is the forward-model telluric spectrum, <img src="https://render.githubusercontent.com/render/math?math=C[p(\lambda)]"> is the continuum, <img src="https://render.githubusercontent.com/render/math?math=T \big[ p^*(\lambda) \big]"> is the earth atmosphere absorption model as a function of airmass and precipitable water vapor (pwv), <img src="https://render.githubusercontent.com/render/math?math=\Delta \nu_{inst}(p)"> is the instrumental line-spread function, and <img src="https://render.githubusercontent.com/render/math?math=C_{flux}"> is the nuisance parameter accouting for small flux offsets.

You can run the following command on the terminal:

```
>>> python /SMART_BASE/smart/smart/forward_model/run_mcmc_telluric_airmass_pwv.py order date_obs tell_data_name tell_path save_to_path -nwalkers 50 -step 600 -burn 300 -pixel_start 50 -pixel_end -80
```

The required parameters are nirspec order sorting filter `order` (e.g. 33), data of observation (e.g. 20100101) `date_obs`, telluric data name (e.g. nspec200101_1001) `tell_data_name`, telluric file path `tell_path` (e.g. BASE/data_obs), saving path `save_to_path`, and optional paramters MCMC number of chains/walkers `-nwalkers`, number of steps `-step`, burn-in `-burn`, starting/ending pixels `-pixel_start` and `-pixel_end`.

The most important parameter used as the input in the science modeling is the NIRSPEC instrumental line-spread function `lsf`.

The science spectrum is modeled with the equation:

<img src="https://render.githubusercontent.com/render/math?math=D[p] = C[p] \times \Bigg[ \bigg(M \Big[p^* \big(\lambda \big[ 1 + \frac{RV^*}{c}\big] \big) , T_{\text{eff}}, \log{g} \Big] \otimes \kappa_R (v\sin{i}) \bigg) \times T \big[ p^*(\lambda) \big] \Bigg] \otimes \kappa_G (\Delta \nu_{inst}) + C_{flux}">

, where where <img src="https://render.githubusercontent.com/render/math?math=D[p]"> is the forward-model science spectrum, <img src="https://render.githubusercontent.com/render/math?math=C[p(\lambda)]"> is the continuum, <img src="https://render.githubusercontent.com/render/math?math=M \big[ p^*(\lambda) \big]"> is the self-consistent synthetic stellar/substellar modeling grids as a function of effective temperature <img src="https://render.githubusercontent.com/render/math?math=T_{\text{eff}}"> and surface gravity <img src="https://render.githubusercontent.com/render/math?math=\log{g}">, corrected with radial velocity <img src="https://render.githubusercontent.com/render/math?math=RV"> and projected rotational velocity <img src="https://render.githubusercontent.com/render/math?math=v\sin{i}">, <img src="https://render.githubusercontent.com/render/math?math=[ T \big[ p^*(\lambda) \big]"> is the earth atmosphere absorption model, <img src="https://render.githubusercontent.com/render/math?math=\Delta \nu_{inst}(p)"> is the instrumental line-spread function, <img src="https://render.githubusercontent.com/render/math?math=C_{flux}"> is the nuisance parameter accouting for small flux offsets. There are also two nudge factors for small wavelength offsets and noise scaling factor in the routine.

You can run the following command on the terminal:

```
>>> python /SMART_BASE/smart/smart/forward_model/run_mcmc_science.py order date_obs sci_data_name tell_data_name data_path tell_path save_to_path lsf -outlier_rejection 3.0 -nwalkers 50 -step 1000 -burn 800 -moves 2.0 -pixel_start 10 -pixel_end -80 -applymask False -modelset btsettl08
```

The required parameters order sorting filter `order`, data of observation `date_obs`, science data name `sci_data_name`, telluric data name `tell_data_name`, science file path `data_path`, telluric file path `tell_path`, saving path `save_to_path`, optional paramters MCMC number of chains/walkers `-nwalkers`, number of steps `-step`, burn-in `-burn`, starting/ending pixels `-pixel_start` and `-pixel_end` are defined the same as the telluric data modeling routine. The NIRSPEC line-spread function `lsf` is obtained from the telluric data modeling (typically 4.8 km/s). The outlier rejection `-outlier_rejection` is to perform a sigma-clipping outlier rejection (in this case sigma=3.0) to remove bad pixels by comparing the resiaudl of the best-fit model and observed data. Finally, the model set to use `-modelset` in this case is the [BT-Settl](https://ui.adsabs.harvard.edu/abs/2012RSPTA.370.2765A/abstract) models. Other model sets are availale and described in detailed [here](https://github.com/chihchunhsu/smart/tree/master/smart/libraries).

## Citation:

If you use this code in your research, please cite [Hsu et al. 2021; ApJS, 257, 45](https://ui.adsabs.harvard.edu/abs/2021ApJS..257...45H/abstract) and the frozen version is on [Hsu et al. 2021 Zenodo](https://ui.adsabs.harvard.edu/abs/2021zndo...4765258H/abstract)


