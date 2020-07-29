# smart (Spectral Modeling Analysis and RV Tool)
The `smart` is a Markov Chain Monte Carlo (MCMC) forward-modeling framework for spectroscopic data, currently working for Keck/NIRSPEC and SDSS/APOGEE.

For NIRSPEC users, required adjustments need to be made before reducing private data using [NIRSPEC-Data-Reduction-Pipeline(NSDRP)](https://github.com/Keck-DataReductionPipelines/NIRSPEC-Data-Reduction-Pipeline), to perform telluric wavelength calibrations, and to forward model spectral data. The code is currently being developed.

Authors:
* Dino Chih-Chun Hsu (UCSD)
* Adam Burgasser, PI (UCSD)
* Chris Theissen (BU, UCSD)
* Jessica Birky (UCSD)

## Code Setup:
Dependencies. The `smart` has been tested under the following environments:
* Python 3.6.10/3.7.3
* astropy 3.0.4/3.0.5
* numpy 1.12.1/1.13.3/1.18.1
* scipy 0.19.0/1.4.1
* matplotlib 2.2.3/3.1.3
* pandas 0.20.1/0.23.4/1.0.1
* emcee 3.0.2/3.0.3.dev4+gc14b212
* corner 2.0.1
* wavelets (for defringeflat)

Download the `smart` and the forked and modified version of the NSDRP to your computer.

There are two ways of setting up the code:

<br/>(i) In the terminal under the `smart` folder, 

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

## Reducing the data using NSDRP:
To add required keywords to the headers before reducing private data using [NSDRP](https://github.com/Keck-DataReductionPipelines/NIRSPEC-Data-Reduction-Pipeline), use the addKeyword function and the [input](https://github.com/chihchunhsu/smart/blob/master/input_reduction.txt) text file:
```
>>> import smart
>>> smart.addKeyword(file='input_reduction.txt')
```
The example txt file is for setting up the data information such as file numbers of darks, flats, arcs, and sources. 

Note that you don't need to perform this task if you download the data directly from the Keck Observatory Archive (KOA).

To reduce the data, use the forked [NSDRP](https://github.com/chihchunhsu/NIRSPEC-Data-Reduction-Pipeline) on the command line:

```
$ python ~/path/to/NIRSPEC-Data-Reduction-Pipeline/nsdrp.py rawData/ reducedData/ -oh_filename ~/path/to/NIRSPEC-Data-Reduction-Pipeline/ir_ohlines.dat -spatial_jump_override -verbose -debug
```

, where the directory rawData/ is the path to the raw data is, and reducedData/ is the path where you want to store the reduce data.

You can also perform the command line code run_nsdrp.py to add keywords, defringe flats, and reduce the data all at once:

```
$ python ~/path/to/smart/utils/run_nsdrp.py -f rawData/
```

<!---*## Dark Subtraction:
You can also optionally subtract the dark frames using subtractDark.py before running the NSDRP. This may be put into the NSDRP in the future.---> 

## Defringe Flats:
The algorithm follows Rojo & Harrington (2006) to remove fringe patterns from flat files. The example and sample outputs are upder the example folder.

```
>>> import smart
>>> smart.defringeflatAll(data_folder_path, diagnostic=False)
```

## Wavelength Calibration using Telluric Standard Spectra:
The algorithm follows Blake at el. (2010) to cross-correlate the ESO atmospheric model and an observed telluric spectrum, fit the residual, and iterate the process until the standard deviation of the residual reaches a mininum. The example and sample outputs are upder the "examples" folder.

```
>>> import smart
>>> smart.run_wave_cal(data_name, data_path, order_list, save_to_path, test=False, save=True)
```

<!---*## Forward Modeling Science Spectra:---> 
