# smart (Spectral Modeling Analysis and RV Tool)
The `smart` is a Markov Chain Monte Carlo (MCMC) forward-modeling framework for spectroscopic data, currently working for Keck/NIRSPEC and SDSS/APOGEE.

For NIRSPEC users, required adjustments need to be made before reducing private data using [NIRSPEC-Data-Reduction-Pipeline(NSDRP)](https://github.com/Keck-DataReductionPipelines/NIRSPEC-Data-Reduction-Pipeline), to perform telluric wavelength calibrations, and to forward model spectral data. The code is currently being developed.

Authors:
* Dino Chih-Chun Hsu (UCSD)
* Adam Burgasser, PI (UCSD)
* Chris Theissen (BU, UCSD)
* Jessica Birky (UCSD)

## Code Setup:
Dependencies:
* astropy 3.0.4
* numpy 1.12.1
* scipy 0.19.0
* matplotlib 2.2.3
* pandas 0.20.1
* wavelets (for defringeflat)

Download the smart and the forked and modified version of the NSDRP to your computer.

Set up the environment variables in the `.bashrc` or `.bash_profile`

```
export PYTHONPATH="/path/to/smart:${PYTHONPATH}"
```

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
