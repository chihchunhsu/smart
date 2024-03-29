## rountine for reducing the NIRSPEC data using the modified NSDRP
## Check KOA keywords --> defringe --> reduce data

import smart
from astropy.io import fits
import os, sys
import warnings
from subprocess import call
import subprocess
import argparse
import glob
import shutil
warnings.filterwarnings("ignore")

## Assume the NSDRP is under the same folder as the smart
FULL_PATH  = os.path.realpath(__file__)
BASE       = os.path.split(os.path.split(os.path.split(FULL_PATH)[0])[0])[0]
BASE       = BASE.split('smart')[0] + 'NIRSPEC-Data-Reduction-Pipeline/'
#BASE = BASE.split('smart')[0] + 'nsdrp_20180925/' #original NSDRP

parser = argparse.ArgumentParser(description="Reduce the NIRSPEC data using NIRSPEC-Data-Reduction-Pipeline",
	usage="python run_nsdrp.py input_dir (output_dir)")

parser.add_argument("files",metavar='f',type=str,
    default=None, help="input_dir (output_dir)", nargs="+")

parser.add_argument("--nodefringe", help="not apply the defringe algorithm", 
    action='store_true')

parser.add_argument("--spatial_rect_flat", help="using median order trace from flat frame to perform spatial rectification", 
    action='store_true')

parser.add_argument("--check_format", help="check if the format satisfies KOA convention keywords", 
    action='store_true')

parser.add_argument("--sowc", help="simple order width calculation", 
    action='store_true')

parser.add_argument("--verbose", help="enables additional logging for debugging", 
    action='store_true')

args = parser.parse_args()
datadir  = args.files

if len(datadir) == 1:
    save_to_path = datadir[0] + '/reduced'
    #save_to_path = 'reduced'
    datadir.append(save_to_path)
else:
    save_to_path = datadir[1]
#datadir.append(datadir2)

originalpath = os.getcwd()
path = originalpath + '/' + datadir[0] + '/'

## store the fits file names
mylist = glob.glob1(path,'*.fits')

if args.check_format:
    #print("Checking the keyword formats: {}".format(datadir[0]))
    for filename in mylist:
        #print(filename)
        file_path = path + filename
        data, header = fits.getdata(file_path, header=True, ignore_missing_end=True)
        if ('IMAGETYP' in header) is False:
            if ('flat lamp off' in str(header['COMMENT'])) is True:
                header['IMAGETYP'] = 'dark'
            elif ('flat field' in str(header['COMMENT'])) is True:
            	header['IMAGETYP'] = 'flatlamp'
            elif ('NeArXeKr' in str(header['COMMENT'])) is True :
            	header['IMAGETYP'] = 'arclamp'
            else:
            	header['IMAGETYP'] = 'object'
            if ('DISPERS' in header) is False:
                header['DISPERS'] = 'high'
    
            fits.writeto(file_path, data, header, overwrite=True, output_verify='ignore')

## defringe flat
if args.nodefringe:
    # check if some non-defringe flat files are under the raw data folder
    if os.path.exists(datadir[0]+'/defringeflat_diagnostic/'):
        for data_name in glob.glob1(datadir[0]+'/defringeflat_diagnostic/', '*.fits'):
            if '_defringe.fits' not in data_name:
                shutil.move(datadir[0]+'/defringeflat_diagnostic/'+data_name, datadir[0]+data_name)
    if os.path.exists(datadir[0]+'/defringeflat/'):
        for data_name in glob.glob1(datadir[0]+'/defringeflat/', '*.fits'):
            if '_defringe.fits' not in data_name:
                shutil.move(datadir[0]+'/defringeflat/'+data_name, datadir[0]+data_name)
    # remove the defringe flat files in the current folder
    for data_name in glob.glob1(datadir[0], '*_defringe.fits'):
        os.system('rm {}/{}'.format(datadir[0], data_name))

else:
    # remove the defringe flat files in the previous reduction
    if os.path.exists(datadir[0]+'/defringeflat_diagnostic/'):
        for data_name in glob.glob1(datadir[0]+'/defringeflat_diagnostic/', '*.fits'):
            shutil.move(datadir[0]+'/defringeflat_diagnostic/'+data_name, datadir[0]+'/'+data_name)
    if os.path.exists(datadir[0]+'/defringeflat/'):
        for data_name in glob.glob1(datadir[0]+'/defringeflat/', '*.fits'):
            shutil.move(datadir[0]+'/defringeflat/'+data_name, datadir[0]+'/'+data_name)
    for data_name in glob.glob1(datadir[0], '*_defringe.fits'):
        os.system('rm {}/{}'.format(datadir[0], data_name))

    # run the defringe flat algorithm
    print("Defringe flat lamp files: {}".format(datadir[0]))
    smart.defringeflatAll(datadir[0], wbin=10, start_col=10, end_col=980, diagnostic=False, movefiles=True)

    defringe_list = glob.glob1(path,'*defringe.fits')
    originalflat_list = glob.glob1(path+'defringeflat/','*.fits')

## reduce the data using NSDRP
action = "python3" + " " + BASE + "nsdrp.py" + " " + datadir[0] + " " + datadir[1] + " " \
       + "-oh_filename" + " " + BASE + "/ir_ohlines.dat -debug -dgn"
if args.spatial_rect_flat:
    action += ' -spatial_rect_flat' #-spatial_jump_override
if args.sowc:
    action += ' -sowc'
if args.verbose:
    action += ' -verbose'

print("Executing:", action)
os.system(action)

## move the original flat files back
if args.nodefringe:
    pass
else:
    for defringeflatfile in defringe_list:
        shutil.move(path+defringeflatfile, path+'defringeflat/'+defringeflatfile)
    for originalflat in originalflat_list:    
        shutil.move(path+'defringeflat/'+originalflat, path+originalflat)

#print("Finish execution: {}".format(action))
