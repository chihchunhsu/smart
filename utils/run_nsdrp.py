## rountine for reducing the NIRSPEC data using the modified NSDRP
## Check KOA keywords --> defringe --> reduce data

import smart
from astropy.io import fits
import os
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

parser = argparse.ArgumentParser(description="Reduce the NIRSPEC data using NIRSPEC-Data-Reduction-Pipeline",\
	usage="run_nsdrp.py input_dir (output_dir)")

parser.add_argument("files",metavar='f',type=str,
    default=None, help="input_dir (output_dir)", nargs="+")

parser.add_argument("--nodefringe", 
    help="not apply the defringe algorithm", 
    action='store_true')

args = parser.parse_args()
datadir  = args.files

if len(datadir) is 1:
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

print("Checking the keyword formats...")
for filename in mylist:
    #print(filename)
    file_path = path + filename
    data, header = fits.getdata(file_path, header=True, ignore_missing_end=True)
    if ('IMAGETYP' in header) is False:
        if ('flat lamp off     ' in str(header['COMMENT'])) is True:
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
    pass
else:
    print("Defringing flat files...")
    smart.defringeflatAll(datadir[0], wbin=10, start_col=10, end_col=980, diagnostic=False, movefiles=True)

    defringe_list = glob.glob1(path,'*defringe.fits')
    originalflat_list = glob.glob1(path+'defringeflat_diagnostic/','*.fits')

## reduce the data using NSDRP
print("Start reducing the data by the NSDRP...")
os.system("python" + " " + BASE + "nsdrp.py" + " " + datadir[0] + " " + datadir[1] + " " \
	+ "-oh_filename" + " " + BASE + "/ir_ohlines.dat -spatial_jump_override -debug -dgn")

## move the original flat files back
if args.nodefringe:
    pass
else:
    for defringeflatfile in defringe_list:
        shutil.move(path+defringeflatfile, path+'defringeflat_diagnostic/'+defringeflatfile)
    for originalflat in originalflat_list:    
        shutil.move(path+'defringeflat_diagnostic/'+originalflat, path+originalflat)
## remove the intermediate products
#os.system("rm *.npy")

print("The NIRSPEC data are reduced successfully by using NSDRP.")
