'''
Created on Jul 15, 2019

@author: nikolay
'''

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--hardneg', type=bool, default=False,help='whether to load dict of hardneg')

parser.add_argument('--NWeight', type=int, default=1, help='weight for negative class')

parser.add_argument('--increaseRes', type=int, default=1,help='denser grid for predict,  0 and 1 and 2 and 4,for slide class, any number for save heatmap')

parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=256, help='resizes crop to that size, the height / width of the input image to network,needs to be power of 2 for full network depth')
parser.add_argument('--imageCrop', type=int, default=256, help="crop from slide, larger than image size")
# parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=128)
parser.add_argument('--BN', type=bool, default=False, help='batchnorm')

parser.add_argument('--VTrain', type=bool, default=False, help='if True, vali plots will be on training set, not test set')

parser.add_argument('--resizeAugment', type=bool, default=True, help='if True, random rotate of data, 2x window taken and  then cropped back')

parser.add_argument('--outf', type=str, default="results")

parser.add_argument('--loadPath', type=str, default="None")


parser.add_argument('--Woff', type=int, default=0)
parser.add_argument('--niter', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--workers', type=int, default=0)
parser.add_argument('--WOS', type=int, default=1000, help="patches to sample for validation statistics, determines systemic grid size")

##new stuff for GAN
parser.add_argument('--nz', type=int, default=50)  # #local are this minus gl
parser.add_argument('--nGL', type=int, default=10, help='global noise dimensions')
parser.add_argument('--nDep', type=int, default=5, help='depth of GAN model')
parser.add_argument('--fRec', type=float, default=0.0, help='gan reconstruct?')
parser.add_argument('--CGAN', type=bool, default=False, help='CGAN')

##stuf for pixelaccurate unet
parser.add_argument('--PXA', type=bool, default=False, help='if True give back pixel annotation')
parser.add_argument('--Ucross', type=bool, default=False, help='if True cross entropy for  segment')

opt = parser.parse_args()
print(opt)

#defaultPath="/home/nikolay/data_ssd/nagelpilz_p150/"#hmm, same speed? very weird...
defaultPath="/mnt/slowdata1/nagelpilz_p150/"#annos_xml/
#defaultPath="/mnt/slowdata1/nagelpilz_p150/" # 17 sec for 256px fullGrid() from patchSampler heatmap 5 rows