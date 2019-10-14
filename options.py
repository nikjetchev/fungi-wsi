'''
Created on Jul 15, 2019

@author: nikolay
'''

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--batchSize', type=int, default=128, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='resizes crop to that size, the height / width of the input image to network,needs to be power of 2 for full network depth')
parser.add_argument('--imageCrop', type=int, default=1024, help="crop from slide, larger than image size")
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
opt = parser.parse_args()
print(opt)

defaultPath="/home/nikolay/data_ssd/nagelpilz_p150/"#hmm, same speed? very weird...
#defaultPath="/mnt/slowdata1/nagelpilz_p150/" # 17 sec for 256px fullGrid() from patchSampler heatmap 5 rows