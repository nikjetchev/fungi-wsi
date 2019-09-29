'''
Created on Jul 14, 2019

@author: nikolay
'''

import torch
import torch.nn as nn
from options import opt

if opt.BN:
    norma = nn.BatchNorm2d
else:
    norma = lambda nf:nn.InstanceNorm2d(nf, affine=True)
#TODO try batchnorm, eval when necessary

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# #TODO get proper CBlock33 and Cblock5 implementations
class _netD(nn.Module):
    def __init__(self, ndf, nDep, nc=3, bSigm=True, bOut1=True):
        super(_netD, self).__init__()
        layers = []
        of = nc
        for i in range(nDep):
            if i == nDep - 1 and bOut1:
                nf = 1
            else:
                nf = min(ndf * 2 ** i, 768)
            layers += [nn.Conv2d(of, nf, 5, 2, 2)]  # #needs input 161 #hmm, also worls loke tis
            if i != 0 and i != nDep - 1:
                layers += [norma(nf)]

            if i < nDep - 1:
                layers += [nn.LeakyReLU(0.2, inplace=True)]
            else:
                layers += [nn.Sigmoid()]
            of = nf
        self.main = nn.Sequential(*layers)

    def forward(self, input):
        output = self.main(input)
        return output
