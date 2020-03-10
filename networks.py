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

class _netD_CGAN(nn.Module):
    def __init__(self, ndf, nDep, nc=3, bSigm=True, bOut1=True):
        super(_netD_CGAN, self).__init__()
        layers = []
        of = nc
        for i in range(nDep):
            if False:
                nf = 1
            else:
                nf = min(ndf * 2 ** i, 768)
            layers += [nn.Conv2d(of, nf, 5, 2, 2)]  # #needs input 161 #hmm, also worls loke tis
            if i != 0 and i != nDep - 1:
                layers += [norma(nf)]

            if False and i < nDep - 1:
                layers += [nn.LeakyReLU(0.2, inplace=True)]
            else:
                layers += [nn.Sigmoid()]
            of = nf
        self.main = nn.Sequential(*layers)
        self.P = nn.Parameter(torch.ones(2,nf,1,1).uniform_(-0.05,0.05))#2x512 embeddings

    def forward(self, input,labels):
        output = self.main(input)
        #pos = labels==1
        #neg = labels == 0
        #print ("pl",self.P[labels].shape,self.P[labels][:,0,0,0])#6x512
        #print ("raw",self.P[:,0])
        return (self.P[labels]*output).sum(1)
        return torch.cat([P,N])#so B x HxW
        return output

# #i.e. the decoder
class _netG(nn.Module):
    def __init__(self, ngf, nDep, Nbottle,nc=3):
        super(_netG, self).__init__()
        layers = []
        # #first nDep layers
        of = Nbottle+int(opt.CGAN)  # nz+NCbot
        for i in range(nDep):
            if i == nDep - 1:
                nf = nc
            else:
                nf = min(762,ngf * 2 ** (nDep - 2 - i))
            layers += [nn.Upsample(scale_factor=2, mode='nearest')]  # #this nearst is deafault anyway
            layers += [nn.Conv2d(of, nf, 4 + 1, 1, 2)]  # #auu, much more memory!!!  also other kernel when doung conv2d ?
            if i == nDep - 1:
                if False:
                    layers += [nn.Tanh()]
            else:
                if True:  # i!=0:##so better effect when having single global noise dim
                    layers += [norma(nf)]  # #every layer batch norm in A Radford
                layers += [nn.ReLU(True)]
            of = nf
        self.main = nn.Sequential(*layers)

    def forward(self, input,labels=None):
        if opt.CGAN and labels is not None:
            input=torch.cat([input,labels.view(-1,1,1,1).expand(input[:,:1].shape).float()],1)#.repeat(1,1,input.shape[2],input.shape[3])]
        output = self.main(input)
        return output

def setNoise(noise,supressR=False):
    nGL = opt.nGL
    noise = 1.0 * noise
    if not supressR:
        noise.uniform_(-1, 1)
    assert(nGL > 0)
    if nGL:
        noise[:, :nGL] = noise[:, :nGL, :1, :1].repeat(1, 1, noise.shape[2], noise.shape[3])
    return noise

class _NetUskip(nn.Module):
    # @param ncOut is output channels
    # @param ncIn is input channels
    # @param ngf is channels of first layer, doubled up after every stride operation, or halved after upsampling
    # @param nDep is depth, both of decoder and of encoder
    # @param nz is dimensionality of stochastic noise we add
    # @param NCbot can optionally specify the bottleneck size explicitly
    # @param bSkip turns skip connections btw encoder and decoder off
    # @param bTanh turns nonlinearity on and off at final output
    def __init__(self, ngf, nDep, nz=0, Ubottleneck=0, ncOut=1, ncIn=3, bSkip=True, bTanh=False):
        super(_NetUskip, self).__init__()
        self.nDep = nDep
        self.eblocks = nn.ModuleList()
        self.dblocks = nn.ModuleList()
        self.bSkip = bSkip

        if ncIn is None:
            of = ncOut
        else:
            of = ncIn  ##in some cases not an RGB conditioning

        MX = 512

        for i in range(self.nDep):
            layers = []
            if i == self.nDep - 1 and Ubottleneck>0:
                nf = Ubottleneck
            else:
                nf = min(MX,ngf * 2 ** i)
            layers += [nn.Conv2d(of, nf, 5, 2, 2)]
            if i != 0:
                layers += [norma(nf)]
            if i < self.nDep - 1:
                layers += [nn.LeakyReLU(0.2, inplace=True)]
            else:
                layers += [nn.Tanh()]#tanh at bottleneck?
            of = nf
            block = nn.Sequential(*layers)
            self.eblocks += [block]

        ##first nDep layers
        if Ubottleneck>0:
            of = nz + Ubottleneck

        for i in range(nDep):
            layers = []
            if i == nDep - 1:
                nf = ncOut
            else:
                nf = min(MX,ngf * 2 ** (nDep - 2 - i))
            print ("unet",of,nf)
            if i > 0 and self.bSkip:
                of *= 2  ##the u skip connections
            layers += [nn.Upsample(scale_factor=2, mode='nearest')]  # nearest is default anyway
            layers += [nn.Conv2d(of, nf, 5, 1, 2)]
            if i == nDep - 1:
                if bTanh:
                    layers += [nn.Tanh()]
            else:
                layers += [norma(nf)]
                layers += [nn.ReLU(True)]
            of = nf
            block = nn.Sequential(*layers)
            self.dblocks += [block]

    def forward(self, input1, input2=None):
        x = input1  ##initial input
        skips = []
        for i in range(self.nDep):
            x = self.eblocks[i].forward(x)
            if i != self.nDep - 1:
               skips += [x]
        bottle = x
        if input2 is not None:
            bottle = torch.cat((x, input2), 1)  ##the det. output and the noise appended
        x = bottle
        for i in range(len(self.dblocks)):
            x = self.dblocks[i].forward(x)
            if i < self.nDep - 1 and self.bSkip:
                x = torch.cat((x, skips[-1 - i]), 1)
        return x-10##prior for 0