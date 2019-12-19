from __future__ import print_function
import os
import random
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from networks import _netG, _netD, weights_init,setNoise
import torchvision.transforms as transforms
import torchvision.utils as vutils
from patchSampler import NailDataset
import torch.nn.functional as F

from options import opt
if opt.CGAN:
    from networks import _netD_CGAN as _netD

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ("device", device)  # input=input.to(device)
bRec = opt.fRec > 0

import datetime
stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
opt.outf = opt.outf+ "/GAN" + stamp + "/"
try:
    os.makedirs(opt.outf)
except OSError:
    pass
print ("outfolder " + opt.outf)

cudnn.benchmark = True


import numpy as np
import matplotlib
matplotlib.use('Agg')  # #when not having displac variable
import matplotlib.pyplot as plt

wh = opt.imageSize
rbuf=[]
if opt.resizeAugment:
    rbuf +=[transforms.RandomRotation(180, resample=False, expand=False)]
    rbuf +=[transforms.CenterCrop(wh)]
tbuf = [transforms.Resize(size=wh, interpolation=2),transforms.RandomVerticalFlip(),transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]
transform = transforms.Compose(rbuf+tbuf)
dataset = NailDataset(transform=transform)

print ("inited live dataset")
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=1,drop_last=True)

ngf = int(opt.ndf)
ndf = int(opt.ndf)
nz = opt.nz
nGL = opt.nGL
nDep = opt.nDep

##TODO for condition - -flag? opt.CGAN
# extra dim in d and g
# extra proj. loss in D -- parameters,
# use labels from iterator
#optional: move to softplus loss
#DONE get rid of tanh generate -- just conv?
#TODO caching in iterator...

netD = _netD(ndf, nDep)
netD.apply(weights_init)
print(netD)

netG = _netG(ngf, nDep, nz)
netG.apply(weights_init)
print(netG)

if False:#opt.load != '':
    name = opt.load
    netG.load_state_dict(torch.load(name))
    print ("loaded from file", name)
    # #hmm,deteriorate first, store ADAM or netD ?

class _netR(nn.Module):
    def __init__(self, ndf=ndf, Ctype=1, nc=3):
        super(_netR, self).__init__()
        layers = []
        of = nc
        for i in range(nDep):
            if i == nDep - 1 and False:
                nf = nGL
            else:
                nf = ndf * 2 ** i
            if Ctype == 1:
                layers += [nn.Conv2d(of, nf, 5, 2, 2)]  # #needs input 161 #hmm, also worls loke tis
            else:
                layers += [nn.Conv2d(of, nf, 4, 2, 1)]  # #todo try ker5 pad2

            print("layersD" + str(of) + ";" + str(nf))
            if i != 0 and i != nDep - 1:
                layers += [nn.BatchNorm2d(nf)]

            if i < nDep - 1:
                layers += [nn.LeakyReLU(0.2, inplace=True)]
            else:
                pass
                # layers+=[nn.Tanh()]
            of = nf
        self.main = nn.Sequential(*layers)
        self.final = nn.Conv2d(of, nGL, 1)

    def forward(self, input):
        output = self.main(input)
        output = output.mean(3).mean(2).unsqueeze(2).unsqueeze(3)  # #single spatial dim
        return self.final(output)

if opt.CGAN:
    def dummy(val, label):
        l = 1 - 2 * label
        return torch.mean(torch.nn.functional.softplus(l * val))
    criterion = dummy
    print ("criteria softplus")
else:
    criterion = nn.BCELoss()  # to device?

NZ = opt.imageSize // 2 ** nDep  # 3#TODO careful function to align witfor i, data in enumerate(dataloader, 0):h padding logic    opt.imageSize/2**5
noise = torch.FloatTensor(opt.batchSize, nz, NZ, NZ)

label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

noise = noise.to(device)
noise = setNoise(noise)  # #no GL audio characteer first
print ("first noise", noise.shape)
netD = netD.to(device)
netG = netG.to(device)

if bRec:
    netR = _netR()
    print("audio rec", netR)
    netR.apply(weights_init)
    netR = netR.to(device)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
if bRec:
    optimizerG = optim.Adam(list(netG.parameters()) + list(netR.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
else:
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

assert(opt.BN)

import sys
buf = []
for epoch in range(opt.niter):
    sys.stdout.flush()
    for i, data in enumerate(dataloader, 0):
        text, labels = data
        labels=labels.to(device)
        ##TODO add class label as 1 extra channel to G and D,
        text = text[:,:3].to(device)
        noise = setNoise(noise)

        optimizerD.zero_grad()
        output = netD(text,labels)  # train with real
        #output=F.sigmoid(output)
        errD_real = criterion(output, output.detach() * 0 + real_label)
        errD_real.backward()
        D_x = output.mean().item()
        outputTrue=1.0*output
        if True:
            with torch.no_grad():
                fake = netG(noise,labels)  # train with fake
            output = netD(fake.detach(),labels)
            errD_fake = criterion(output, output.detach() * 0 + fake_label)
            errD_fake.backward()
        D_G_z1 = output.mean().item()
        errD = errD_real + errD_fake
        if opt.CGAN:
            output_s = netD(text, 1-labels)  # train with real
            #output_s = F.sigmoid(output_s)
            errD_swap = criterion(output_s, output.detach() * 0 + fake_label)
            errD_swap.backward()
            if np.random.rand()<-0.1:
                print(labels,"CGAN real and swap",outputTrue[:1,2,2],output_s[:1,2,2])# F.sigmoid(output).mean(),F.sigmoid(output_s).mean())  ##av.prob
                print ("loss",errD.item(),errD_swap.item())
        optimizerD.step()
        optimizerG.zero_grad()
        noise = setNoise(noise)
        fake = netG(noise,labels)  # train with fake -- create again
        output = netD(fake,labels)
        if bRec:
            recZ = netR(fake)  # #is detach important, suspect some bug with netR -- no detach, netG should adapt tohether with netR
            # errR = (((recZ - noise[:, :nGL]).mean(3).mean(2)) ** 2).mean()
            errR = (((recZ - noise[:, :nGL, :1, :1])) ** 2).mean()
        else:
            errR = noise.sum() * 0
        errG = criterion(output, output.detach() * 0 + real_label) + opt.fRec * errR
        errG.backward()
        D_G_z2 = output.mean().item()
        optimizerG.step()
        buf += [[D_x, D_G_z1, D_G_z2, errR.item()]]

        #########################################

        if i % 20 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f content %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2, errR))
            print ("error averages", np.array(buf)[-50:].mean(0))
        if i % 100 == 0:
            vutils.save_image(text * 0.5 + 0.5, '%s/real_textures.jpg' % opt.outf, normalize=False)
            vutils.save_image(fake * 0.5 + 0.5, '%s/tex_epoch%03d_GL%d_%d.jpg' % (opt.outf, epoch, nGL, ngf), normalize=False)

            with torch.no_grad():
                netG.eval()
                n2 = noise[:1].repeat(8, 1, 3, 3)#so same local noise, diff global
                n2 = setNoise(n2)
                labels=labels[:8]#hack for 8 labels , to be overwrotten
                print ("n2",labels.shape,n2.shape)
                fake2 = netG(n2,labels*0)
                vutils.save_image(fake2 * 0.5 + 0.5, '%s/tex2_%03d_GL%d_%d.jpg' % (opt.outf, epoch, nGL, ngf), normalize=False)
                if opt.CGAN:
                    fake2 = netG(n2, labels * 0+1)
                    vutils.save_image(fake2 * 0.5 + 0.5, '%s/tex2_pos_%03d_GL%d_%d.jpg' % (opt.outf, epoch, nGL, ngf),
                                  normalize=False)
                netG.train()
            if i == 0:
                torch.save(netG.state_dict(), '%s/fungi_modelngf%d_dep%d.dat' % (opt.outf, ngf, nDep))

# TODO idea Duncan: rotate or permute dimensions as inference time
