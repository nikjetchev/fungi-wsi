'''
Created on Jul 14, 2019

@author: nikolay
'''

from patchSampler import NailDataset, getRandomUP, getSUP, getNUP,sanity

import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
import matplotlib.pyplot as plt

import time
import random

from networks import _netD, weights_init
import numpy as  np
import torch.optim as optim
from options import opt
#savePath = "results/"

if opt.outf == 'results':
    print ("reset output folder")
    import datetime
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    opt.outf = opt.outf + "/" + stamp + "/"
import os
try:
    os.makedirs(opt.outf)
except OSError:
    pass
print ("outfolder " + opt.outf)

savePath=opt.outf

wh = opt.imageSize  # #resizes and eventually downsamples the cropped image slice, which should be larger usually
# transforms.RandomCrop(wh),
# transforms.Resize(size=wh, interpolation=2)
#DONE augment with random mirror H and W
#augment with random rotate
rbuf=[]
if opt.resizeAugment:
    rbuf +=[transforms.RandomRotation(180, resample=False, expand=False)]
    rbuf +=[transforms.CenterCrop(wh)]

tbuf = [transforms.Resize(size=wh, interpolation=2),transforms.RandomVerticalFlip(),transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]
transform = transforms.Compose(rbuf+tbuf)
dataset = NailDataset(transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers, drop_last=False)  # #simpler to drop in training?
print ("data augment train set",transform)

tbuf = [transforms.Resize(size=wh, interpolation=2),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]
ttransform = transforms.Compose([transforms.CenterCrop(wh)]+tbuf)#test set not mirrored, or rotated
print ("data augment test set",ttransform)
tdataset = NailDataset(transform=ttransform, train=False)#for test no random augmenting
tdataloader = torch.utils.data.DataLoader(tdataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)  # o drop here -- but random patch samplign anyway

print ("med data loader length, train", len(dataloader))
print ("med data loader length, test", len(tdataloader))
#raise Exception

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ("device", device)

ndf = opt.ndf
netD = _netD(ndf, int(np.log2(wh)), nc=4 - 1)
# 99 20 0.013797604478895664 0.05623283432505559 0.04545524820001447
netD.apply(weights_init)
print(netD)
netD = netD.to(device)

if False:
    criterion = nn.BCELoss()
else:
    bce=nn.BCELoss()
    ##pred is prediction, lab is true label image
    def criterion(pred,lab):
        err=0
        for b in range(pred.shape[0]):
            #print (b,lab[b].sum())
            if lab[b].sum()==0:#neg
                mask = lab[b]*0+opt.NWeight#10#more weight for such errors
            else:
                mask=lab[b]
            err += (mask*bce(pred[b],lab[b]))#already squeezed
        return err/pred.shape[0]

## weight criterion more!!

if False:
    opti = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
else:
    from ranger import Ranger
    opti = Ranger(netD.parameters())
    print ("training with ranger")

def valiScore():
    err = []
    nTEV = 10  #how many dataset iterations for total error validation, more totally random patches, both slide and label
    with torch.no_grad():
        for it in range(nTEV):
            print ("vali",it,"/",nTEV)
            np.random.seed(it)
            random.seed(it)
            for i, data in enumerate(tdataloader, 0):
                im, lab = data
                im = im[:, :3].to(device)  # trim alpha channel
                lab = lab.to(device).float()
                pred = netD(im)
                for b in range(pred.shape[0]):
                    err += [criterion(pred[b:b + 1, 0, 0, 0],lab[b:b+1]).item()]
    err = np.array(err).mean()
    print("total err validation",err)
    return err

bufP = []#for patch prob per SGD step#keep these buffers small, otherwise too large for whole training
bufN = []
buf=[]#for average minibatch error, train set
bufTest=[]#for average error, test set - -called fewer times

if opt.hardneg:
    import pickle
    hname = "resultsVali/2020-03-26_10-48-14_factor10_vtrainfull/" + "top11445.dat"
    hn=pickle.load(open(hname,'rb'))
    print ("hdic",len(hn))
    br=0
    for n in list(hn.keys()):
        if len(hn[n]) ==0:##delete empty nslide
            print ("empty",n)
            del hn[n]
            continue
        d2=[]
        for p, x, y in hn[n]:
            if p >0.2:
                d2.append((p,x,y))
        hn[n]+=d2
        br+=len(d2)
    print ("add 0.2",br)
    print("hdic", len(hn))
    lhn=list(hn.keys())

def sampleHN(N=32):
    def getName(n):
        i = n.find("N")
        n = n[i:]
        i = n.find(".tif")
        return n[:i]

    from options import defaultPath
    path = defaultPath + "wsi/"

    import openslide
    from patchSampler import centerPatch

    img=[]
    label=[]

    for i in range(N):
        ix= np.random.randint(len(lhn))
        n=lhn[ix]
        name = path + getName(n)+".tif"
        ix=np.random.randint(len(hn[n]))##probable neg patches
        p,x,y=hn[n][ix]
        slide = openslide.open_slide(name)
        cp= centerPatch(slide,x,y)
        slide.close()
        img.append(transform(cp).to(device)[:3].unsqueeze(0))
        label.append(torch.zeros(1).to(device).float())
        ##TODO proportional sampler to p?
    #print ("list",len(img),len(label))
    img = torch.cat(img)
    label = torch.cat(label)

    return img,label

def appendHN(i,l,ih,lh):
    #print (i.shape,l.shape)
    #print(ih.shape,lh.shape)
    return torch.cat([i,ih]),torch.cat([l,lh])


    ##usage python classify.py --imageSize=128 --imageCrop=1024
if __name__ == '__main__':
    if len(opt.loadPath) >0:##warm start
        loadPath = opt.loadPath
        name = "%s/model_px%s_cr%s_BN%s.dat" % (loadPath, opt.imageSize, opt.imageCrop,str(opt.BN))
        print("loading name", name)
        netD.load_state_dict(torch.load(name))

    for it in range(1000):
        if it ==0 and False:#sanity check debug
            valiScore()

        print ("it",it,"len data",len(dataloader))
        np.random.seed(it)
        random.seed(it)

        bufN = bufN[-2000:]
        bufP = bufP[-2000:]##keep only last probs, avoid too much overload

        for i, data in enumerate(dataloader, 0):
            if i >0:
                print ("total step",i, "iter time",time.time()-t0)
            t0=time.time()
            im, lab = data
            # print (im[:, 3].mean(), im[:, 3].min(), im[:, 3].max())
            im = im[:, :3].to(device)  # trim alpha channel

            lab = lab.to(device).float()  # not long but float...

            if opt.hardneg:
                ih,lh=sampleHN()
                im,lab=appendHN(im,lab,ih,lh)
                if i==0 or i ==1:
                    print ("expanded batch",im.shape)

            if i==0 and it ==0:
                vutils.save_image(im*0.5+0.5, "%s/debugRotate.png" % (savePath),normalize=False)

            opti.zero_grad()
            pred = netD(im).squeeze()
            err = criterion(pred, lab)

            err.backward()
            opti.step()

            buf.append(err.item())
            for j in range(lab.shape[0]):##single element, 1 number per patch
                if lab[j]==0:
                    bufN.append(pred[j])
                else:
                    bufP.append(pred[j])

            if i%5==0:
                print ("step i",i,"err",np.array(buf)[-5:].mean())

            win = 50
            if i == 0 and len(buf) >3 and len(bufTest)>1:
                a = np.array(buf)
                abufN = np.array(bufN)
                abufP = np.array(bufP)
                print ("iter", it, i, "training loss", err.item(), "-100 -200 lag average", a[-100:].mean(), a[-200:].mean())
                plt.figure(figsize=(12, 12))
                plt.subplot(3, 1,1)
                plt.semilogy(buf,alpha=0.5)
                av=np.convolve(a, np.ones((win,))/win, mode='valid')
                plt.semilogy(av,color='r',lw=3,alpha=0.7)
                plt.xlabel("ADAM step")
                plt.ylabel("cross entropy error")

                plt.subplot(3,1,2)
                win = 50##more averaged
                abufP = np.convolve(abufP, np.ones((win,)) / win, mode='valid')
                abufN = np.convolve(abufN, np.ones((win,)) / win, mode='valid')
                plt.plot(abufP, color='g', lw=1,label='pos')
                plt.plot(abufN, color='c', lw=1,label='neg')
                plt.xlabel("patch sample")
                plt.ylabel("probability")

                win=20
                plt.subplot(3, 1, 3)
                plt.semilogy(bufTest, lw=1, alpha=0.5)
                abufT = np.convolve(np.array(bufTest), np.ones((win,)) / win, mode='valid')
                plt.semilogy(abufT, lw=3, alpha=0.7)
                plt.xlabel("test step")
                plt.ylabel("cross entropy error")

                plt.legend()


                plt.savefig("%s/loss_px%s_cr%s_BN%s.png" % (savePath, opt.imageSize, opt.imageCrop,str(opt.BN)))
                plt.close()

            print ("SGD time", time.time() - t0)
            t0 = time.time()#to measure hust iter

        print ("eval with net.eval()")
        netD.eval()
        bufTest.append(valiScore())
        netD.train()

        if it % 20 == 1:# and it >4:
            if it % 40 == 1:
                torch.save(netD.state_dict(), '%s/model_px%s_cr%s_BN%s.dat' % (savePath, opt.imageSize, opt.imageCrop,str(opt.BN)))


            #try:
            #    showRandomFullSlide_Heatmap()
            #except Exception as e:
            #    print ("heatmap",e)
            #try:
            #    validate()
            #except Exception as e:
            #    print ("validate",e)

