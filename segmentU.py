'''
Created on Jul 14, 2019

@author: nikolay
'''

from patchSampler import NailDataset

import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
import time
import random
import matplotlib.pyplot as plt
from networks import _NetUskip, weights_init
import numpy as  np
import torch.optim as optim
from options import opt
#savePath = "results/"

if opt.outf == 'results':
    print ("reset output folder")
    import datetime
    stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    opt.outf = opt.outf + "/UNET" + "_cross%s_"%(opt.Ucross)+stamp + "/"
import os
try:
    os.makedirs(opt.outf)
except OSError:
    pass
print ("outfolder " + opt.outf)

savePath=opt.outf

wh = opt.imageSize  # #resizes and eventually downsamples the cropped image slice, which should be larger usually
rbuf=[]
if opt.resizeAugment:
    #raise Exception('not yet supported')
    import PIL
    rbuf += [transforms.RandomRotation(180, resample=PIL.Image.BILINEAR,expand=False)]#
    rbuf += [transforms.CenterCrop(wh)]
rbuf += [transforms.RandomVerticalFlip(),transforms.RandomHorizontalFlip()]#flip still benefits from pixel set labels from alpha

#transforms.RandomVerticalFlip(),transforms.RandomHorizontalFlip(),
tbuf = [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]
transform = transforms.Compose(rbuf+tbuf)
dataset = NailDataset(transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers, drop_last=False)  # #simpler to drop in training?
print ("data augment train set",transform)

tbuf = [transforms.Resize(size=wh, interpolation=2),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]
transform = transforms.Compose([transforms.CenterCrop(wh)]+tbuf)#test set not mirrored, or rotated
print ("data augment test set",transform)
tdataset = NailDataset(transform=transform, train=False)#for test no random augmenting
tdataloader = torch.utils.data.DataLoader(tdataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)  # o drop here -- but random patch samplign anyway

print ("med data loader length, train", len(dataloader))
print ("med data loader length, test", len(tdataloader))
#raise Exception

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ("device", device)

ndf = opt.ndf
netD = nn.Sequential(_NetUskip(ndf, int(np.log2(wh))-4, ncIn=4 - 1),nn.Sigmoid())##output in 01
# 99 20 0.013797604478895664 0.05623283432505559 0.04545524820001447
netD.apply(weights_init)
print(netD)
netD = netD.to(device)

bce=nn.BCELoss()
##pred is prediction, lab is true label image
def criterion(pred,lab):
    err=0
    for b in range(pred.shape[0]):
        #print (b,lab[b].sum())
        if lab[b].sum()==0:
            mask = lab[b]*0+1
        else:
            mask=lab[b]*1
        mask/=mask.sum()#so sum to 1 always
        if not opt.Ucross:
            err += (mask*((pred[b]-lab[b])**2)).sum()
        else:
            err += (mask*bce(pred[b],lab[b])).sum()
    return err/pred.shape[0]#divida by batch

if False:
    opti = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
else:
    from ranger import Ranger
    opti = Ranger(netD.parameters())
    print ("training with ranger")

def valiScore():
    err = []
    bufP = []
    bufN = []
    bufME = []

    nTEV = 5  #how many dataset iterations for total error validation, more totally random patches, both slide and label
    with torch.no_grad():
        for it in range(nTEV):
            print ("vali",it,"/",nTEV)
            np.random.seed(it)
            random.seed(it)
            for i, data in enumerate(tdataloader, 0):
                im, lab = data
                lab = alpha_to_label(im)
                im = im[:, :3].to(device)  # trim alpha channel
                lab = lab.to(device).float()
                pred = netD(im)
                for b in range(pred.shape[0]):
                    err += [criterion(pred[b:b + 1, 0, 0, 0],lab[b:b+1]).item()]
                    if True:
                        j=b
                        if lab[j].sum() == 0:
                            bufN.append(pred[j].max().item())
                            bufME.append(pred[j].mean().item())
                        else:
                            bufP.append(pred[j].max().item())
    err = np.array(err).mean()
    print("total err validation",err)
    abufN = np.array(bufN)
    abufP = np.array(bufP)
    abufME = np.array(bufME)
    print("buffers", abufN.shape, abufP.shape,abufME.shape)
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 1, 1)
    plt.semilogy(buf, alpha=0.5)
    win = 50
    av = np.convolve(err, np.ones((win,)) / win, mode='valid')
    plt.semilogy(av, color='r', lw=3, alpha=0.7)
    plt.xlabel("ADAM step")
    plt.ylabel("L2 error")
    win = 50  ##more averaged
    abufP = np.convolve(abufP, np.ones((win,)) / win, mode='valid')
    abufN = np.convolve(abufN, np.ones((win,)) / win, mode='valid')
    plt.plot(abufP, color='g', lw=1, label='pos')
    plt.plot(abufN, color='c', lw=1, label='neg')
    abufME = np.convolve(abufME, np.ones((win,)) / win, mode='valid')
    plt.plot(abufME, color='m', lw=2, label='mean')
    plt.xlabel("patch sample")
    plt.ylabel("probability")
    plt.legend()
    plt.savefig("%s/VALIDATION_loss_px%s_cr%s_BN%s.png" % (savePath, opt.imageSize, opt.imageCrop, str(opt.BN)))
    return err

bufP = []#for patch prob per SGD step#keep these buffers small, otherwise too large for whole training
bufN = []
bufME = []
buf=[]#for average minibatch error, train set
bufTest=[]#for average error, test set - -called fewer times

def alpha_to_label(im):
    # read from overwritten alpha
    lab = (-im[:, 3:]) * 0.5 + 0.5  # transformed already to [-1 1] -- swap and map to [0 1]
    # lab[lab > 1e-10]=1
    lab /= lab + 1e-15  ##so that we make floats binary for pos annos
    return lab

# python segmentU.py --imageSize=256 --imageCrop=256 --ndf=64 --batchSize=16 --BN=True --PXA=True --resizeAugment=
if __name__ == '__main__':
    if False:##warm start
        name = '%s/model_px%s_cr%s.dat' % (savePath, opt.imageSize, opt.imageCrop)
        print("loading name", name)
        netD.load_state_dict(torch.load(name))

    #bufTest.append(valiScore())##just debug if routine ok
    for it in range(1000):
        print ("it",it,"len data",len(dataloader))
        #np.random.seed(it)
        #random.seed(it)

        bufN = bufN[-5000:]
        bufP = bufP[-5000:]##keep only last probs, avoid too much overload
        bufME = bufME[-5000:]
        for i, data in enumerate(dataloader, 0):
            if i >0:
                print ("total step",i, "iter time",time.time()-t0)
            t0=time.time()
            im, lab = data
            lab=alpha_to_label(im)

            if i%20==0:
                print ("img and alpha",lab.shape,im.shape,lab.dtype,lab.max(),lab.min(),lab.sum(),"orig alpha",im[:,3:].min(),im[:,3:].max())
            # print (im[:, 3].mean(), im[:, 3].min(), im[:, 3].max())
            im = im[:, :3].to(device)  # trim alpha channel

            if i==0 and it ==0:
                vutils.save_image(im*0.5+0.5, "%s/debugUnet.png" % (savePath),
                              normalize=False)
                vutils.save_image(lab, "%s/debugUnetLabel.png" % (savePath),
                                  normalize=False)
                print ("initial label",lab.sum(),lab.max(),lab.mean())

                plt.figure(figsize=(12, 12))
                sr=lab.view(-1)
                sr=torch.sort(sr)[0]
                print ("sr",sr.shape)
                plt.plot(sr[-1000:].numpy())
                plt.savefig("%s/labvalues_px%s_cr%s_BN%s.png" % (savePath, opt.imageSize, opt.imageCrop,str(opt.BN)))
                plt.close()
                #raise Exception

            lab = lab.to(device).float()  # not long but float...

            opti.zero_grad()
            pred = netD(im)
            err = criterion(pred, lab)

            err.backward()
            opti.step()

            buf.append(err.item())

            for j in range(lab.shape[0]):##single element, 1 number per patch
                if lab[j].sum()==0:
                    bufN.append(pred[j].max().item())
                    bufME.append(pred[j].mean().item())
                else:
                    bufP.append(pred[j].max().item())

            if i%20 ==0 and False:
                print("lab", lab.sum(3).sum(2), "pred", pred.sum(3).sum(2))
                print ("buf",len(bufN),len(bufP))

            if i%5==0:
                print ("step i",i,"err",np.array(buf)[-5:].mean())

            if (i == 0 or (it ==0 and i%30==0)) and len(buf) >3:
                vutils.save_image(im * 0.5 + 0.5, "%s/resUnet.png" % (savePath),
                                  normalize=False)
                vutils.save_image(torch.cat([lab,pred,pred*0+0.5],1), "%s/resUnetLabel.png" % (savePath),
                                  normalize=False)

                a = np.array(buf)
                abufN = np.array(bufN)
                abufP = np.array(bufP)
                abufME = np.array(bufME)
                print ("buffers",abufN.shape,abufP.shape)
                print ("iter", it, i, "training loss", err.item(), "-100 -200 lag average", a[-100:].mean(), a[-200:].mean())
                plt.figure(figsize=(12, 12))
                plt.subplot(3, 1,1)
                plt.semilogy(buf,alpha=0.5)
                win = 50
                av=np.convolve(a, np.ones((win,))/win, mode='valid')
                plt.semilogy(av,color='r',lw=3,alpha=0.7)
                plt.xlabel("ADAM step")
                plt.ylabel("L2")

                plt.subplot(3,1,2)
                win = 50##more averaged
                abufP = np.convolve(abufP, np.ones((win,)) / win, mode='valid')
                abufN = np.convolve(abufN, np.ones((win,)) / win, mode='valid')
                plt.plot(abufP, color='g', lw=1,label='pos')
                plt.plot(abufN, color='c', lw=1,label='neg')
                abufME = np.convolve(abufME, np.ones((win,)) / win, mode='valid')
                plt.plot(abufME, color='m', lw=2, label='mean')
                plt.xlabel("patch sample")
                plt.ylabel("probability")

                if len(bufTest)>1:
                    win = 20
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
                torch.save(netD.state_dict(), '%s/Umodel_px%s_cr%s_BN%s.dat' % (savePath, opt.imageSize, opt.imageCrop,str(opt.BN)))
            #try:
            #    showRandomFullSlide_Heatmap()
            #except Exception as e:
            #    print ("heatmap",e)
            #try:
            #    validate()
            #except Exception as e:
            #    print ("validate",e)

