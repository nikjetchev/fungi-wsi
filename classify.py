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
savePath = "results/"

wh = opt.imageSize  # #resizes and eventually downsamples the cropped image slice, which should be larger usually
# transforms.RandomCrop(wh),
# transforms.Resize(size=wh, interpolation=2)
##TODO random mirror H and W
tbuf = [transforms.Resize(size=wh, interpolation=2),transforms.RandomVerticalFlip(),transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]
transform = transforms.Compose(tbuf)
dataset = NailDataset(transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers, drop_last=False)  # #simpler to drop in training?

tbuf = [transforms.Resize(size=wh, interpolation=2),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]
transform = transforms.Compose(tbuf)
tdataset = NailDataset(transform=transform, train=False)#for test no random augmenting
tdataloader = torch.utils.data.DataLoader(tdataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)  # o drop here -- but random patch samplign anyway

print ("med data loader length", len(dataloader))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ("device", device)

ndf = opt.ndf
netD = _netD(ndf, int(np.log2(wh)), nc=4 - 1)
# 99 20 0.013797604478895664 0.05623283432505559 0.04545524820001447
netD.apply(weights_init)
print(netD)
netD = netD.to(device)

criterion = nn.BCELoss()

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

#statistics on test set, using same distributions as on training set
#also saves random patches from each class
def validate():
    err = []
    pl = []  # predict and label

    visImg = [[], [], []]
    nVI = 20*20  # count of patches for auxilliary visualization routine

    nTEV = 3  #how many dataset iterations for total error validation, more totally random patches, both slide and label
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
                    pl += [[pred[b,0,0,0].item(), lab[b].item()]]
                    err += [criterion(pred[b:b + 1, 0, 0, 0], lab[b:b + 1]).item()]

                    l = lab[b].item()#save image patch example in list for type: pos or neg
                    #if len(visImg[int(l)]) < nVI:
                    visImg[int(l)].append((pred[b,0,0,0].item(),im[b:b + 1].cpu()))

    pl = np.array(pl)
    print("total err validation", np.array(err).mean(), pl.shape)
    unlabeledP = sampleUnlabeledPos(tdataset, visImg, nVI, pl)##unlabelled postiive, also fills inplace list visIMG

    # #now grids with img, buffer filled for vis
    for l in range(3):
        visL = visImg[l]#all images for one category, a list, also prediction probabilitie<
        visL.sort(key=lambda x: x[0],reverse = True)
        top=[]#keep highest scored patches
        for prob,im in visL[:nVI]:
            top.append(im)

        #if l==1:
        #    for i in range(len(visL)-1):
        #        d=(visL[i][1]-visL[i+1][1]).abs().mean()
        #        print (l,i,"dist",d)
        buf = torch.cat(top) * 0.5 + 0.5
        print("class examples", buf.shape, buf.max(), buf.min())
        vutils.save_image(buf, '%s/class%s_%spx_cr%s.jpg' % (savePath, l, opt.imageSize, opt.imageCrop),
                          normalize=False)

    #uses the  training distribution, equal prob. for true and false class
    visualizeEqualProb(unlabeledP, pl)
    #uses proper spatial frequency, so many more blank patches than sick ones
    validateSpatialGridSampling(tdataset)
    ##TODO visualize full grid model -- once happy with performance, save filename and prediction

# @param pl are positive patches sampled from test set
# @output unlabeledP patches -- from positive slides without exact annotations, same count as PL
# random patches sampled
# visImg is just visualization buffer
# #TODO speedup data loader
##TODO refactor codE: ugly that labeled POS comes from dataloader, and unlabeledPOS is manually iterated
def sampleUnlabeledPos(dataset, visImg, nVI, pl):
    unlabeledP = []  # #TODO speed-up by batching..
    N = (pl.shape[0] // 2 // opt.batchSize) * opt.batchSize
    print ("unlabeled to get",N,pl.shape[0],opt.batchSize)
    # N = 20 * opt.batchSize
    buf = []
    for i in range(N):  # count points: same as other 2 classes, fifty fifty distribution in pl
        img = getRandomUP(dataset,count=1)[0]
        # #get single image from list
        img = img.to(device)
        #if len(visImg[2]) < nVI:
        #visImg[2].append(img)

        buf.append(img)
        if len(buf) == opt.batchSize:
            with torch.no_grad():
                pred = netD(torch.cat(buf))
            for b in range(pred.shape[0]):
                unlabeledP.append([pred[b,0,0,0].item(), 2])
                visImg[2].append((pred[b,0,0,0].item(),buf[b].cpu()))
            buf = []##resets buffer

    unlabeledP = np.array(unlabeledP)
    print ("unlabelled positive patches", unlabeledP.shape, img.shape, pred.shape, unlabeledP.mean(0))
    return unlabeledP

## plot scatter of patch probabilities
## plot AUC
def visualizeEqualProb(unlabeledP, pl):
    plt.figure(figsize=(10, 10))
    plt.scatter(pl[:, 0], pl[:, 1] + np.random.rand(pl.shape[0]) * 0.1, alpha=0.1, s=50)
    plt.scatter(unlabeledP[:, 0], unlabeledP[:, 1] + np.random.rand(unlabeledP.shape[0]) * 0.1, alpha=0.1, s=50)
    plt.savefig("%s/probs_%spx_cr%s.png" % (savePath,opt.imageSize, opt.imageCrop))
    plt.close()
#
    plt.figure(figsize=(10, 10))
    import sklearn.metrics as metrics

    fpr, tpr, threshold = metrics.roc_curve(pl[:, 1], pl[:, 0])  # #labels are first argument
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("%s/AUC_%spx_cr%s.png" % (savePath,opt.imageSize, opt.imageCrop))
    print ("AUC", roc_auc)

    # #histograms
    plt.figure(figsize=(10, 10))
    neg = pl[pl[:, 1] == 0, 0]
    plt.plot(np.sort(unlabeledP[:, 0]), label='patch POS')
    plt.plot(np.sort(neg), label='patch NEG')
    # plt.hist(unlabeledP[:, 0], bins=100, alpha=0.5, label='P')
    # plt.hist(neg, bins=100, alpha=0.5, label='N')
    plt.legend(loc='upper left')
    plt.xlabel("lowest rank")
    plt.ylabel("prediction p")
    plt.savefig("%s/PatchH_%spx_cr%s.png" % (savePath,opt.imageSize, opt.imageCrop))

    plt.figure(figsize=(10, 10))
    neg = pl[pl[:, 1] == 0, 0]
    cup = np.cumsum(-np.sort(-unlabeledP[:, 0]))
    cun = np.cumsum(-np.sort(-neg))
    plt.plot(cup, label='patch POS')
    plt.plot(cun, label='patch NEG')
    plt.legend(loc='upper left')
    plt.xlabel("highest rank")
    plt.ylabel("sum of ranked p per patch")
    plt.savefig("%s/PatchC_%spx_cr%s.png" % (savePath,opt.imageSize, opt.imageCrop))

    print ("cumulp cumuln", cup.mean(), cun.mean(), cup.mean() - cun.mean())

from patchSampler import fullGrid
import scipy.misc
import torch.nn.functional as F
def showRandomFullSlide_Heatmap():
    def save(name):
        #sanity(name)##will save thumbnail, make sure it looks ok
        buf,N0,N1 = fullGrid(name,tdataset.transform)
        smallI=[]##image much resized than original slide, further factor after the imgSize/imgCrop factor

        heatmap=[]
        for i in range(N1):
            patches=buf[i*N0:i*N0+N0]
            patches=torch.cat(patches).to(device)
            with torch.no_grad():
                    pred = netD(patches)
                    heatmap.append(pred.view(1,-1))##a row
                    I = F.interpolate(patches, size=(patches.shape[2] // 4, patches.shape[3] // 4), mode='bilinear')##make smaller
                    smallI.append(I.cpu())
        name = name[len(dataset.img_path):-4]##for save last part just
        heatmap=torch.cat(heatmap)##size N1xN0

        heatmap = F.upsample(heatmap.unsqueeze(0).unsqueeze(0), size=(I.shape[2]*N1,I.shape[3]*N0), mode='bilinear')[0, 0]
        print ("heatmap done",heatmap.shape)

        scipy.misc.imsave("%s/heatPred_%s_px%s_cr%s.png" % (savePath,name, opt.imageSize, opt.imageCrop),1- heatmap.cpu().numpy())
        smallI=torch.cat(smallI)
        vutils.save_image(smallI, "%s/heatPatch_%s_px%s_cr%s.png" % (savePath,name, opt.imageSize, opt.imageCrop), normalize=False,nrow=N0,padding=0)



    i = np.random.randint(len(tdataset.testP))
    name = tdataset.testP[i]
    save(name)

    i = np.random.randint(len(tdataset.Xneg))
    name = tdataset.Xneg[i]
    save(name)

# #sequential slide walk
# opt.WOS determined how many patches per slide to read
# #TODO use parallel workers -- too slow otherwise?
def validateSpatialGridSampling(dataset):
    posp = getSUP(dataset, count=opt.WOS,ntake=len(dataset.Xneg)*2)
    negp = getNUP(dataset, count=opt.WOS)  # #a big slicde count ensures that the slides can be decided easily
    print ("patch slices", len(posp), len(negp))

    pred_posp=[]
    pred_negp=[]
    ##now predict for all and keep probabilities
    for buf in posp:
        buf = torch.cat(buf).to(device)
        with torch.no_grad():
                pred = netD(buf)
        pred_posp.append(pred)
    for buf in negp:
        buf = torch.cat(buf).to(device)
        with torch.no_grad():
                pred = netD(buf)
        pred_negp.append(pred)

    posI = torch.cat(pred_posp,1).squeeze()#shape is gridpoints x count files
    negI = torch.cat(pred_negp,1).squeeze()
    I=torch.cat([posI,negI*0,negI],1)
    I=torch.sort(I,0,descending=True)[0]#so for each file, sorted by magnitude
    I=I[:300]##keep top patches, simpler

    ##TODO better save image directly with vutils
    plt.figure(figsize=(12, 12))
    plt.imshow(I.cpu().numpy(),interpolation='none')
    plt.ylabel('sorted patches, fixed count per slide')
    plt.colorbar()
    plt.savefig("%s/gridProbs_px%s_cr%s.png" % (savePath, opt.imageSize, opt.imageCrop),bbox_inches='tight')

    #now visualize these probabilities -- curve per plot
    if False:
        for pred,name in zip(pred_posp,dataset.testP):
            name=name[len(dataset.img_path):-4]#just human readable slide name
            plt.figure(figsize=(8,8))
            plt.plot(np.sort(pred[:, 0, 0, 0].cpu().numpy()), c='b', alpha=0.2)
            plt.savefig("%s/equalprobPos_%s_px%s_cr%s.png" % (savePath,name,opt.imageSize, opt.imageCrop))
            plt.close()

        for predp,name in zip(pred_negp,dataset.Xneg):
            name = name[len(dataset.img_path):-4]  # just human readable slide name
            plt.figure(figsize=(8,8))
            plt.plot(np.sort(pred[:, 0, 0, 0].cpu().numpy()), c='r', alpha=0.2)
            plt.savefig("%s/equalprobNeg_%s_px%s_cr%s.png" % (savePath,name,opt.imageSize, opt.imageCrop))
            plt.close()

bufP = []#for patch prob per SGD step#keep these buffers small, otherwise too large for whole training
bufN = []
buf=[]#for average minibatch error, train set
bufTest=[]#for average error, test set - -called fewer times

##usage python classify.py --imageSize=128 --imageCrop=1024
if __name__ == '__main__':
    if False:##warm start
        name = '%s/model_px%s_cr%s.dat' % (savePath, opt.imageSize, opt.imageCrop)
        print("loading name", name)
        netD.load_state_dict(torch.load(name))

    for it in range(1000):
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
                print ("step i",i)

            win = 50
            if i == 0 and len(buf) >3 and len(bufTest)>1:
                a = np.array(buf)
                abufN = np.array(bufN)
                abufP = np.array(bufP)
                print ("iter", it, i, "training loss", err.item(), "-100 -200 lag average", a[-100:].mean(), a[-200:].mean())
                plt.figure(figsize=(12, 12))
                plt.subplot(3, 1,1)
                plt.semilogy(buf,alpha=0.2)
                av=np.convolve(a, np.ones((win,))/win, mode='valid')
                plt.semilogy(av,color='r',lw=3,alpha=0.2)
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
                plt.semilogy(bufTest, lw=1, alpha=0.2)
                abufT = np.convolve(np.array(bufTest), np.ones((win,)) / win, mode='valid')
                plt.semilogy(abufT, lw=3, alpha=0.2)
                plt.xlabel("test step")
                plt.ylabel("cross entropy error")

                plt.legend()


                plt.savefig("%s/loss_px%s_cr%s_BN%s.png" % (savePath, opt.imageSize, opt.imageCrop,str(opt.BN)))
                plt.close()

            print ("SGD time", time.time() - t0)
            t0 = time.time()#to measure hust iter

        bufTest.append(valiScore())

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

