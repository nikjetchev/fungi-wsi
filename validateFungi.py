'''
Created on Jul 14, 2019

@author: nikolay
'''

from patchSampler import NailDataset, getRandomUP, getSUP, getNUP,sanity,getUp

import openslide
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
import matplotlib.pyplot as plt

import time
import random

from patchSampler import fullGrid,fullGrid_tissue
import scipy.misc
import torch.nn.functional as F
import sklearn.metrics as metrics
import pickle

from networks import _netD, weights_init
import numpy as  np
import torch.optim as optim
from options import opt


import datetime
stamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
savePath = "resultsVali/"+ stamp + "/"
import os
try:
    os.makedirs(savePath)
except OSError:
    pass

wh = opt.imageSize  # #resizes and eventually downsamples the cropped image slice, which should be larger usually
tbuf = [transforms.Resize(size=wh, interpolation=2),transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]
transform = transforms.Compose(tbuf)
tdataset = NailDataset(transform=transform, train=opt.VTrain)#for test no random augmenting
tdataloader = torch.utils.data.DataLoader(tdataset, batch_size=opt.batchSize, shuffle=True, num_workers=opt.workers)  # o drop here -- but random patch samplign anyway

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print ("device", device)
ndf = opt.ndf
netD = _netD(ndf, int(np.log2(wh)), nc=4 - 1)
print(netD)
netD=netD.eval()
netD = netD.to(device)

criterion = nn.BCELoss()

#statistics on test set, using same distributions as on training set
#also saves random patches from each class
def validate():
    err = []
    pl = []  # predict and label

    visImg = [[], [], []]
    nVI = 15*15  # count of patches for auxilliary visualization routine

    nTEV = 5  #how many dataset iterations for total error validation, more totally random patches, both slide and label
    if opt.VTrain:
        nTEV=2
    with torch.no_grad():
        for it in range(nTEV):
            print ("vali",it,"/",nTEV)
            np.random.seed(it)
            random.seed(it)
            for i, data in enumerate(tdataloader, 0):
                im, lab = data
                im = im[:, :3].to(device)  # trim alpha channel
                lab = lab.to(device).float()
                with torch.no_grad():
                    pred = netD(im)

                for b in range(pred.shape[0]):
                    pl += [[pred[b,0,0,0].item(), lab[b].item()]]
                    err += [criterion(pred[b:b + 1, 0, 0, 0], lab[b:b + 1]).item()]

                    l = lab[b].item()#save image patch example in list for type: pos or neg
                    #if len(visImg[int(l)]) < nVI:

                    v=pred[b,0,0,0].item()
                    visImg[int(l)].append((v,im[b:b + 1].cpu()))

    pl = np.array(pl)
    print("total err validation", np.array(err).mean(), pl.shape)

    if not opt.VTrain:
        unlabeledP = sampleUnlabeledPos(tdataset, visImg, nVI, pl)##unlabelled postiive, also fills inplace list visIMG -- index2
        names = ["negative", "positiveLabelled", "positiveUnlabelled"]
    else:
        names = ["negative", "positiveLabelled"]##if VTrain only overfit statistics
        unlabeledP =pl[:10]

    def savePList(visL,l):
        top=[]#keep highest scored patches
        for prob,im in visL[:nVI]:
            top.append(im)

        buf = torch.cat(top) * 0.5 + 0.5
        print("class examples", buf.shape, buf.max(), buf.min())
        vutils.save_image(buf, '%s/class%s_%spx_cr%s_train%s.jpg' % (savePath, l, opt.imageSize, opt.imageCrop,str(opt.VTrain)),
                          normalize=False)

    # #now grids with img, buffer filled for vis
    for l in range(len(names)):
        visL = visImg[l]#all images for one category, a list, also prediction probabilitie<
        visL.sort(key=lambda x: x[0],reverse = True)
        savePList(visL,names[l])

    whole=visImg[0]+visImg[1]+visImg[2]
    whole.sort(key=lambda x: x[0],reverse = True)
    print("M", whole[0][0], whole[200][0])
    savePList(whole,"mostlikely")

    #TODO set all totally white to high value, see low prob tissue samples
    def countBlack(img):
        #img=img.view(img.shape[2],img.shape[3])
        #c=torch.FloatTensor(img<1)
        c=(img<0.8).double()
        m= c.mean()
        return m
        #print (m)

    br=0
    for i in range(len(whole)):
        if countBlack(whole[i][1])<0.5:#so image less than half tissue
            whole[i]=(10000,whole[i][1])#so will not be taken for display of least prob
            br+=1
        if i%200==0:
            print ("len white filtering",i,br)

    print ("erased whites",br)
    whole.sort(key=lambda x: x[0],reverse = False)
    print ("L", whole[0][0], whole[200][0])
    savePList(whole,"leastlikely")

    #uses the  training distribution, equal prob. for true and false class
    visualizeEqualProb(unlabeledP, pl)
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

##very slow -- makes full patch buffer, eats RAM TODO make run online...
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
        name = name[len(tdataset.img_path):-4]##for save last part just
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

def slideClassify(posI,negI,N=200,type="MEAN"):
    #N=200##how many to keep

    posI=torch.sort(posI,0,descending=True)[0]#so for each file, sorted by magnitude
    posI=posI[:N]##keep top patches, simpler

    negI=torch.sort(negI,0,descending=True)[0]#so for each file, sorted by magnitude
    negI=negI[:N]##keep top patches, simpler

    for i in range(posI.shape[1]):
        v=posI[:,i].mean()
        if v < 0.01:
            print (i,tdataset.testP[i],v)##sanity -- why some testP slides have weak probability...

    print ("arrays ready",negI.shape,posI.shape)
    if type=="MEAN":
        negI=negI.mean(0)
        posI = posI.mean(0)
        type +=str(N)
    else:
        negI=negI.max(0)[0]
        posI=posI.max(0)[0]##

    ##scatter plot with respective class and probability
    plt.figure(figsize=(10, 10))
    plt.scatter(posI,(posI*0).uniform_(-0.1,0.1)+1, alpha=0.1, s=50,c='r',label='pos')
    plt.scatter(negI,(negI*0).uniform_(-0.1,0.1), alpha=0.1, s=50,c='b',label='neg')
    plt.ylabel('class')
    #plt.hist(posI, bins=50, alpha=0.4, label='pos',density=True)
    #plt.hist(negI, bins=50, alpha=0.4, label='neg',density=True)
    plt.xlabel('probability predicted')
    #plt.ylabel('count')
    plt.legend()
    plt.savefig("%s/slideProb_%s_%spx_cr%s.png" % (savePath,type,opt.imageSize, opt.imageCrop))
    plt.close()

    plt.figure(figsize=(10, 10))
    x= torch.cat([posI,negI])
    y= torch.cat([posI*0+1,negI*0])

    fpr, tpr, threshold = metrics.roc_curve(y,x)  # #labels are first argument
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("%s/sliceAUC_%s_%spx_cr%s.png" % (savePath,type,opt.imageSize, opt.imageCrop))
    print ("AUC", roc_auc)
    plt.close()

    #https://en.wikipedia.org/wiki/F1_score
    from sklearn.metrics import precision_recall_curve
    precision, recall, T = precision_recall_curve(y, x)
    f1= 2 * (precision * recall) / (precision + recall)
    plt.plot(f1)
    plt.plot(T)
    plt.ylabel('F1 score')
    #plt.xlabel('thresg')
    plt.savefig("%s/sliceF1_%s_%spx_cr%s.png" % (savePath, type, opt.imageSize, opt.imageCrop))
    plt.close()

# #sequential slide walk
# opt.WOS determined how many patches per slide to read
# #TODO use parallel workers -- too slow otherwise?
def validateSpatialGridSampling(dataset):
    ##no need to save lists -- directly work on each slide, save predictions
    def perfP(fname=None):
        fname2= fname[dataset.img_pathLen:]
        #l=getUp(openslide.open_slide(fname), dataset, opt.WOS, fname2)
        l=fullGrid_tissue(fname, dataset.transform)
        ans=[]
        a=torch.cat(l)##very large list, on cpu
        chunks=a.chunk(max(1,a.shape[0]//10))
        print ("chunks",len(chunks))
        print (chunks[0].shape)
        with torch.no_grad():
            for c in chunks:
                c=c.to(device)
                pred = netD(c)
                ans.append(pred)

        out= torch.cat(ans).cpu()
        ix = np.argsort(out.squeeze())##save patches -10: from this list -- highest ranked
        print (ix.shape,out.shape,out.squeeze()[ix[-10:]].shape)
        print ("top patches",fname,out.squeeze()[ix[-10:]])
        maxRanked=a[ix[-10:]]*0.5+0.5
        vutils.save_image(maxRanked, '%s/maxProbPatch%s.jpg' % (savePath,fname2),normalize=False)
        return out
    pred_posp=[]
    pred_negp=[]
    ##now predict for all and keep probabilities
    #for buf in posp:
    for name in dataset.testP[:5]+dataset.Xpos[:5]:
        pred=perfP(name)
        pred_posp.append(pred)
        print ("probs",len(pred_posp),pred.shape)
    for name in dataset.Xneg[:5]:
        pred=perfP(name)
        pred_negp.append(pred)
        print("probs", len(pred_negp), pred.shape)

    posI = torch.cat(pred_posp,1).squeeze()#shape is gridpoints x count files
    negI = torch.cat(pred_negp,1).squeeze()

    slideClassify(posI,negI,100,"MEAN")
    slideClassify(posI,negI,10,"MEAN")
    slideClassify(posI, negI, 5, "MEAN")
    slideClassify(posI, negI,type= "MAX")

    I=torch.cat([posI,negI*0,negI],1)
    I=torch.sort(I,0,descending=True)[0]#so for each file, sorted by magnitude
    I=I[:300]##keep top patches, simpler

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

bufP = []##keep these buffers small, otherwise too large for whole training
bufN = []
buf=[]

##usage python validateFungi.py --imageSize=128 --imageCrop=1024 --WOS=1200
if __name__ == '__main__':
    # --imageSize=256 --imageCrop=256 --ndf=64 --batchSize=40 --BN=True --loadPath='results/2019-10-08_19-41-03model_px256_cr256_BNTrue.dat'
    name=opt.loadPath
    print ("name",name)
    netD.load_state_dict(torch.load(name))
    try:
        pass
        #showRandomFullSlide_Heatmap()
    except Exception as e:
         print ("heatmap",e)
    try:
        # uses proper spatial frequency, so many more blank patches than sick ones
        # obsolete - now we call full tissue grid
        #if opt.WOS>200:
        validateSpatialGridSampling(tdataset)
        validate()
    except Exception as e:
        print ("validate error",e)


