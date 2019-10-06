import openslide
# import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch.utils.data
import os
import numpy as np
from options import opt
import time as time
import random
import pickle

from PIL import ImageStat#for white filter

#array Nx2 of one patch
def getDots(filename_xml):
    tree = ET.parse(filename_xml)
    annos = []
    for c in tree.iter("Coordinate"):
        annos.append([int(float(c.get("X"))), int(float(c.get("Y")))])  # #int or float -- does slide tolerate float values?
    a = np.array(annos)
    return a

wh = opt.imageCrop  # #global constant for patch size#how much we read from open slide (level 0)#given by options
ClassRatio = 0.5  # #positive to negative examples

def sanity(name):
    s = openslide.open_slide(name)
    I = s.get_thumbnail((s.dimensions[0] // 100, s.dimensions[1] // 100))
    print("I", I.size)
    I.save("sanity%s.tiff"%(name.replace("/","_")))

##for vis. grid without holes
##buf has N0xN1 patches
def fullGrid(name,transform):
    s = openslide.open_slide(name)

    N0 = s.dimensions[0] //wh
    N1 = s.dimensions[1] //wh
    d0 = wh#s.dimensions[0] // N0  ##distance btw grid points
    d1 = wh#(s.dimensions[0] // N1

    print (s.dimensions,"full count patches",N0,N1,N0*N1)

    buf=[]
    for i in range(1, N1):##height
        if i%5==0:
            print ("fullGrid hmap",i,"/",N1)
        for j in range(1, N0):##width
            patch = centerPatch(s, j * d0, i * d1)
            buf.append(transform(patch)[:3].unsqueeze(0))
    return buf,N0-1,N1-1

# #TODO replace uniform random sampling with grids sampler.....
# # for each image in pos set, a set of randomly sampled patches
# @param ntake says how many slides to take, if -1 take all -- avoid the fact that more unlabeledPos than negative slides
# output is list of lists
# takes images from the unlabelled positive slides
def getSUP(dat, count=1,ntake=-1):
    if ntake <0:
        ntake=len(dat.testP)
    else:# random up to ntake
        random.shuffle(dat.testP)
    testP = dat.testP
    a = []  # list of lists

    ntake=min(len(testP),ntake)
    for i in range(ntake):
        s = openslide.open_slide(testP[i])
        a.append(getUp(s, dat, count,testP[i][dat.img_pathLen:]))
        s.close()
    return a

# # for each image in neg set, a set of randomly sampled patches
def getNUP(dat, count=1):
    testP = dat.Xneg
    a = []  # list of lists
    for i in range(len(testP)):
        s = openslide.open_slide(testP[i])
        a.append(getUp(s, dat, count,testP[i][dat.img_pathLen:]))
        s.close()
    return a

# given list of unlabelled P files, give random patch back -- single image
def getRandomUP(dat, count=1):
    testP = dat.testP
    i = random.randrange(len(testP))  # rand slide
    s = openslide.open_slide(testP[i])
    a= getUp(s, dat, count,testP[i][dat.img_pathLen:])
    s.close()
    return a
## get some patches given that slide s, # #slide already fixed!!
##if count>1 will use systemic spatial sampling, i.e. a grid instead of uniform prior
#@param name is partial name already
def getUp(s, dat, count=1,name=None):
    t0 = time.time()
    a = []
    if count > 1 and True:
        N0 = int(np.sqrt(count))
        N1 = N0##set counts
        d0= s.dimensions[0]//N0##distance btw grid points
        d1 = s.dimensions[1] // N1
        #if slie is not square d0!=d1, the wholes will be irregular -- does not matter, still an estimator

        for i in range(1, N0):
            for j in range(1, N1):
                ##strange, why strange looking predictions?
                if not checkCache(name,i * d0, j * d1):##so white:
                    a.append(torch.FloatTensor(torch.ones(1,3,opt.imageSize,opt.imageSize)))
                    #patch = dat.transform(centerPatch(s, i * d0, j * d1))[:3].unsqueeze(0)
                    #print ("sanity",a[-1].shape,patch.shape,patch.mean(3).mean(2))
                else:#if True:#so give proper info
                    patch=centerPatch(s,i * d0, j * d1)
                    a.append(dat.transform(patch)[:3].unsqueeze(0))
        print("fast data", len(a), "time", time.time() - t0)
        '''
        OBSOLETE -- fully random uniform sampling
        sn = SNailDataset(s, count, dat.transform)
        dataloader = torch.utils.data.DataLoader(sn, batch_size=count, shuffle=True, num_workers=opt.workers, drop_last=False)
        # #full batchsize, 1 step
        for i, data in enumerate(dataloader, 0):
            data, _ = data
            print ("fast data", data.shape, i, "time", time.time() - t0)
            # TODO debug how fast -- strange, fast has no benefit from workers, no gain?? 12.1s for 500 count worker3, worker1 11.9
            for b in range(data.shape[0]):
                a.append(data[b:b + 1, :3])
        '''

        return a

    #so count=1, getting random unlabelled pos patch
    for i in range(count):
        while True:
            xy = randDot(s.dimensions)  # random point there
            good = checkCache(name, xy[0], xy[1])
            if good or random.randrange(200) == 0:  # either nonwhite, or a random chance to give back anyway
                break
        img = centerPatch(s, xy[0], xy[1])
        if dat.transform is not None:
            img = dat.transform(img)
            a.append(img.unsqueeze(0)[:, :3])  # #alpha channels trimmed!

    if count > 1 :
        print ("slow data", time.time() - t0)
    return a

cacheRes=1024#cache was saved at this resolution
try:
    archive=pickle.load(open("cache%d.dat" % (cacheRes), 'rb'))
    print ("got archive",len(archive))
    print (archive.keys())
except Exception as e:
    print (e,"no archive")
    archive=None
def checkCache(name, x, y):
    nonwhite=archive[name]
    #print (nonwhite,"check",x-x%delta,y-y%delta)
    if (x-x%cacheRes,y-y%cacheRes) in nonwhite:
        return True
    return False

##TODO option to give batch quick whity in getUp, fullGrid--change centerpatch

# just random dot inside sldie, used to sample negative patches from  negative slides
def randDotNW(dimensions,name):
    #good = False
    while True:
        xy = [random.randrange(dimensions[0]), random.randrange(dimensions[1])]
        if checkCache(name, xy[0], xy[1]):
            print ("good")
            return np.array(xy)
        else:
            if random.rand() <0.005:#take a white as well
                print ("rand")
                return np.array(xy)

def randDot(dimensions):
    xy = [random.randrange(dimensions[0]), random.randrange(dimensions[1])]
    return np.array(xy)

##rejection sampling of points that are sick
##single points only -- TODO faster to make whole batch?
##uses Manhattan distance -- should I use Euclid?
def MCratio(coords, dimensions,N=1000000):
    clean=0
    for i in range(N):
        xy = [[random.randrange(dimensions[0]), random.randrange(dimensions[1])]]
        delta = coords - np.array(xy)  # #will broadcasting work in numpy? xy needs to be size (1,2)
        d = np.sqrt((delta**2).sum(1)).min()
        #d = np.abs(delta).sum(1).min()
        if d > wh:
            clean +=1

    pdirty= 1-clean/float(N)
    print ("statistics dirty",pdirty,"slide size",dimensions,"points_anno",coords.shape)
    return pdirty

##TODO couple N and d0....
##systematic spatial sampling
def Gridratio(coords, dimensions,N=500):
    clean=0
    #d0= dimensions[0]/float(N)#equal grid
    #d1 = dimensions[1] / float(N)  # equal grid
    d0=wh##distance btw grid points
    d1=wh
    N0=dimensions[0]//d0
    N1 = dimensions[1] // d1
    for i in range(1,N0):
        for j in range(1,N1):
            xy = [[i*d0, j*d1]]
            delta = coords - np.array(xy)  # #will broadcasting work in numpy? xy needs to be size (1,2)
            d = np.sqrt((delta**2).sum(1)).min()
            if d > wh:
                clean +=1

    pdirty= 1-clean/float(N0*N1)
    print ("statistics dirty",pdirty,"slide size",dimensions,"points_anno",coords.shape,"d",d0,d1,"N",N0,N1)
    return pdirty

##unused -- all points away from annotations
'''def randDotN(coords, dimensions):
    done = False
    while not done:
        xy = [[random.randrange(dimensions[0]), random.randrange(dimensions[1])]]
        delta = coords - np.array(xy)  # #will broadcasting work in numpy? xy needs to be size (1,2)
        if np.abs(delta).sum(1).min() > wh:  # #so away from all annotations
            return xy[0]  # a vector in R2
'''


# sample randomly any annotation points
# TODO consider equal density in image or labeled space?
def randDotC(coords,verbose=False):
    #random.randrange
    ri=random.randrange(coords.shape[0])
    xy = coords[ri]
    rxy = xy+np.int32((np.random.rand(2) - 0.5) * 2 / 3.0* wh )  # some small offset around center, +-0.33 wh
    # so we know center always on patch we crop with Centerpatch
    if verbose:
        print (ri,xy,rxy)
    return rxy

# #TODO sanity check when out of borders??
##x,y are center coordinates
##region is wh*wh , centered
def centerPatch(s, x, y, wh=wh, level=int(np.log2(opt.imageCrop/opt.imageSize))):
    if level >0:
        x=x-x%2**level
        y = y - y % 2 ** level
        #print ("center patch level",(wh//2**level, wh//2**level),level,x,y)
    if opt.resizeAugment:
        wh*=2#allow to keep center ok, the random translate is smaller anyway
    img = s.read_region((x - wh // 2, y - wh // 2), level, (wh//2**level, wh//2**level))
    return img  # #square image size s

# #for sequential sliding approach given slide s
class SingleNailDataset(Dataset):
    def __init__(self, s, count, transform=None):
        self.data = range(count)
        self.transform = transform
        self.slide = s

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
            # index = random.randrange(len(self.data))
            s = self.slide
            label = -1
            xy = randDot(s.dimensions)
            img = centerPatch(s, xy[0], xy[1])
            if self.transform is not None:
                # print (img.size, img.getbands()) 256, 256) ('R', 'G', 'B', 'A')
                img = self.transform(img)
            return img, label

class NailDataset(Dataset):
    """Dataset wrapping images from a random folder with medical images
        transform given outside from main torch script
        train: True is train set, False is test set
    """

    def __init__(self, img_path="/mnt/slowdata1/nagelpilz_p150/", transform=None, train=True):
        self.img_path = img_path + "wsi/"
        self.img_pathLen =len(self.img_path)
        anno_path = img_path + "anno_pre/"
        self.transform = transform

        totC = 0
        if True:  # multi worker compatible?
            names = os.listdir(self.img_path)
            print ("full image count",len(names))
            print (names)
            names.sort()
            self.Xpos = []  # names of positive images, restrain to 50 with annotations
            self.Xneg = []  # names of negative
            self.coords = []  # #coords for xpos

            self.testP = []  # names of unlabelled
            for n in names:
                if n[0] == "N":  # #negative examples
                    name = self.img_path + n
                    self.Xneg += [name]
                    continue

                name = self.img_path + n
                namea = anno_path + n[:-3] + "xml"
                try:
                    coords = getDots(namea)
                except Exception as e:
                    self.testP += [name]
                    #print (e)
                    continue  # #no coords labelling, use as positive validation set
                # s = openslide.open_slide(name)#do per patch sampled at train&inference time, very cheap
                self.Xpos += [name]
                self.coords += [coords]

        # #TODO add labels for some items, better split train and test
        # #coords is list of arrays -- stable with memory?

        #print (self.Xpos)
        #print (self.Xneg)
        #print (self.testP)

        print ("total CPU threads", torch.get_num_threads())

        Bp = len(self.Xpos) // 5 * 4  # boundary for pos
        Bn = len(self.Xneg) // 5 * 4

        ##for train and test split labeled posSlides and negSlides
        ##unlabeled posSlides always used fully for validation
        if train:
            self.Xpos = self.Xpos[:Bp]
            self.Xneg = self.Xneg[:Bn]
            self.coords = self.coords[:Bp]
        else:
            self.Xpos = self.Xpos[Bp:]
            self.Xneg = self.Xneg[Bn:]
            self.coords = self.coords[Bp:]

        self.train=train

        for c in self.coords:
            totC += c.shape[0]
        print ("total annotations", totC)
        print (n, "img added", coords.shape, totC, "total length", len(self.Xpos), len(self.Xneg),len(self.testP))

        # self.Xneg *= 10 ## no need - always random chance
        #self.Xpos *= 1  # #so more items, simplify data iterations in external programs
        #self.coords *= 1

    #what percentage of spatial area is infected?
    def printStatisticsSpatialPos(self):
        pvals= []
        for name,coords in zip(self.Xpos,self.coords):
            pvals += [Gridratio(coords,openslide.open_slide(name).dimensions)]

        r=np.array(pvals).mean()
        print ("overall percentage",r,1/r)

    #@param dummy is index - we do not use it, always sample random element, so we tune better epoch length
    def __getitem__(self, dummy):
        if np.random.rand() < ClassRatio:  # pos class
            index = random.randrange(len(self.Xpos))#random slide
            s = openslide.open_slide(self.Xpos[index])  # slide opening 1ms --fast enough
            coords = self.coords[index]
            label = 1
            xy = randDotC(coords)#,verbose=not self.train
            img = centerPatch(s, xy[0], xy[1])
        else:  # neg class
            index =random.randrange(len(self.Xneg))
            partialN=self.Xneg[index][self.img_pathLen:]
            s = openslide.open_slide(self.Xneg[index])
            label = 0
            while True:
                xy = randDot(s.dimensions)
                if partialN in archive:#faster way to get this info
                    good = checkCache(partialN, xy[0], xy[1])
                    if good or random.randrange(5000)==0:#either nonwhite, or a random chance to give back anyway
                        img = centerPatch(s, xy[0], xy[1])
                        break
                else:#long way, manual check
                    raise Exception('slow')
                    img = centerPatch(s, xy[0], xy[1])
                    ps = ImageStat.Stat(img)
                    m = np.array(ps.mean)
                    good = m.mean() < 250
                    if good or random.randrange(200)==0:#either nonwhite, or a random chance to give back anyway
                        break
        s.close()

        if self.transform is not None:
            # print (img.size, img.getbands()) 256, 256) ('R', 'G', 'B', 'A')
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.Xpos)*50  # #pos is limited by true annotations -- Xneg is unlimited patches#multiply to have longer iterations

# #main routine to check if iterator works

def memoryAll(dataset):
    L = 3
    S = 2 ** L

    buf=[]
    for name in dataset.Xpos:
        s = openslide.open_slide(name)
        print (name)
        I = s.get_thumbnail((s.dimensions[0] // S, s.dimensions[1] // S))
        buf.append(I)
    print ("full buf",len(buf))
    for I in buf:
        print (I.size)
        #I.save("imt/I_%s.tiff"%(name))

def checkLevels(dataset):
    pix=0
    s = openslide.open_slide(dataset.Xpos[pix])
    print (dataset.Xpos[pix])
    print (s.level_dimensions)
    print (s.level_downsamples)
    # print (s.properties)
    #print (s.associated_images)

    '''
    ((79992, 182160), (39996, 91080), (19998, 45540), (9999, 22770), (4999, 11385), (2499, 5692), (1249, 2846), (624, 1423))
(1.0, 2.0, 4.0, 8.0, 16.000800160032007, 32.00620740214568, 64.02522889710222, 128.1017757716633)'''
    import PIL
    PIL.Image.MAX_IMAGE_PIXELS = 1e9*8

    coords = dataset.coords[0]
    c=coords[20]##one point
    print ("selected coordinate",c,c.dtype)
    L = int(np.log2(opt.imageCrop/opt.imageSize))
    S = 2 ** L

    print ("S L",S,L)

    #c*=0
    #c[0]=int(c[0])
    #c[1]=opt.Woff#int(c[1])

    from time import time
    t0=time()
    for z in range(10):
        s = openslide.open_slide(dataset.Xpos[0])
        p0=centerPatch(s,c[0],c[1],level=0)
        p0=p0.resize((wh//S,wh//S))
    print ("fullres",time()-t0)
    ##why different image, different alpha?
    ##fullres 0.17575931549072266
    #partial 0.051888227462768555

    #with sldie becomes fullres 1.7532885074615479
    #partial 1.0513935089111328
    t0 = time()
    for z in range(10):
        s = openslide.open_slide(dataset.Xpos[0])
        p1 = centerPatch(s, c[0], c[1], level=L,wh=wh)
    print("partial", time() - t0)
    print (p0.size)
    print (p1.size)
    print ("p0 p1",np.array(p0).mean(),np.array(p1).mean())

    p0.save("p0.tiff")
    p1.save("p1.tiff")
    raise Exception()
    '''
    import scipy.misc
    a1=np.array(p1)[:,:,:3]
    print ("a1",a1.shape,a1.mean(0).mean(0),a1.max(),a1.min())
    scipy.misc.imsave("a1.png",a1)
    scipy.misc.imsave("a11.png", a1[:,:,::-1])
    #p1.save("p1.png")

    print (p0.mode)
    print (p1.mode)
    '''
    I = s.get_thumbnail((s.dimensions[0] // S, s.dimensions[1] // S))

    print("I", I.size)
    I.save("I.tiff")

    #whs=wh//S*0.5##half side
    ##box = (c[0] // S-whs, c[1] // S-whs, c[0] // S + whs, c[1] // S + whs)
    #box = (0, 0, 2000, 2000)
    #print(box)

    t0 = time()
    for z in range(100):
        #from PIL import Image
        #I=Image.open("I.tiff")#Icrop time 42.348323583602905 with this
        ##0.02 without the open
        IS = openslide.open_slide("I.tiff")##also this slow -- create new slide?
        p2=centerPatch(IS, c[0]//S, c[1]//S,wh=wh//S)
        #p2 = I.crop(box)  # .thumbnail((wh//S,wh//S))
    print("crop", p2.size)
    print("Icrop time", time() - t0)

    from PIL import ImageStat
    ps = ImageStat.Stat(p2)
    print("mean2", ps.mean)
    ps = ImageStat.Stat(p0)
    print("mean0", ps.mean)
    ps = ImageStat.Stat(p1)
    print("mean1", ps.mean)

    p2.save("p2.tiff")

    '''
    I = s.get_thumbnail((s.dimensions[0]//S,s.dimensions[1]//S))
    print("I",I.size)
    I.save("I.tiff")
    #from PIL import Image
    #I=Image.open("I.tiff")
    #file looks goodI.save("I.tiff")
    box=(c[0]//S-50,c[1]//S-50,c[0]//S+100,c[1]//S+100)
    box = (0,0,2000,2000)
    print (box)
    p2=I.crop(box)#.thumbnail((wh//S,wh//S))
    print("crop",p2.size)

    from PIL import ImageStat
    ps=ImageStat.Stat(p2)
    print ("mean",ps.mean)
    ps = ImageStat.Stat(I)
    print("Full mean",ps.mean)

    p2.save("p2.tiff")

    #p2 = I.crop(box)  # .thumbnail((wh//S,wh//S))
    #print("crop", p2.size)
    #p2.save("p2.tiff")'''

if __name__ == "__main__":
    import torchvision.transforms as transforms
    import torch
    import torch.utils.data

    tbuf = [transforms.RandomCrop(wh), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))]
    transform = transforms.Compose(tbuf)
    dataset = NailDataset(transform=transform)

    #memoryAll(dataset)
    checkLevels(dataset)
    raise Exception

    dataset.printStatisticsSpatialPos()

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1, drop_last=True)

    for i, data in enumerate(dataloader, 0):
        im, lab = data
        print (i, im.shape, lab.shape)
        print (lab)

    if False:
        path = "/mnt/slowdata1/2019_03_29__p150__nagelpilz_positive/"
        path = ""
        filename_wsi = path + "wsi/P001.tif"
        filename_xml = path + "anno_pre/P001.xml"
        import numpy as np
        a = getDots(filename_xml)
        import matplotlib.pyplot as plt
        plt.scatter(a[:, 0], a[:, 1])
        plt.show()

