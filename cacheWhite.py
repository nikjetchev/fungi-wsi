from patchSampler import centerPatch,wh
import os
import openslide
import numpy as np
from PIL import ImageStat
import pickle

img_path="/mnt/slowdata1/nagelpilz_p150/"
img_path+= "wsi/"

def Gridratio_(s):
    nonwhite=set()
    d0=wh##distance btw grid points
    d1=wh
    N0=s.dimensions[0]//d0
    N1 = s.dimensions[1] // d1

    print ("slide size",s.dimensions,"d",d0,d1,"N",N0,N1)

    ##to call for each min max pair
    #if boring pink return true
    def pinkCriteria(ext):
        if ext[0]>240 and ext[1]-ext[0]<7:#very bright, small bandwidth
            return True
        return False

    for i in range(1,N0):
        if i%20==0:
            print ("row",i)
        for j in range(1,N1):
            patch = centerPatch(s,i*d0,j*d1)##level will be auto found by imageCrop, imageSize
            ext=patch.getextrema()

            filt=False
            if ext[0][0]>254 and ext[1][0]>254 and ext[2][0]>254:#so all RGB pixels are white, min is 255
                filt=True
            elif pinkCriteria(ext[0]) and pinkCriteria(ext[1]) and pinkCriteria(ext[2]): ##check if constant values, also boring case
                filt=True
            else:               #so potentially good, non-boring by other 2 conditions
                #more expensive filter
                stat = ImageStat.Stat(patch)
                r, g, b,alpha = stat.mean
                if r > 235 and g > 235 and b > 235:
                    filt=True # possible case: few tissue pixels, no tissue overall
            if not filt:#so some pixels with non-white value
                nonwhite.add((i*d0,j*d1))
                if len(nonwhite)%5==1:
                    print(ext, " i ",i,"found",len(nonwhite))

    pdirty= len(nonwhite)/float(N0*N1)
    print('')
    print ("statistics tissue",pdirty)
    print ('')
    return nonwhite

def Gridratio(name):
    print (name)
    s=openslide.open_slide(img_path+name)
    res=Gridratio_(s)
    return res

##example usage python cacheWhite.py --imageCrop=1024 --imageSize=64
if __name__ == "__main__":
    names = os.listdir(img_path)
    d={}
    for n in names:
        res=Gridratio(n)
        d[n] = res
        pickle.dump(d,open("cache%d.dat"%(wh),'wb'))
        print ("done cache",len(d))