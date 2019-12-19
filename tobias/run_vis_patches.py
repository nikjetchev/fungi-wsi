import os
import numpy as np
import pandas as pd
import openslide
import imutils
import cv2
import imageio


# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------

dir_tifs = "/home/tobias/tmp/derma/wsi/"
dir_results = "/home/tobias/tmp/derma/results/2019-10-17_22-04-35/"
dir_out = "out/"

w = 256


# --------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------


cases = []

for i in range(100, 200):
    for t in ["N", "P"]:
        case = "{}{}".format(t, i)
        filename_probs = os.path.join(dir_results, "probPatch{}.tif.csv".format(case))
        if os.path.exists(filename_probs):
            cases.append(case)


print("#cases", len(cases))



for c in cases:
    print("=== {}".format(c))

    filename_probs = os.path.join(dir_results, "probPatch{}.tif.csv".format(c))
    filename_tif = os.path.join(dir_tifs, "{}.tif".format(c))

    df_probs = pd.read_csv(filename_probs, names=["x0", "y0", "p"])
    df_probs["x0"] = df_probs["x0"].astype(int)
    df_probs["y0"] = df_probs["y0"].astype(int)

    # --------------------------------------------------------------------------------------------------

    df_probs = df_probs.sort_values("p", ascending=False)

    slide = openslide.open_slide(filename_tif)

    top_imgs = []
    low_imgs = []

    probs = []
    n = 10
    for idx in df_probs.index[:n]:
        p = df_probs.loc[idx, "p"]
        probs.append(p)
        img = slide.read_region((df_probs.loc[idx, "x0"] - w//2, df_probs.loc[idx, "y0"] - w//2), 0, (w, w))
        arr = np.asarray(img.convert('RGB'))
        cv2.putText(arr, "{}%".format(int(100*p)), (70, 70), cv2.FONT_HERSHEY_PLAIN, 2.4, (0, 0, 0), 2)
        top_imgs.append(arr)

    for idx in df_probs.index[-n:]:
        p = df_probs.loc[idx, "p"]
        probs.append(p)
        img = slide.read_region((df_probs.loc[idx, "x0"] - w//2, df_probs.loc[idx, "y0"] - w//2), 0, (w, w))
        arr = np.asarray(img.convert('RGB'))
        cv2.putText(arr, "{}%".format(int(100*p)), (70, 70), cv2.FONT_HERSHEY_PLAIN, 2.4, (0, 0, 0), 2)
        low_imgs.append(arr)

    m = imutils.build_montages(top_imgs[:9] + low_imgs[-9:], (w, w), (3, 6))[0]
    imageio.imwrite("{}/patches_{}.png".format(dir_out, c), m)


