import os
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics


# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

dir_results = "/home/tobias/tmp/derma/results/2019-10-22_22-57-56/"

dir_results = "/home/nikolay/medtorch/resultsVali/2019-10-22_22-57-56/"

dir_results = "/home/nikolay/medtorch/resultsVali/2019-11-12_08-59-09/"
prefix = "probPatch_256px_"

#dir_results = "/home/nikolay/medtorch/resultsVali/2019-11-10_23-34-19/"

#dir_results = "/home/nikolay/medtorch/resultsVali/2019-11-12_09-03-06/"
#prefix="probPatch_256px_doubleGridRes_"
#prefix="probPatch_smallerPatch128px_"


dir_out = "out2/"

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------

cases = []

for i in range(100, 200):
    for t in "N", "P":
        case = "{}{}".format(t, i)
        #filename_probs = os.path.join(dir_results, "probPatch{}.tif.csv".format(case))
        filename_probs = os.path.join(dir_results, "{}{}.tif.csv".format(prefix,case))
        if os.path.exists(filename_probs):
            cases.append(case)
print("#cases", len(cases))


# ------------------------------------------------------------------------------------------------------------

df_cases = pd.DataFrame(index=sorted(cases), columns=["true", "pred"])

for c in cases:
    #print("=== {}".format(c))

    #filename_probs = os.path.join(dir_results, "probPatch{}.tif.csv".format(c))
    filename_probs = os.path.join(dir_results, "{}{}.tif.csv".format(prefix, c))

    df_patches = pd.read_csv(filename_probs, names=["x0", "y0", "p","label"])
    df_patches["x0"] = df_patches["x0"].astype(int)
    df_patches["y0"] = df_patches["y0"].astype(int)
    df_patches = df_patches.sort_values("p", ascending=False)

    df_cases.loc[c, "true"] = 1 if c.startswith("P") else 0
    df_cases.loc[c, "pred"] = df_patches["p"].max()


df_cases["true"] = df_cases["true"].astype(int)

plt.figure()
df_cases["pred"].plot()
# plt.show()
plt.savefig(os.path.join(dir_out, "probsline.png"))


auc = sklearn.metrics.roc_auc_score(df_cases["true"], (df_cases["pred"]))
df_cases["pred_binary"] = df_cases["pred"] > 0.5
acc = sklearn.metrics.accuracy_score(df_cases["true"], df_cases["pred_binary"])
tn, fp, fn, tp = sklearn.metrics.confusion_matrix(df_cases["true"], df_cases["pred_binary"]).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)

print("auc         = {}".format(auc))
print("acc         = {}".format(acc))
print("sensitivity = {}".format(sensitivity))
print("specificity = {}".format(specificity))
print("precision   = {}".format(precision))

