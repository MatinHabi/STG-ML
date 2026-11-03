#THIS SCRIPT ANSWERS THE QUESTION OF "whose selected markers are best VBS or STG"
#VBS and STG give their markeres - ridge uses their selected markers to make predictions the one with predictions closest to the true value wins
#this is done by comparing real and predicted yield via pearson's correlation (r) & MSE
#this is important as it shows which models MARKERS (not pipeline and architecture) is actually more effective
import sys
import numpy as np
import torch
from sklearn.linear_model import RidgeCV #Ridge regression: a linear model with a penulty alpha that keeps the weights small so it doesnt overfit with many markers
                                         #ridge regression exists as a baseline comparison for both VBS & STG

d = torch.load(sys.argv[1])
mu = torch.load(sys.argv[2]) if len(sys.argv > 2) else None
vbs = np.loadtxt("VBS-ML-Genomic-Prediction/2014_Data/354_idx.txt", dtype = int) #loading VBS 354 selected markers
K = len(vbs)

Xtr, ytr = d["X_train"].numpy(), d["Y_train"].numpy()
if not d.get("no_valid", False): #if there isn't a validation set (same as how VBS was trained & tested)
    Xtr = np.vstack([Xtr, d["X_valid"].numpy()]) #putting validation plants rows under training plants (because validation doesnt exist)
    ytr = np.concatenate([ytr, d["Y_valid"].numpy()]) #vstack to add rows and concatonate to add the markers to the end of the 1D array

Xte, yte = d["X_test"].numpy(), d["Y_test"].numpy()

def score(cols=slice(None)): #slice(None) is the same as : so Xtr[:,slice(None)] = Xtr[:,:] which means use all 17,114 markers
    baseline = RidgeCV(alphas=np.logspace(-1,6,15)).fit(Xtr[:,cols],ytr) #np.logspace(-1,6,15) gives 15 nmbers spread evenly on a log scale from 10^-1 to 10^6
                                                                    #these 15 nums are the the alpha values to try because it starts really small and gets huge (0.1, 0.32, 1, 3.2, 10, 32, 100, ..., 316228, 1000000)
                                                                    # .fit(Xtr[:,cols],ytr) trains the model -> learns a single weight per marker in cols. finds and keeps the best alpha value
                                                                    # baseline is now the baseline model to use to comapre VBS ans STG markers
    
    pred = baseline.predict(Xtr[:,cols])

    return f"r = {np.corrcoef(pred,yte)[0,1]:.3f} || MSE = {np.mean((pred - yte) ** 2):.3f}"

print(f"ridge with all {Xtr.shape[1]} markers: {score()}  (paper GBLUP/LMM r: 0.52 0.49 0.50 0.47)") #sanity check to see if the implimenantion of Ridge is correct - should get very similar numbers 
print(f"ridge using VBS's {K} markers: {score(vbs)}")
print(f"ridge with {K} random markers: {score(np.random.default_rng(0).choice(Xtr.shape[1], K, replace=False))}")

if mu is not None:
    stg = torch.argsort(mu, descending=True)[:K].numpy() #getting the column for markers ordered from highest gate value to lowest and only keeps the top K 
    print(f"ridge using STG's top {K} markers: {score(stg)} || Overlap with VBS: {len(np.intersect1d(stg, vbs))}/{K}") #printing rigde's score using K markers and how many VBS of the same markers VBS also picked