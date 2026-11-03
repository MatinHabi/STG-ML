import model3 as m
from pathlib import Path
import torch

EPOCH = 1500
EVAL_EVERY = 10 
MATCH_K = 354 #number of markers picked by VBS on teh 2014 data

def fmt(tag,d):
    return f"{tag}: r={d['r']:.3f} RE={d['re']:.2f}% MSE={d['mse']:.3f} gates={d['gates']}"


