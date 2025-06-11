import pandas as pd
import numpy as np
import torchvision
import torch
import subprocess
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from sklearn.manifold import TSNE
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from io import StringIO
from .Models_20 import *
from tqdm import tqdm
import progressbar
import os
from importlib.resources import files

def ELIGOS_feature_index(feature_path, index_path):
    filesize = os.path.getsize(feature_path)
    bar = progressbar.ProgressBar(maxval=filesize)
    bar.start()
    with open (feature_path, 'r') as f, open(index_path,'w') as fa:
        Ina = 0
        Hikari = 1
        old_txid = ''
        head = len(f.readline())
        fa.write('id,start,end\n')
        for line in f:
            Ina += 1
            if Ina % 10000 == 0:
                Hikari += 1
            line1 = line.rstrip().split(",")
            inters = len(line)
            txid = f'{line1[0]}_{Hikari}'
            if txid == old_txid:
                pos_end += inters
            else:
                if old_txid == '':
                    pos_start = head
                    pos_end = pos_start + inters
                    old_txid = txid
                else:
                    fa.write(f"{old_txid},{pos_start},{pos_end}\n")
                    pos_start = pos_end
                    pos_end += inters
                    old_txid = txid
            bar.update(pos_end)
        fa.write(f"{txid},{pos_start},{pos_end}\n")


def feature_index(feature_path, index_path):
    filesize = os.path.getsize(feature_path)
    bar = progressbar.ProgressBar(maxval=filesize)
    bar.start()
    with open (feature_path, 'r') as f, open(index_path,'w') as fa:
        old_txid = ''
        head = len(f.readline())
        fa.write('id,start,end\n')
        for line in f:
            line1 = line.rstrip().split(",")
            inters = len(line)
            
            txid = line1[0]
            if txid == old_txid:
                pos_end += inters
            else:
                if old_txid == '':
                    pos_start = head
                    pos_end = pos_start + inters
                    old_txid = txid
                else:
                    fa.write(f"{old_txid},{pos_start},{pos_end}\n")
                    pos_start = pos_end
                    pos_end += inters
                    old_txid = txid
            bar.update(pos_end)
        fa.write(f"{txid},{pos_start},{pos_end}\n")
        
def model_load(fe_path, cc_path):
    feature_extractor = Extractor()
    class_classifier = Class_classifier()
    feature_extractor.cuda()
    class_classifier.cuda()
    bfw = torch.load(fe_path)
    bcw = torch.load(cc_path)
    feature_extractor.load_state_dict(bfw)
    class_classifier.load_state_dict(bcw)
    
    return feature_extractor, class_classifier

def dataframe_split(input_df: pd.DataFrame, n_processes: int):
    rows = len(input_df)
    
    n = min(n_processes, rows)
    
    df_splits = np.array_split(input_df, n)
    
    return df_splits


def feature_read(index_row, feature_path, New_Features):

    chunk_start_1 = index_row['start']
    chunk_end_1 = index_row['end']

    # chunk_start_2 = index_csv.iloc[-1]['start']
    # chunk_end_2 = index_csv.iloc[-1]['end']

    feature_file = open(feature_path)
    headers = feature_file.readline().rstrip().split(",")
    feature_file.seek(chunk_start_1,0)
    feature_str = feature_file.read(chunk_end_1 - chunk_start_1)

    # feature_file.flush()
    # feature_file.seek(chunk_start_2,0)
    # else_str = feature_file.read(chunk_end_2 - chunk_start_2)

    feature_file.close()
    
    feature_csv = pd.read_csv(StringIO(feature_str),sep=",",names=headers, dtype={'position':int})
    # else_csv = pd.read_csv(StringIO(else_str),sep=",",names=headers, dtype={'position':int})
    # feature_csv = pd.concat([feature_csv, else_csv], axis=0)

    feature_csv = feature_csv.set_index(['id','position'])

    kmers = list(feature_csv['kmer'])
    depths = list(feature_csv['depth'])
    
    if 'mod_rate' in feature_csv.columns:
    
        mod_rate = list(feature_csv['mod_rate'])
        feature_csv = feature_csv.drop(['kmer', 'depth', 'mod_rate'], axis=1)
    else:
        mod_rate = [0 for i in range(0, len(feature_csv))]
        feature_csv = feature_csv.drop(['kmer', 'depth'], axis=1)
        
    feature_values = feature_csv.values.reshape(len(feature_csv),5,56)
    feature_tensor = torch.from_numpy(feature_values).float().cuda()
    
    txids = list(feature_csv.index)
    
    return txids, feature_tensor, kmers, depths, mod_rate

def prediction(feature_tensor, extractor, classifier, output_path,txids, kmers, depths, label_rate):
    feature_feature = extractor(feature_tensor)
    class_preds, rate_preds = classifier(feature_feature)
    label_pred_np = np.exp(class_preds.data.cpu().numpy()[:,1])
    rate_pred_np = rate_preds.data.cpu().numpy()
    
    with open(output_path,'a') as f:
        for txid, kmer, depth, pval, pr, mr in zip(txids, kmers, depths, label_pred_np, rate_pred_np, label_rate):
            idx, position = txid
            f.write(f"{idx},{position},{kmer},{depth},{pval},{pr}\n")

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)
