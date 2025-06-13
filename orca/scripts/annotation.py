import pandas as pd
import numpy as np
import torch
from itertools import product
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
tqdm.pandas()
import time
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import datetime
import os
import pysam

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIGNAL_COLS = [f'{x}_{y}' for x in ['-2', '-1', '0', '1', '2'] for y in ['Mean_average', 'Stdv_average', 'Mean_svar', 'Stdv_svar', 'Cov']]
# SEQUEN_COLS = [f'pos_{x}_{y}' for x in range(3, 8) for y in ['A', 'C', 'G', 'T']]
SEQUEN_COLS = [f'New_Seq_{x}' for x in range(0, 256)]
CHANGE_COLS = [f'{x}_{y}' for x in ['-2', '-1', '0', '+1', '+2'] for y in ['Mismatch_Ratio', 'Insertion_Ratio', 'Deletion_Ratio', 'Qual_Mean', 'Qual_Median', 'Qual_Stdv']]

class RNADataset(Dataset):
    def __init__(self, features, labels1, labels2):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels1 = torch.tensor(labels1, dtype=torch.long)
        self.labels2 = torch.tensor(labels2, dtype=torch.long)
    def __len__(self):
        return len(self.features)
    def __getitem__(self, idx):
        return self.features[idx], (self.labels1[idx], self.labels2[idx])

class DualOutputAutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes, dropout_rate=0.4):
        super(DualOutputAutoEncoder, self).__init__()
        # Encoder part
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(128, latent_dim)
        
        # Decoder part
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, input_dim)
        )
        
        # Classification head for modification type prediction
        self.task1_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        # Phase classification head
        self.task2_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 5)
        )
        
    def forward(self, x):
        h = self.encoder(x)
        z = self.fc_mu(h)  # latent vector
        x_recon = self.decoder(z)
        logits1 = self.task1_head(z)  # modification type classification
        logits2 = self.task2_head(z)  # phase classification
        return x_recon, logits1, logits2, z


def pretrain_AE(model, full_data, num_epochs=50, batch_size=1024, beta=1.0, best_model_path='best_pretrain.pth'):
    """
    Pretrain the AE part on all data.
    """
    class FullDataset(torch.utils.data.Dataset):
        def __init__(self, data):
            self.data = torch.tensor(data, dtype=torch.float32)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = FullDataset(full_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.002)
    model.train()
    best_loss = float('inf')
    epochs_no_improve = 0
    patience = 10
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        total_samples = 0
        
        for inputs in loader:
            inputs = inputs.to(DEVICE)
            optimizer.zero_grad()
            x_recon, _, _, _ = model(inputs)
            loss_recon = F.mse_loss(x_recon, inputs)
            loss = beta * loss_recon
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
        
        avg_loss = total_loss / total_samples
        if epoch % 20 == 0:
            print(f"Pretrain Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping pretraining at epoch {epoch}")
            break
    
    print("Loading best pretrained model...")
    model.load_state_dict(torch.load(best_model_path))
    return model


def fine_tune_AE_with_adversarial(model, train_data, train_label1, train_label2, test_data, test_df, mod_dict, tcid_dom, num_epochs=100, beta=1.0, lambd=1.0, alpha=0.95):
    """
    Fine-tune the AE model on labeled data, using adversarial learning for unlabeled data.
    """
    # Labeled training set
    train_dataset = RNADataset(train_data, train_label1, train_label2)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    
    # Unlabeled training set
    # unlabeled_dataset = UnlabeledDataset(unlabeled_data)
    # unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=256, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.002)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_samples = 0
        correct1 = 0
        correct2 = 0
        
        # Train on labeled data
        for inputs, (labels1, labels2) in train_loader:
            inputs = inputs.to(DEVICE)
            labels1 = labels1.to(DEVICE)
            labels2 = labels2.to(DEVICE)
            optimizer.zero_grad()
            
            x_recon, logits1, logits2, _ = model(inputs)
            
            # Classification loss
            loss_cls1 = criterion1(logits1, labels1)
            loss_cls2 = criterion2(logits2, labels2)
            loss_cls = alpha * loss_cls1 + (1 - alpha) * loss_cls2
            
            # Reconstruction loss
            loss_recon = F.mse_loss(x_recon, inputs)
            
            # Total loss
            loss = loss_cls + beta * loss_recon
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            # Compute classification accuracy
            pred1 = logits1.argmax(dim=1)
            pred2 = logits2.argmax(dim=1)
            correct1 += (pred1 == labels1).sum().item()
            correct2 += (pred2 == labels2).sum().item()

        avg_loss = total_loss / total_samples
        acc1 = correct1 / total_samples
        acc2 = correct2 / total_samples
        if epoch % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc1: {acc1*100:.2f}%, Acc2: {acc2*100:.2f}%")

        # if epoch % 10 == 0:
        #     model.eval()
        #     _, _ = evaluate_multiclassification(model, test_data, test_df, mod_dict, tcid_dom, total_thr)
    
    return model


def assign_split(group):
    n = len(group)
    # If there is only 1 sample, mark as train
    if n == 1:
        group['if_train'] = 'train'
    else:
        # Calculate number of test samples, at least 1 sample
        n_test = max(1, int(round(n * 1/5)))
        # n_train = min(2000, int(round(n * 1/5)))
        # n_train = max(1, int(round(n * 2/5)))
        # If there are only 2 samples, ensure 1 train and 1 test
        if n == 2:
            n_test = 1
        # Randomly select test samples
        test_idx = group.sample(n=n_test, random_state=42).index
        group['if_train'] = 'train'
        group.loc[test_idx, 'if_train'] = 'test'
    return group

def get_new_trad_df(full_df, nega_usage):
    trad_df = full_df.copy()
    trad_df = trad_df[trad_df['label'] != 'unlabelled'].reset_index(drop=True)
    # trad_df['label2'] = trad_df['label'].str.split('_').str[0]
    trad_df['label2'] = trad_df['label'].apply(lambda x: x.split('_')[0] if '_' in x else 0)
    trad_df['label1'] = trad_df['label'].apply(lambda x: x.split('_')[1] if '_' in x else x)
    # trad_df = pd.concat([trad_df[trad_df['label1'] != 'm5C'], trad_df[trad_df['label1'] == 'm5C'].sample(n=int(1/3 * trad_df[trad_df['label1'] == 'm5C'].shape[0]))]).reset_index(drop=True)
    trad_df = pd.concat([trad_df[trad_df['label1'] != 'm6A'], trad_df[trad_df['label1'] == 'm6A'].sample(n=int(1/3 * trad_df[trad_df['label1'] == 'm6A'].shape[0]))]).reset_index(drop=True)
    # trad_df = pd.concat([trad_df[trad_df['label1'] != 'Nm'], trad_df[trad_df['label1'] == 'Nm'].sample(n=int(1/3 * trad_df[trad_df['label1'] == 'Nm'].shape[0]))]).reset_index(drop=True)

    trad_df['label'] = trad_df['label2'].astype(str) + '_' + trad_df['label1']
    # allowed_values = {'-2', '-1', '0', '+1', '+2', '+3'}
    # trad_df.loc[~trad_df['label2'].astype(str).isin(allowed_values), 'label2'] = '+3'
    trad_df['label2'] = trad_df['label2'].astype(int)
    trad_df['epoch'] = 0
    trad_df = trad_df.groupby('label', group_keys=False).apply(assign_split)
    trad_df = pd.concat([trad_df, nega_usage], axis=0).reset_index(drop=True)
    tqdm.pandas()
    # trad_seq_features = trad_df.progress_apply(lambda row: compute_5mer_smoothed_frequency(row['txome_11_mers'], sim_matrix, five_mers), axis=1)
    trad_seq_features = trad_df['txome_11_mers'].progress_apply(compute_5mer_frequency)
    array = pd.DataFrame(np.array(trad_seq_features.tolist()))
    array.columns = [f'New_Seq_{x}' for x in array]
    trad_df = pd.concat([trad_df, array], axis=1)
    return trad_df

def compute_5mer_frequency(seq):
    k = 4
    # Generate all possible 4-mers in lexicographical order (A, C, G, T)
    possible_kmers = [''.join(p) for p in product('ACGT', repeat=k)]
    possible_kmers.sort()  # Ensure correct sorting
    
    # Initialize all 4-mer counts to 0
    counts = {kmer: 0 for kmer in possible_kmers}
    
    # Count occurrences of each 4-mer in the sequence
    total_positions = len(seq) - k + 1
    for i in range(total_positions):
        kmer = seq[i:i+k]
        if kmer in counts:  # Ensure only A, C, G, T in sequence
            counts[kmer] += 1
    
    # Convert counts to frequencies (occurrences divided by total possible windows)
    # If sequence length is less than 5, total_positions<=0, return all-zero vector
    if total_positions > 0:
        freqs = [counts[kmer] for kmer in possible_kmers]
    else:
        freqs = [0] * len(possible_kmers)
    
    # Construct a 1-row, 1024-column matrix (numpy array)
    matrix = np.array(freqs).reshape(-1)
    return matrix

def get_data_features(df):

    trad_seq_features = df['txome_11_mers'].progress_apply(compute_5mer_frequency)
    array = pd.DataFrame(np.array(trad_seq_features.tolist()))
    array.columns = [f'New_Seq_{x}' for x in array]
    df = pd.concat([df, array], axis=1)

    data_features = np.concatenate([
        df[SEQUEN_COLS].values.astype(np.float32),
        df[SIGNAL_COLS].values.astype(np.float32),
        df[CHANGE_COLS].values.astype(np.float32)
    ], axis=1)

    return data_features

def get_mod_dict(test_df):
    mod_dict = dict()
    
    for i in set(test_df.set_index(['label1', 'label1_encoded']).index):
        mod_dict[i[1]] = i[0]
    tcid_dom = {v: k for k, v in mod_dict.items()}
    
    return mod_dict, tcid_dom

def get_openSet(mod, trad_df, prop):
    trad_df = trad_df.copy()
    # Get open set part (reserved for open set testing)
    openSet_test = trad_df[trad_df['label1'] == mod][SEQUEN_COLS + SIGNAL_COLS + CHANGE_COLS] \
                   .values.astype(np.float32)
    openSet_df = trad_df[trad_df['label1'] == mod].reset_index(drop=True)
    
    # Remove open set part
    trad_df = trad_df[trad_df['label1'] != mod]
    
    # Split training and test sets (original split basis)
    train_df = trad_df[trad_df['if_train'] == 'train'].reset_index(drop=True)
    test_df = trad_df[trad_df['if_train'] == 'test'].reset_index(drop=True)
    
    # ---- Training set balancing (oversampling) ----
    # Group by label1, sample each class to the maximum group size
    grouped = train_df.groupby('label1')
    max_count = int(grouped.size().max() * prop)
    balanced_train_df = pd.concat([
        group.sample(max_count, replace=True, random_state=42) 
        for _, group in grouped
    ]).reset_index(drop=True)
    balanced_train_df_m6A = pd.concat([balanced_train_df[balanced_train_df['label1'] == 'm6A'], train_df[train_df['label1'] == 'm6A']]).drop_duplicates()
    balanced_train_df = pd.concat([balanced_train_df[balanced_train_df['label1'] != 'm6A'], balanced_train_df_m6A]).reset_index(drop=True)
    # Shuffle order randomly
    balanced_train_df = balanced_train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    # ----------------------------------
    
    # Compute features for balanced train and test sets
    train_features = np.concatenate([
        balanced_train_df[SEQUEN_COLS].values.astype(np.float32),
        balanced_train_df[SIGNAL_COLS].values.astype(np.float32),
        balanced_train_df[CHANGE_COLS].values.astype(np.float32)
    ], axis=1)
    
    test_features = np.concatenate([
        test_df[SEQUEN_COLS].values.astype(np.float32),
        test_df[SIGNAL_COLS].values.astype(np.float32),
        test_df[CHANGE_COLS].values.astype(np.float32)
    ], axis=1)
    
    # Label encoding
    # Fit LabelEncoder on balanced training set to ensure all classes are mapped
    le_label1 = LabelEncoder()
    balanced_train_df['label1_enc'] = le_label1.fit_transform(balanced_train_df['label1'])
    # Transform test set with same mapping
    test_df['label1_enc'] = le_label1.transform(test_df['label1'])
    
    # Map label2
    label2_mapping = {-2: 0, -1: 1, 0: 2, 1: 3, 2: 4}
    balanced_train_df['label2_enc'] = balanced_train_df['label2'].map(label2_mapping)
    test_df['label2_enc'] = test_df['label2'].map(label2_mapping)
    
    train_labels1 = balanced_train_df['label1_enc'].values
    test_labels1 = test_df['label1_enc'].values
    train_labels2 = balanced_train_df['label2_enc'].values
    test_labels2 = test_df['label2_enc'].values
    
    num_classes = len(balanced_train_df['label1'].unique())
    
    test_df['label1_encoded'] = test_labels1
    test_df['label2_encoded'] = test_labels2
    
    return train_features, test_features, train_labels1, test_labels1, \
           train_labels2, test_labels2, openSet_test, openSet_df, test_df, balanced_train_df

def batch_prediction(full_data_features, full_model):
    full_model.eval()
    # Build mini-batch data loader (adjust batch_size according to GPU size)
    BATCH_SIZE = 512
    full_tensor = torch.tensor(full_data_features, dtype=torch.float32)
    dataset = TensorDataset(full_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Store inference outputs for each batch
    all_logits1 = []
    all_logits2 = []

    full_model.eval()
    with torch.no_grad():
        for batch in loader:
            t_batch = batch[0].to(DEVICE)
            _, logits1_batch, logits2_batch, _ = full_model(t_batch)
            all_logits1.append(logits1_batch.cpu())
            all_logits2.append(logits2_batch.cpu())

    # Concatenate full outputs
    logits1 = torch.cat(all_logits1, dim=0)
    logits2 = torch.cat(all_logits2, dim=0)

    return logits1, logits2

def get_real_df(logits11, logits2, m5c_df, mod_dict):

    m5c_df1 = m5c_df.copy()

    max_logits, pred_idx = logits11.max(dim=1)
    pred_label1 = [mod_dict[idx.item()] for logit, idx in zip(max_logits, pred_idx)]
    _, pred_idx2 = logits2.max(dim=1)
    pred_label2 = [idx.item() - 2 for logit, idx in zip(_, pred_idx2)]
    m5c_df1['preds1'] = pred_label1
    m5c_df1['preds2'] = pred_label2
    
    top3_values, top3_indices = logits11.topk(len(mod_dict), dim=1)
    
    out_df = pd.concat([pd.DataFrame(top3_indices.cpu().data.numpy(), columns=[f'ind_{x}' for x in range(0, len(mod_dict))]), pd.DataFrame(top3_values.cpu().data.numpy(), columns=[f'val_{x}' for x in range(0, len(mod_dict))])], axis=1)
    m5c_df1 = pd.concat([m5c_df1, out_df], axis=1)

    return m5c_df1

def manual_correct(t1, t2, kmer, bas_dict):
    if t1 == 'unlabelled':
        return True
    if bas_dict[t1] == 'N':
        return True
    else:
        if kmer[2-t2] == bas_dict[t1]:
            return True
        else:
            return False

def process_prediction(full_df, nega_usage, logits1, logits2, bas_dict, all_train_df, all_test_df, all_mod_dict, num_classes):

    full_df_meta = full_df[['id', 'position', 'kmer', 'contig', 'gen_position', 'strand', 'label', 'label1', 'label2', 'txome_11_mers']]

    usage_train = nega_usage[['id', 'position', 'if_train']].set_index(['id', 'position'])
    full_df_meta = pd.concat([full_df_meta.set_index(['id', 'position']), usage_train], axis=1)
    full_df_meta = full_df_meta.fillna('unimportant').reset_index()

    full_df_meta['usage'] = 'tbl'
    full_df_meta.loc[full_df_meta.set_index(['id', 'position']).index.isin(all_train_df.set_index(['id', 'position']).index), 'usage'] = 'train'
    full_df_meta.loc[full_df_meta.set_index(['id', 'position']).index.isin(all_test_df.set_index(['id', 'position']).index), 'usage'] = 'test'
    full_df_meta.loc[(full_df_meta.set_index(['id', 'position']).index.isin(nega_usage.set_index(['id', 'position']).index)) & (full_df_meta['if_train'] == 'train'), 'usage'] = 'nega_train'
    full_df_meta.loc[(full_df_meta.set_index(['id', 'position']).index.isin(nega_usage.set_index(['id', 'position']).index)) & (full_df_meta['if_train'] == 'test'), 'usage'] = 'nega_test'
    full_df_meta = get_real_df(logits1, logits2, full_df_meta, all_mod_dict)
    origin_vals = pd.DataFrame(logits1.numpy())
    origin_vals.columns = [f'origin_val_{x}' for x in range(0, num_classes)]
    full_df_meta = pd.concat([full_df_meta, origin_vals], axis=1)

    full_df_meta['manual'] = full_df_meta.apply(lambda row: manual_correct(row['preds1'], row['preds2'], row['kmer'], bas_dict), axis=1)
    full_df_meta['if_DRACH'] = full_df_meta.apply(lambda row: if_DRACH(row['txome_11_mers'], row['preds2'], row['preds1']), axis=1)

    full_df_meta.loc[full_df_meta['manual'] == False, 'preds1'] = 'unlabelled'
    full_df_meta.loc[full_df_meta['if_DRACH'] == 0, 'preds1'] = 'unlabelled'

    # full_df_meta = full_df_meta[full_df_meta['manual'] == True]
    # full_df_meta = full_df_meta[(full_df_meta['if_DRACH'] != 0) | (full_df_meta['usage'] != 'tbl')]

    return full_df_meta


def get_genome_kmer(genome, contig, gen_position):
    seq = genome.fetch(contig, gen_position-5, gen_position+6)
    return seq

def RF(i):
    o = ''
    for l in i:
        if l == 'A':
            o = 'T' + o
        elif l == 'T':
            o = 'A' + o
        elif l == 'C':
            o = 'G' + o
        elif l == 'G':
            o = 'C' + o
    return o

def get_RF(strand, mers):
    if strand == '+':
        return mers
    else:
        return RF(mers)

def phase_shift(p, s, i):
    if s == '+':
        return p + i
    else:
        return p - i
    
def phase_shift_r(p, s, i):
    if s == '+':
        return p - i
    else:
        return p + i

def get_mod(c, g, s, multi_answers):
    if (c, g, s) in multi_answers.index:
        return multi_answers.loc[(c, g, s)]['modification']
    else:
        return 'unlabelled'

def get_phase(c, g, s, multi_answers):
    if (c, g, s) in multi_answers.index:
        return multi_answers.loc[(c, g, s)]['phase']
    else:
        return 'unlabelled'

def get_multi_answer(answer_path):
    answers = pd.read_csv(answer_path, names=['contig', 'gen_position', 'strand', 'modification'])
    ans = dict()
    for i in [-2, -1, 0, 1, 2]:
        if i == 0:
            ans[i] = answers.copy()
            ans[i]['phase'] = i
        else:
            ans[i] = answers.copy()
            ans[i]['gen_position'] = ans[i].apply(lambda row: phase_shift(row['gen_position'], row['strand'], i), axis=1)
            ans[i]['phase'] = i
    
    multi_answers = pd.concat([ans[x] for x in ans], axis=0)
    multi_answers = multi_answers.drop_duplicates(['contig', 'gen_position', 'strand'])
    multi_answers = multi_answers.set_index(['contig', 'gen_position', 'strand'])

    return multi_answers

def full_feature(bascal_path, signal_path, answer_path, mod_num_threshold, ref):

    bas = pd.read_csv(bascal_path)
    sig = pd.read_csv(signal_path)
    multi_answers = get_multi_answer(answer_path)
    full_df = pd.concat([sig.set_index(['id', 'position', 'kmer']), bas.set_index(['id', 'position', 'kmer'])], axis=1).reset_index()
    genome = pysam.FastaFile(ref)  # Read indexed genome file
    full_df['11_mers'] = full_df.progress_apply(lambda row: get_genome_kmer(genome, row['contig'], row['gen_position']), axis=1)
    full_df['txome_11_mers'] = full_df.apply(lambda row: get_RF(row['strand'], row['11_mers']), axis=1)
    genome.close()

    full_df['modification'] = full_df.progress_apply(lambda row: get_mod(row['contig'], row['gen_position'], row['strand'], multi_answers), axis=1)
    full_df['phase'] = full_df.progress_apply(lambda row: get_phase(row['contig'], row['gen_position'], row['strand'], multi_answers), axis=1)
    full_df['label'] = full_df['phase'].astype(str) + '_' + full_df['modification']
    full_df.loc[full_df['label'] == 'unlabelled_unlabelled', 'label'] = 'unlabelled'
    # print(full_df['label'].value_counts())ll


    # full_df = pd.read_csv(full_df_path)
    full_df['label2'] = full_df['label'].apply(lambda x: x.split('_')[0] if '_' in x else 0)
    full_df['label1'] = full_df['label'].apply(lambda x: x.split('_')[1] if '_' in x else x)
    mod_counts = full_df['label1'].value_counts()
    mod_list = [x for x in mod_counts.index if mod_counts[x] >= mod_num_threshold]
    mod_list.sort()
    full_df = full_df[full_df['label1'].isin(mod_list)].reset_index(drop=True)
    # full_df['label'] = full_df['label2'].astype(str) + '_' + full_df['label1']
    # full_df.loc[full_df['label'] == 'unlabelled_unlabelled', 'label'] = 'unlabelled'
    full_data_features = get_data_features(full_df)
    return full_df, full_data_features, mod_list

def if_DRACH(motif, preds2, mod):
    if mod != 'm6A':
        return 2
    else:
        f = motif[3-preds2:8-preds2]
        if (f[0] in ['A', 'G', 'T']) and (f[1] in ['A', 'G']) and (f[2] == 'A') and (f[3] == 'C') and (f[4] in ['A', 'C', 'T']):
            return 1
        else:
            return 0

def train(full_data_features, all_train_data, all_train_label1, all_train_label2, all_test_data, all_test_df, num_classes, all_mod_dict, all_tcid_dom, model_path):
    start_time = time.time()
    full_model = DualOutputAutoEncoder(input_dim=311, num_classes=num_classes, latent_dim=128).to(DEVICE)
    full_model = pretrain_AE(full_model, full_data_features, num_epochs=200, batch_size=1024, beta=1.0, best_model_path=f'{model_path}/pretrained.AE.pth')
    full_model = fine_tune_AE_with_adversarial(full_model, all_train_data, all_train_label1, all_train_label2, all_test_data, all_test_df, all_mod_dict, all_tcid_dom, num_epochs=200, beta=0.8, lambd=.35, alpha=0.95)
    current_time = datetime.datetime.now()
    torch.save(full_model.state_dict(), f'{model_path}/Novel.best.AE.full.{str(current_time)}.pth')
    end_time = time.time()  # Record end time
    elapsed_min = (end_time - start_time) // 60  # Calculate elapsed minutes
    elapsed_sec = (end_time - start_time) % 60  # Calculate elapsed seconds
    print(f"Code execution time: {elapsed_min} min {elapsed_sec} sec")
    return full_model

def get_negative_trainset(full_df):

    nega_usage = full_df[full_df['label'] == 'unlabelled'].sample(n=int(0.05 * full_df.shape[0])).reset_index(drop=True)

    nega_usage['label1'] = 'unlabelled'
    nega_usage['label2'] = 0
    nega_usage['label'] = '0_unlabelled'
    nega_usage['epoch'] = 0
    nega_usage['if_train'] = 'train'
    nega_usage.loc[nega_usage.sample(n=int(nega_usage.shape[0] * 0.2)).index, 'if_train'] = 'test'

    return nega_usage

def get_final_mod(label_mod, label_phase, preds_mod, preds_phase):
    if label_mod == preds_mod:
        return label_mod, label_phase
    else:
        if label_mod != 'unlabelled':
            return label_mod, label_phase

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--answer_path', type=str, required=True, help='Path to the NGS-based answers')
    parser.add_argument('--ref_path', type=str, required=True, help='Path to the reference GENOME path')
    parser.add_argument('--threshold', type=int, default=50, help='Only consider modifications with at least this number of sites supported by NGS answers. Default: 50')
    parser.add_argument('--prefix', type=str, default='data', help='prefix of output file, please keep it THE SAME AS the one used in previous steps. Default: data')
    parser.add_argument('--work_dir', required=True, help='Working directory of your job, please keep it THE SAME AS the one used in previous steps. ')


    args = parser.parse_args()

    bascal_path = f'{args.work_dir}/{args.prefix}.annot.bascal.feature.per.site'
    signal_path = f'{args.work_dir}/{args.prefix}.annot.signal.feature.per.site'
    answer_path = args.answer_path
    output_path = f'{args.work_dir}/{args.prefix}.annotation.per.site'
    model_path = args.work_dir
    mod_num_threshold = args.mod_num_threshold
    reference = args.ref_path

    bas_dict = {
        'm5U': 'T', 'm6Am': 'A', 'otherMod': 'N', 'm7G': 'G',
        'Nm': 'N', 'm1A': 'A', 'm5C': 'C', 'pseudoU': 'T',
        'm6A': 'A', 'Inosine': 'A', 'unlabelled': 'N',
        'unknown': 'unknown'
    }

    full_df, full_data_features, mod_list = full_feature(bascal_path, signal_path, answer_path, mod_num_threshold, reference)
    num_classes = len(mod_list)

    nega_usage = get_negative_trainset(full_df)
    trad_df = get_new_trad_df(full_df, nega_usage)
    all_train_data, all_test_data, all_train_label1, all_test_label1, all_train_label2, all_test_label2, _, _, all_test_df, all_train_df = get_openSet('All', trad_df, 0.85)
    all_mod_dict, all_tcid_dom = get_mod_dict(all_test_df)
    
    full_model = train(full_data_features, all_train_data, all_train_label1, all_train_label2, all_test_data, all_test_df, num_classes, all_mod_dict, all_tcid_dom, model_path )
    full_model.eval()
    
    logits1, logits2 = batch_prediction(full_data_features, full_model)
    full_df_meta = process_prediction(full_df, nega_usage, logits1, logits2, bas_dict, all_train_df, all_test_df, all_mod_dict, num_classes)
    full_df_meta['preds2'] = full_df_meta['preds2'].astype(int)
    full_df_meta['label2'] = full_df_meta['label2'].astype(int)

    preds = full_df_meta[(full_df_meta['label1'] == 'unlabelled') & (full_df_meta['preds1'] != 'unlabelled')]
    preds.loc[:, 'position'] = preds['position'] - preds['preds2']
    preds.loc[:, 'gen_position'] = preds.apply(lambda row: phase_shift_r(row['gen_position'], row['strand'], row['preds2']), axis=1)
    preds.loc[:, 'kmer'] = preds.apply(lambda row: get_genome_kmer(pysam.FastaFile(reference), row['contig'], row['gen_position']), axis=1)
    preds.loc[:, 'kmer'] = preds.apply(lambda row: get_RF(row['strand'], row['kmer']), axis=1)
    preds = preds[['id', 'position', 'kmer', 'contig', 'gen_position', 'strand', 'preds1']]
    preds = preds.rename(columns={'preds1': 'modification'})
    preds['source'] = 'predicted'

    labels = full_df_meta[full_df_meta['label1'] != 'unlabelled']
    labels.loc[:, 'position'] = labels['position'] - labels['label2']
    labels.loc[:, 'gen_position'] = labels.apply(lambda row: phase_shift_r(row['gen_position'], row['strand'], row['label2']), axis=1)
    labels.loc[:, 'kmer'] = labels.apply(lambda row: get_genome_kmer(pysam.FastaFile(reference), row['contig'], row['gen_position']), axis=1)
    labels.loc[:, 'kmer'] = labels.apply(lambda row: get_RF(row['strand'], row['kmer']), axis=1)
    labels = labels[['id', 'position', 'kmer', 'contig', 'gen_position', 'strand', 'label1']]
    labels = labels.rename(columns={'label1': 'modification'})
    labels['source'] = 'labelled'

    full_df = pd.concat([preds, labels], axis=0).reset_index(drop=True)
    full_df.to_csv(output_path, index=False)

if __name__ == '__main__':
    main()
