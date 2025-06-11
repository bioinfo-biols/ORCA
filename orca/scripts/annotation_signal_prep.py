import pandas as pd
import os
from sklearn.mixture import GaussianMixture
import numpy as np
import ast
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
import argparse

# Global variables for subprocess initialization
refk_csv = None
s1_indexed = None

def init_process():
    global refk_csv
    refk_csv = pd.read_csv(
        '/histor/zhao/donghan/.conda/envs/New_xPore/lib/python3.8/site-packages/xpore/diffmod/model_kmer.csv',
        sep=',', 
        index_col=['model_kmer'])

def init_process_for_flatten(s1_indexed_path):
    global s1_indexed
    s1_indexed = pd.read_csv(s1_indexed_path, index_col=['id', 'position'])

def GMM_3D_Training(signals, norm_stdv):
    """Keep the original training function unchanged"""
    gmm1 = GaussianMixture(n_components=1, covariance_type='full', init_params='k-means++')
    gmm1.fit(signals)
    
    F1_Mean = gmm1.means_[0][0]
    F2_Mean = (gmm1.means_[0][1] - norm_stdv) / ((gmm1.means_[0][1] + norm_stdv) / 2)
    F1_Stdv = np.sqrt(gmm1.covariances_[0][0][0])
    F2_Stdv = np.sqrt(gmm1.covariances_[0][1][1])
    F1F2_CoStdv = gmm1.covariances_[0][0][1]

    return [F1_Mean, F2_Mean, F1_Stdv, F2_Stdv, F1F2_CoStdv]

def GMM_execute_wrapper(args):
    """Wrapper function for argument unpacking"""
    return GMM_execute(*args)

def GMM_execute(mean_list, stdv_list, kmer):
    """Modified execution function (keep original logic)"""
    mean_list = np.array(ast.literal_eval(mean_list))
    stdv_list = np.array(ast.literal_eval(stdv_list))
    
    model_mean = refk_csv.loc[kmer]['model_mean']
    model_stdv = refk_csv.loc[kmer]['model_stdv']
    
    mean_list -= model_mean
    stdv_list = np.log(stdv_list + 0.001)
    
    signals = np.vstack((mean_list, stdv_list)).T
    norm_stdv = np.log(model_stdv + 0.001)
    
    feat_list = GMM_3D_Training(signals, norm_stdv)
    return feat_list

def parallel_processing(s, num_processes=16):
    """Main function for parallel processing"""
    # Prepare parameter list
    params = s[['mean', 'stdv', 'kmer']].itertuples(index=False, name=None)
    
    # Create process pool
    with mp.Pool(processes=num_processes, initializer=init_process) as pool:
        # Use imap for processing and show progress bar
        results = list(tqdm(pool.imap(GMM_execute_wrapper, params, chunksize=100),
                        total=len(s),
                        desc="Processing rows"))
    
    # Add results to DataFrame
    result_columns = ['Mean_average', 'Stdv_average', 'Mean_svar', 'Stdv_svar', 'Cov']
    s[result_columns] = pd.DataFrame(results, index=s.index)
    return s


def process_item(task):
    idx, position = task
    try:
        # Extract rows centered at current (id, position), window size 5 (position±2)
        window = s1_indexed.loc[[
            (idx, position-2),
            (idx, position-1),
            (idx, position),
            (idx, position+1),
            (idx, position+2)
        ]]
    except KeyError:
        return None  # Skip if any index is missing
    if window.shape[0] != 5:
        return None   # Skip if window has less than 5 rows
    # kmer value of the center row
    kmer_val = window.iloc[2]['kmer']
    # Extract the last 5 feature columns in the window and flatten to a 1D list
    features_flat = window.iloc[:, -5:].to_numpy().reshape(-1).tolist()
    return [idx, position, kmer_val] + features_flat

def get_signal_raw(target, signal_path, outs):

    print('Getting Positive Positions...')
    t = pd.read_csv(target)
    t = t[t['modScore'] >= .9]
    t1 = t.set_index(['id', 'position'])
    t2 = t[['id', 'position']]
    mt = pd.concat([t2, t2.assign(position = t2['position'] + 1), t2.assign(position = t2['position'] + 2), t2.assign(position = t2['position'] - 1), t2.assign(position = t2['position'] - 2)])
    tot = set(mt.set_index(['id', 'position']).index)
    print('Getting Signals...')

    with open(signal_path, 'r') as f, open(outs, 'w') as fa:
        header = f.readline()
        # bar.update(len(header))
        fa.write(header)
        
        for line in f:
            # bar.update(len(line))
            line1 = line.split('\t')
            if (line1[0], int(line1[1])) in tot:
                fa.write(line)

    signal = pd.read_csv(outs, sep='\t')
    os.remove(outs)
    signal = parallel_processing(signal)
    signal['id'] = signal['id'].str.split('|').str[0]
    s1 = signal[['id', 'position', 'kmer', 'Mean_average', 'Stdv_average', 'Mean_svar', 'Stdv_svar', 'Cov']]
    o = t.copy()
    # o = o[o['modScore'] >= .9]
    o['id'] = o['id'].str.split('|').str[0]
    o = o.set_index(['id', 'position'])
    
    s1['id'] = s1['id'].str.split('|').str[0]
    s2 = s1[s1.set_index(['id', 'position']).index.isin(o.index)].reset_index(drop=True)
    
    return s1, s2


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--prefix', type=str, default='data', help='prefix of output file, please keep it THE SAME AS the one used in previous steps. Default: data')
    parser.add_argument('--work_dir', required=True, help='Working directory of your job, please keep it THE SAME AS the one used in previous steps. ')
    args = parser.parse_args()

    flat = f'{args.work_dir}/{args.prefix}.annot.signal.feature.per.site'
    outs = f'{args.work_dir}/{args.prefix}.annot.signal.temp'
    target = f'{args.work_dir}/{args.prefix}.preds.per.site'
    signal_path = f'{args.work_dir}/{args.prefix}.data.for.annotaion'
    s1_indexed_path = f'{args.work_dir}/s1.indexed.temp.csv'

    s1, s2 = get_signal_raw(target, signal_path, outs)
    s1_indexed = s1.set_index(['id', 'position'])
    s1_indexed.to_csv(s1_indexed_path)

    print('Flattening.....')


    tasks = list(zip(s2['id'], s2['position']))
    trad_list = []

    with Pool(cpu_count(), initializer=init_process_for_flatten, initargs=(s1_indexed_path,)) as pool:
        for result in tqdm(pool.imap(process_item, tasks, chunksize=1000),
                           total=len(tasks), desc="提取特征", unit="个", unit_scale=True):
            if result is not None:
                trad_list.append(result)


    new_cols = [f'{off}_{col}' for off in [-2, -1, 0, 1, 2] for col in s1.columns[-5:]]
    trad_df = pd.DataFrame(trad_list, columns=['id', 'position', 'kmer'] + new_cols)

    trad_df.to_csv(flat, index=False)
    os.remove(s1_indexed_path)

if __name__ == '__main__':
    main()