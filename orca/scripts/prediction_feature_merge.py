import pandas as pd
from tqdm import tqdm
import os
import sys
from io import StringIO
import numpy as np
import argparse
from multiprocessing import Pool, Manager, Lock, cpu_count


# Define constants
Index = ['id', 'position', 'kmer', 'depth']
onepos = ['Mismatch_Ratio', 'Insertion_Ratio', 'Deletion_Ratio', 'Qual_Mean', 
          'Qual_Median', 'Qual_Stdv'] + [f'{x}_shape' for x in range(50)]
mer_cols = Index + onepos

def get_one_tx_df(path, start, end, header):
    with open (path, 'r') as f:
        f.seek(start)
        s = f.read(end - start)
        df = pd.read_csv(StringIO(s), sep=',', names=header, index_col=['id', 'position'])
    return df

def process_txid(args):
    """Main function to process a single txid"""
    txid, signal_path, bascal_path, output_path, signal_dict, bascal_dict, lock, sig_header, bas_header = args
    # try:
    # Get index information
    sig_info = signal_dict.get(txid)
    bas_info = bascal_dict.get(txid)
    # if not sig_info or not bas_info:
    #     return 0

    # Read data
    one_sig_df = get_one_tx_df(signal_path, sig_info['start'], sig_info['end'], sig_header)
    one_bas_df = get_one_tx_df(bascal_path, bas_info['start'], bas_info['end'], bas_header).drop(['ref'], axis=1)
    one_bas_df.columns = ['depth'] + onepos[:6]  # First 6 features

    # Merge data
    one_mer_df = pd.concat([one_bas_df, one_sig_df], axis=1, join='inner').reset_index()
    one_mer_df = one_mer_df[mer_cols]

    one_tx_out = ''
    for i in range(0, one_mer_df.shape[0] - 4):
        one5 = one_mer_df.iloc[i:i+5]
        if one5.iloc[0]['position'] + 4 != one5.iloc[-1]['position']:
            continue
        meta = one5.iloc[2][Index]
        feat = one5.drop(Index, axis=1).values.reshape(-1)
        combined_row = np.concatenate([meta, feat])
        combined_row_str = ','.join(map(str, combined_row)) + '\n'
        one_tx_out += combined_row_str
        # break
    with lock, open (output_path, 'a') as out:
        out.write(one_tx_out)

        
    return 1
    # except Exception as e:
    #     print(f"Error processing {txid}: {str(e)}")
    #     return 0

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='data', help='prefix of output file, please keep it THE SAME AS the one used in previous steps. Default: data')
    parser.add_argument('--work_dir', required=True, help='Working directory of your job, please keep it THE SAME AS the one used in previous steps. ')
    parser.add_argument('--n_processes', type=int, default=cpu_count(), help='Number of parallel processes. Default: All available CPU cores')
    args = parser.parse_args()

    signal_index = f'{args.work_dir}/{args.prefix}.signal.feature.index'
    bascal_index = f'{args.work_dir}/{args.prefix}.bascal.feature.index'
    signal_data = f'{args.work_dir}/{args.prefix}.signal.feature.per.site'
    bascal_data = f'{args.work_dir}/{args.prefix}.bascal.feature.per.site'
    output = f'{args.work_dir}/{args.prefix}.merged.feature.per.site'

    # Read index
    signal_index = pd.read_csv(signal_index, index_col=['id'])
    bascal_index = pd.read_csv(bascal_index, index_col=['id'])
    valid_idx = list(set(signal_index.index) & set(bascal_index.index))

    # Read CSV header
    with open(signal_data, 'r') as f:
        sig_header = f.readline().strip().split(',')
    with open(bascal_data, 'r') as f:
        bas_header = f.readline().strip().split(',')

    # Initialize output file
    with open(output, 'w') as f:
        header = Index + [f"{p}_{f}" for p in ['-2', '-1', '0', '+1', '+2'] for f in onepos]
        f.write(','.join(header) + '\n')

    # Create shared objects
    manager = Manager()
    lock = manager.Lock()
    signal_dict = signal_index.to_dict('index')
    bascal_dict = bascal_index.to_dict('index')

    # Create task queue
    tasks = [(txid, signal_data, bascal_data, output, 
             signal_dict, bascal_dict, lock, sig_header, bas_header) 
            for txid in valid_idx]

    # Start process pool
    with Pool(processes=args.n_processes) as pool:
        with tqdm(total=len(valid_idx)) as pbar:
            for result in pool.imap_unordered(process_txid, tasks):
                pbar.update(result)

if __name__ == '__main__':
    main()