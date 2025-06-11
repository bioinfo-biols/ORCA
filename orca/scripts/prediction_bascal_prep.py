import pysam
import numpy as np
from tqdm import tqdm
import os
import multiprocessing
import sys
import pandas as pd
import re
from collections import Counter
import argparse
from multiprocessing import cpu_count


def index_pileup(pileup, ind):
    indbar = tqdm(total=os.path.getsize(pileup))
    with open(pileup, 'r') as f, open(ind, 'w') as fa:
        fa.write('id,start,end\n')
        old_id = ''
        start = 0
        end = 0
        for line in f:
            indbar.update(len(line))
            line1 = line.split('\t')
            idx = line1[0]
            if old_id == '':
                old_id = idx
                end += len(line)
            else:
                if idx == old_id:
                    end += len(line)
                else:
                    fa.write(f'{old_id},{start},{end}\n')
                    start = end
                    end += len(line)
                    old_id = idx

def quality_scores(quality_string):
    return [ord(char) - 33 for char in quality_string]

def get_splited(s):
    out_list = []
    val_list = []
    dc = 0
    mic = 0
    mac = 0
    ic = 0
    ifpass = 0
    
    for i in range(len(s)):
        if ifpass > 0:
            ifpass -= 1
            continue
        char = s[i]
        if char == '*':
            out_list.append(char)
            val_list.append('Del')
            dc += 1
        elif char in ['.', ',']:
            if i == len(s) - 1:
                out_list.append(char)
                val_list.append('Mat')
                mac += 1
            else:
                next_char = s[i+1]
                if next_char not in '+-':
                    out_list.append(char)
                    val_list.append('Mat')
                    mac += 1
                else:
                    num_chars = int(s[i+2]) if i+2 < len(s) else 0
                    seq_length = num_chars
                    snippet = s[i:i+3+seq_length]
                    out_list.append(snippet)
                    ifpass = 2 + seq_length
                    if next_char == '+':
                        val_list.append('Ins')
                        ic += 1
                    else:
                        val_list.append('Mat')
                        mac += 1
        elif char == '>':
            out_list.append(char)
            val_list.append('Ski')
        elif char in ['A', 'T', 'C', 'G']:
            out_list.append(char)
            val_list.append('Mis')
            mic += 1
    depth = dc + mic + mac + ic
    if depth == 0:
        depth = 1
    mismatch_ratio = mic / depth
    insertion_ratio = ic / depth
    deletion_ratio = dc / depth
    return pd.Series([depth, mismatch_ratio, insertion_ratio, deletion_ratio, out_list, val_list])

# def get_new_quals(val_list, qs):
#     new_qs = []
#     q_iter = iter(qs)
#     for val in val_list:
#         if val != 'Ski':
#             try:
#                 new_qs.append(next(q_iter))
#             except StopIteration:
#                 break
#         else:
#             next(q_iter)
#     return ''.join(new_qs)

def get_new_quals(val_list, qs):
    new_qs = ''
    for val, q in zip(val_list, qs):
        if val != 'Ski':
            new_qs += q
    return new_qs

def calculate_quality_stats(quality_scores):
    if not quality_scores:
        return pd.Series([np.nan, np.nan, np.nan])
    return pd.Series([
        np.mean(quality_scores),
        np.median(quality_scores),
        np.std(quality_scores)
    ])

def process_transcript(transcript_id, start_byte, end_byte, pileup_path, output_path, index_path, lock):
    output_lines = []
    with open(pileup_path, 'rb') as f:
        f.seek(start_byte)
        data = f.read(end_byte - start_byte)
    for line in data.decode().splitlines():
        parts = line.split('\t')
        if len(parts) != 6:
            continue
        tx_id, pos, ref, _, cig, qual = parts
        pos = int(pos) - 1
        
        depth, mis_ratio, ins_ratio, del_ratio, _, val_list = get_splited(cig)
        if depth < 10:
            continue
        
        filtered_qual = get_new_quals(val_list, qual)
        qual_scores = quality_scores(filtered_qual)
        q_mean, q_median, q_std = calculate_quality_stats(qual_scores)
        
        output_lines.append(
            f"{tx_id},{pos},{ref},{depth},{mis_ratio},{ins_ratio},{del_ratio},"
            f"{q_mean},{q_median},{q_std}\n"
        )
    
    if output_lines:
        with lock:
            with open(output_path, 'a') as f_out, open(index_path, 'a') as f_idx:
                start_pos = f_out.tell()
                f_out.writelines(output_lines)
                end_pos = f_out.tell()
                f_idx.write(f"{transcript_id},{start_pos},{end_pos}\n")
    return len(output_lines)
    
def multi_get_bascal_err(output, output_index, pileup, pileup_index, n_processes):
    # Initialize output files
    with open(output, 'w') as f:
        f.write('id,position,ref,depth,Mismatch_Ratio,Insertion_Ratio,Deletion_Ratio,Qual_Mean,Qual_Median,Qual_Stdv\n')
    with open(output_index, 'w') as f:
        f.write('id,start,end\n')

    # Read index file
    index_df = pd.read_csv(pileup_index)
    total_transcripts = len(index_df)

    # Create progress bar
    with tqdm(total=total_transcripts, desc="Processing transcripts") as pbar:
        # Create pool and manager
        with multiprocessing.Pool(n_processes) as pool:
            manager = multiprocessing.Manager()
            lock = manager.Lock()
            
            # Create task queue
            tasks = [
                (row['id'], row['start'], row['end'], pileup, output, output_index, lock)
                for _, row in index_df.iterrows()
            ]
            
            # 使用partial函数处理额外的pbar参数
            def update_progress(_):
                pbar.update(1)
            
            # Submit tasks
            results = []
            for task in tasks:
                res = pool.apply_async(process_transcript, task, callback=update_progress)
                results.append(res)
            
            # Wait for all tasks to complete
            for res in results:
                res.get()
            
            # Clean up
            pool.close()
            pool.join()


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--pileup', required=True, help='pileup FILE PATH from samtools mpileup')
    parser.add_argument('--prefix', type=str, default='data', help='prefix of output file, please keep it THE SAME AS the one used in previous steps. Default: data')
    parser.add_argument('--work_dir', required=True, help='Working directory of your job, please keep it THE SAME AS the one used in previous steps. ')
    parser.add_argument('--n_processes', type=int, default=cpu_count(), help='Number of parallel processes. Default: All available CPU cores')
    args = parser.parse_args()

    output_path = args.work_dir
    prefix = args.prefix
    output_feature = f'{output_path}/{prefix}.bascal.feature.per.site'
    output_index = f'{output_path}/{prefix}.bascal.feature.index'
    pileup = args.pileup
    pileup_index = f'{pileup}.index'

    index_pileup(pileup, pileup_index)
    multi_get_bascal_err(output_feature, output_index, pileup, pileup_index, args.n_processes)


if __name__ == '__main__':
    main()