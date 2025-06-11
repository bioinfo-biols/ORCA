import argparse
import numpy as np
import pandas as pd
import os,re
import multiprocessing 
from io import StringIO
from tqdm import tqdm
import warnings
from pandas.errors import PerformanceWarning
from scipy.interpolate import interp1d
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*swapaxes.*")
warnings.simplefilter('ignore', PerformanceWarning)
# from Multiprocessing_99 import *

def index(eventalign_result, pos_start, out_path):
    eventalign_result = eventalign_result.set_index(['contig','read_index'])
    pos_end = pos_start
    written_lines = []

    with open(out_path, 'a') as f_index:
        for index in list(dict.fromkeys(eventalign_result.index)):
            transcript_id, read_index = index
            pos_end += eventalign_result.loc[index]['line_length'].sum()
            try:
                f_index.write('%s,%d,%d,%d\n' % (transcript_id, read_index, pos_start, pos_end))
            except:
                pass
            pos_start = pos_end

    return pos_end

def index_worker(args):
    chunk_df, pos_start, out_path = args
    return index(chunk_df, pos_start, out_path) - pos_start

def parallel_index(eventalign_filepath, chunk_size, output_path, n_processes):
    out_path = os.path.join(output_path, 'eventalign.index')
    file_size = os.path.getsize(eventalign_filepath)

    with open(out_path, 'w') as f:
        f.write('id,read_index,pos_start,pos_end\n')

    pool = multiprocessing.Pool(processes=n_processes)
    results = []
    pos_start = 0
    chunk_split = None
    index_features = ['contig', 'read_index', 'line_length']

    with open(eventalign_filepath, 'r') as f:
        header = f.readline()
        pos_start += len(header)
        f_text = f

        with tqdm(total=file_size, desc="Indexing") as pbar:
            def update_pbar(bytes_processed):
                pbar.update(bytes_processed)

            reader = pd.read_csv(eventalign_filepath, sep='\t', chunksize=chunk_size)

            for chunk in reader:
                chunk_complete = chunk[chunk['read_index'] != chunk.iloc[-1]['read_index']]
                chunk_concat = pd.concat([chunk_split, chunk_complete])
                chunk_concat_size = len(chunk_concat.index)

                lines = [len(f_text.readline()) for _ in range(chunk_concat_size)]
                chunk_concat['line_length'] = np.array(lines)

                pool.apply_async(
                    index_worker,
                    args=((chunk_concat[index_features], pos_start, out_path),),
                    callback=update_pbar
                )

                pos_start += sum(lines)
                chunk_split = chunk[chunk['read_index'] == chunk.iloc[-1]['read_index']]

            if chunk_split is not None and not chunk_split.empty:
                lines = [len(f_text.readline()) for _ in range(len(chunk_split))]
                chunk_split.loc[:, 'line_length'] = np.array(lines)

                pool.apply_async(
                    index_worker,
                    args=((chunk_split[index_features], pos_start, out_path),),
                    callback=update_pbar
                )

    pool.close()
    pool.join()




def get_df(events_str):
    f_string = StringIO(events_str)
    eventalign_result = pd.read_csv(f_string, delimiter='\t', names=[
        'contig','position','reference_kmer','read_index',
        'strand','event_index','event_level_mean','event_stdv',
        'event_length','model_kmer','model_mean','model_stdv',
        'standardized_level','start_idx','end_idx'
    ])
    eventalign_result['id'] = eventalign_result['contig']
    eventalign_result['position'] = eventalign_result['position'].astype(int) + 2

    features = [
        'id', 'position', 'reference_kmer',
        'event_level_mean', 'event_stdv', 'read_index'
    ]
    return eventalign_result[features]

def _preprocess_worker(args):
    tx_id, index_info, eventalign_filepath, out_paths, refk_csv, locks = args
    # try:
    data_dict = {}
    with open(eventalign_filepath, 'r') as f_eventalign:
        for _, row in index_info.iterrows():
            pos_start = row['pos_start']
            pos_end = row['pos_end']
            f_eventalign.seek(pos_start)
            event_str = f_eventalign.read(pos_end - pos_start)
            data = get_df(event_str)
            if not data.empty:
                data_dict[row['read_index']] = data
    
    if not data_dict:
        return 0
    return preprocess_tx(tx_id, data_dict, out_paths, refk_csv, locks)
    # except Exception as e:
    #     print(f"Error processing {tx_id}: {str(e)}")
    #     return 0

def parallel_preprocess_tx(eventalign_filepath, output_path, prefix, n_processes, refk_csv):
    os.makedirs(output_path, exist_ok=True)
    out_paths = {
        'csv': f'{output_path}/{prefix}.signal.feature.per.site', 
        'index': f'{output_path}/{prefix}.signal.feature.index', 
        'signal': f'{output_path}/{prefix}.data.for.annotaion'
    }

    # Use Manager to create cross-process locks
    with multiprocessing.Manager() as manager:
        locks = {
            'csv': manager.Lock(),
            'index': manager.Lock(),
            'signal': manager.Lock()
        }

        # Initialize file headers
        with open(out_paths['csv'], 'w') as f:
            f.write('id,position,kmer,' + ','.join(f'{x}_shape' for x in range(50)) + '\n')
        with open(out_paths['index'], 'w') as f:
            f.write('id,start,end\n')
        with open(out_paths['signal'], 'w') as f:
            f.write('id\tposition\tkmer\tmean\tstdv\tread_index\n')

        # Read index
        index_path = os.path.join(output_path, 'eventalign.index')
        df_index = pd.read_csv(index_path).set_index('id')
        tx_ids = df_index.index.unique().tolist()

        # Prepare tasks
        tasks = [
            (tx_id, 
             df_index.loc[[tx_id]].reset_index(),
             eventalign_filepath,
             out_paths,
             refk_csv,
             locks)
            for tx_id in tx_ids
        ]

        # Execute parallel tasks
        with multiprocessing.Pool(n_processes) as pool:
            results = []
            with tqdm(total=len(tasks), desc="Processing transcripts") as pbar:
                for result in pool.imap_unordered(_preprocess_worker, tasks):
                    pbar.update(1)
                    results.append(result)
        
        print(f"Successfully processed {sum(results)}/{len(tasks)} transcripts")

def resample_array_spline(x, num_points=50):
    if len(x) < 4:
        return np.linspace(np.min(x), np.max(x), num_points)
    x_sorted = np.sort(x)
    new_indices = np.linspace(0, len(x_sorted)-1, num_points)
    return interp1d(np.arange(len(x_sorted)), x_sorted, 'cubic')(new_indices)

def preprocess_tx(tx_id, data_dict, out_paths, refk_csv, locks):
    # try:
    events = pd.concat(data_dict.values(), axis=0)
    if events.empty:
        return 0

    sorted_idx = np.argsort(events['position'])
    unique_pos, split_idx = np.unique(events['position'].iloc[sorted_idx], return_index=True)
    
    y_arrays = np.split(events['event_level_mean'].iloc[sorted_idx], split_idx[1:])
    x_arrays = np.split(events['event_stdv'].iloc[sorted_idx], split_idx[1:])
    n_arrays = np.split(events['read_index'].iloc[sorted_idx], split_idx[1:])
    kmers = np.split(events['reference_kmer'].iloc[sorted_idx], split_idx[1:])

    with locks['csv'], open(out_paths['csv'], 'a') as f_csv, locks['signal'], open(out_paths['signal'], 'a') as f_sig:
        
        pos_start = f_csv.tell()
        
        for pos, y_arr, x_arr, n_arr, kmer_arr in zip(unique_pos, y_arrays, x_arrays, n_arrays, kmers):
            if len(y_arr) < 10:
                continue
            assert len(set(kmer_arr)) == 1
            kmer = kmer_arr.iloc[0]
            ref_mean = refk_csv.loc[kmer, 'model_mean']
            ref_stdv = refk_csv.loc[kmer, 'model_stdv']
            
            stdl = ((y_arr - ref_mean) / ref_stdv).sort_values().values
            interpolated = resample_array_spline(stdl)
            
            f_csv.write(f"{tx_id},{pos},{kmer}," + ",".join(f"{x:.4f}" for x in interpolated) + "\n")
            f_sig.write(f"{tx_id}\t{pos}\t{kmer}\t{list(y_arr)}\t{list(x_arr)}\t{list(n_arr)}\n")
        pos_end = f_csv.tell()
    if pos_start != pos_end:
        with locks['index'], open(out_paths['index'], 'a') as f_idx:
            f_idx.write(f"{tx_id},{pos_start},{pos_end}\n")
    
    return 1
    
def dataprep(args):
    #
    n_processes = args.n_processes
    eventalign_filepath = args.eventalign
    chunk_size = args.chunk_size
    output_path = args.work_dir
    prefix = args.prefix
    refk_csv = pd.read_csv(f'{os.path.dirname(__file__)}/ref_kmer.csv', index_col=['model_kmer'])
    os.makedirs(output_path, exist_ok=True)

    
    
    parallel_index(eventalign_filepath,chunk_size,output_path,n_processes)
    parallel_preprocess_tx(eventalign_filepath,output_path,prefix,n_processes, refk_csv)


def main():
    parser = argparse.ArgumentParser(description="Extract signal alignment features from the eventalign results.")
    parser.add_argument('--n_processes', type=int, default=cpu_count(), help='Number of parallel processes. Default: All available CPU cores')
    parser.add_argument('--eventalign', type=str, required=True, help='Path to the eventalign file.')
    parser.add_argument('--chunk_size', type=int, default=100000, help='Chunk size for reading eventalign files for indexing. Default: 100000')
    parser.add_argument('--prefix', type=str, default='data', help='prefix of output file, please keep it THE SAME AS the one used in previous steps. Default: data')
    parser.add_argument('--work_dir', required=True, help='Working directory of your job, please keep it THE SAME AS the one used in previous steps. ')

    args = parser.parse_args()
    dataprep(args)

if __name__ == '__main__':
    main()