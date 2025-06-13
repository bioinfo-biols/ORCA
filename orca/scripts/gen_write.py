import pandas as pd
from io import StringIO
from .Multiprocessing_99 import *
from tqdm import tqdm
import re
from multiprocessing import cpu_count

def get_gtf_indexed(gtf_path):
    exon_path = f'{gtf_path}.exon'
    index_path = f'{gtf_path}.exon.index'
    with open (gtf_path, 'r') as gtf, open(exon_path, 'w') as exon, open(index_path, 'w') as index:
        exon.write('contig,source,type,start,end,unk1,strand,unk2,meta,txid\n')
        index.write('txid,start,end\n')
        old_txid = ''
        for line in gtf:
            if line[0] == '#':
                continue
            writ = ','.join(line.rstrip().split('\t'))
            clas = line.split('\t')[2]
            if clas == 'exon':
                # match = re.search(r'transcript_id\s+"([^"]+)"', line)
                match = re.search(r'ENST\d+(?:\.\d+)?', line)
                txid = match.group(0)
                if txid != old_txid:
                    if old_txid != '':
                        end = exon.tell()
                        index.write(f'{old_txid},{start},{end}\n')
                        start = end
                        exon.write(f'{writ},{txid}\n')
                        old_txid = txid
                    else:
                        start = exon.tell()
                        exon.write(f'{writ},{txid}\n')
                        old_txid = txid
                else:
                    exon.write(f'{writ},{txid}\n')
        end = exon.tell()
        index.write(f'{old_txid},{start},{end}\n')
        

def ed_read(txid, exon, exon_index):
    start = exon_index.loc[txid]['start']
    end = exon_index.loc[txid]['end']
    # print(start)
    exon.seek(start, 0)
    ed_str = exon.read(end - start)
    ed_df = pd.read_csv(StringIO(ed_str), sep=',', names=['contig', 'source', 'type', 'start', 'end', 'unk1', 'strand', 'unk2', 'meta', 'txid'])
    
    return ed_df
    
def gen_write(txome, exon_path, exon_index, output_path, sep, locks):
    with open (exon_path) as exon:
        ed = dict()
        for tx in txome['id'].unique():
            ed[tx] = ed_read(tx, exon, exon_index)
        contig_list = list()
        genpo_list = list()
        strands = list()
        Hikari = 0
        ifbreak = False
        for line in txome.iterrows():
            Hikari += 1
            _, row = line
            txid = row['id']
            position = row['position']
            exondf = ed[txid]
            contig = exondf.iloc[0]['contig']
            strand = exondf.iloc[0]['strand']
            if strand == '+':
                for _, e_row in exondf.iterrows():
                    onelen = e_row['end'] - (e_row['start']-1)
                    if position <= onelen:
                        genpo = position + (e_row['start']-1)
                        break
                    elif position > onelen:
                        position -= onelen
            elif strand == '-':
                ifbreak = True
                for _, e_row in exondf.iterrows():
                    onelen = e_row['end'] - (e_row['start']-1)
                    if position <= onelen:
                        genpo = e_row['end'] - position
                        break
                    elif position > onelen:
                        position -= onelen
                genpo -= 1
            contig_list.append(contig)
            genpo_list.append(genpo)
            strands.append(strand)
            
        txome['contig'] = contig_list
        txome['gen_position'] = genpo_list
        txome['strand'] = strands
        with locks['feature']:
            txome.to_csv(output_path, mode='a', index=False, header=None, sep=sep)


def multi_gen_write(input_path, output_path, sep, n_processes, line_count, exon_path, exon_index):
    col_names = pd.read_csv(input_path, nrows=0, sep=sep).columns
    # n_processes = 16
    one_rows = line_count // n_processes + 1
    cols = sep.join(col_names) + f'{sep}contig{sep}gen_position{sep}strand\n'

    with open (output_path, 'w') as f:
        f.write(cols)

    task_queue = multiprocessing.JoinableQueue(maxsize=n_processes*2)
    locks = dict()
    locks['feature'] = multiprocessing.Lock()
    consumers = [Consumer(task_queue=task_queue, task_function=gen_write, locks=locks) for i in range(n_processes)]
    for p in consumers:
        p.start()
    for i in range(0, n_processes):
        sample = pd.read_csv(input_path, skiprows=1+i*one_rows, nrows=one_rows, sep=sep, header=None, names=col_names)
        # print(sample.columns)
        sample['id'] = sample['id'].str.split('|').str[0]
        sample = sample[sample['id'].isin(exon_index.index)]
        task_queue.put((sample, exon_path, exon_index, output_path, sep))
    task_queue = end_queue(task_queue, n_processes)
    task_queue.join()


import argparse

def main():
    parser = argparse.ArgumentParser(description='Convert transcriptomic coordinates to genomic coordinates.')
    parser.add_argument('--gtf_path', required=True, help='Path to directory containing exon_hg38.gtf and exon_hg38.index')
    parser.add_argument('--prefix', type=str, default='data', help='prefix of output file, please keep it THE SAME AS the one used in previous steps. Default: data')
    parser.add_argument('--work_dir', required=True, help='Working directory of your job, please keep it THE SAME AS the one used in previous steps. ')
    parser.add_argument('--n_processes', type=int, default=cpu_count(), help='Number of parallel processes. Default: All available CPU cores')
    args = parser.parse_args()

    inputs = f'{args.work_dir}/{args.prefix}.preds.per.site'
    outputs = f'{args.work_dir}/{args.prefix}.preds.per.site.gen'

    get_gtf_indexed(args.gtf_path)

    exon_path = f'{args.gtf_path}.exon'
    
    exon_index = pd.read_csv(f'{args.gtf_path}.exon.index').set_index('txid')

    with open(inputs, 'r') as f:
        line_count = sum(1 for _ in f)

    print(f'row: {line_count}')
    multi_gen_write(
        input_path=inputs,
        output_path=outputs,
        sep=',',
        n_processes=args.n_processes,
        line_count=line_count,
        exon_path=exon_path,
        exon_index=exon_index
    )

if __name__ == '__main__':
    main()