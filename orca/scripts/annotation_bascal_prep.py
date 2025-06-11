import pandas as pd
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix', type=str, default='data', help='prefix of output file, please keep it THE SAME AS the one used in previous steps. Default: data')
    parser.add_argument('--work_dir', required=True, help='Working directory of your job, please keep it THE SAME AS the one used in previous steps. ')

    args = parser.parse_args()

    feature_list = ['id', 'position', 'kmer', 'depth'] + [
        f'{x}_{y}' for x in ['-2', '-1', '0', '+1', '+2']
        for y in ['Mismatch_Ratio', 'Insertion_Ratio', 'Deletion_Ratio', 'Qual_Mean', 'Qual_Median', 'Qual_Stdv']
    ]

    target = f'{args.work_dir}/{args.prefix}.preds.per.site.gen'
    feature = f'{args.work_dir}/{args.prefix}.merged.feature.per.site'
    output = f'{args.work_dir}/{args.prefix}.annot.bascal.feature.per.site'

    print('Reading significant sites...')
    t = pd.read_csv(target)
    t = t[t['modScore'] >= .9].set_index(['id', 'position'])

    print('Writing header...')
    with open(output, 'w') as f:
        f.write(','.join(feature_list + ['contig', 'gen_position', 'strand']) + '\n')

    print('Filtering features...')
    reader = pd.read_csv(feature, usecols=feature_list, chunksize=100000)
    for one in tqdm(reader):
        one['id'] = one['id'].str.split('|').str[0]
        one = one.set_index(['id', 'position'])
        one = one[one.index.isin(t.index)].reset_index()
        t_reset = t.reset_index()[['id', 'position', 'contig', 'gen_position', 'strand']]
        one = one.merge(t_reset, on=['id', 'position'], how='left')
        # one[['contig', 'gen_position', 'strand']] = one.apply(lambda row: get_gen_inform(row['id'], row['position'], t), axis=1)
        one.to_csv(output, index=False, mode='a', header=False)

if __name__ == '__main__':
    main()
