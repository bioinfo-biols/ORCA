import argparse
import os
import pandas as pd
from tqdm import tqdm
from .Predict_22 import *

def predict(index, feature_path, New_Features, feature_extractor, class_classifier, output_path):
    txids, feature_tensor, kmers, depths, mod_rate = feature_read(index, feature_path, New_Features)
    prediction(feature_tensor, feature_extractor, class_classifier, output_path, txids, kmers, depths, mod_rate)

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

def main():
    print('Performing prediction...\n')
    _MODEL_DIR = files('orca').joinpath('models')
    _DEFAULT_EXTRACTOR = str(_MODEL_DIR.joinpath('feature_extractor.pt'))
    _DEFAULT_CLASSIFIER = str(_MODEL_DIR.joinpath('class_classifier.pt'))


    parser = argparse.ArgumentParser(description="Run prediction on sample feature data using prediction models.")
    parser.add_argument('--prefix', type=str, default='data', help='prefix of output file, please keep it THE SAME AS the one used in previous steps. Default: data')
    parser.add_argument('--work_dir', required=True, help='Working directory of your job, please keep it THE SAME AS the one used in previous steps. ')
    parser.add_argument('--extractor_path', default=_DEFAULT_EXTRACTOR, help='Path to the feature extractor model, default: orca/models/feature_extractor.pt')
    parser.add_argument('--classifier_path', default=_DEFAULT_CLASSIFIER, help='Path to the class classifier model, default: orca/models/class_classifier.pt')
    args = parser.parse_args()
    
    feature_index = f'{args.work_dir}/{args.prefix}.merged.feature.index'
    feature_path = f'{args.work_dir}/{args.prefix}.merged.feature.per.site'
    output = f'{args.work_dir}/{args.prefix}.preds.per.site'

    onepos = ['Match_Ratio','Mismatch_Ratio','Insertion_Ratio','Deletion_Ratio','Qual_Mean','Qual_Stdv'] + [f'{x}_shape' for x in range(0, 50)]
    New_Features = [f'{x}_{y}' for x in ['-2', '-1', '0', '+1', '+2'] for y in onepos]

    # os.makedirs(os.path.dirname(args.work_dir), exist_ok=True)

    with open(output, 'w') as output_file:
        output_file.write("id,position,kmer,depth,modScore,pred_rate\n")


    ELIGOS_feature_index(feature_path, feature_index)

    index_csv = pd.read_csv(feature_index)

    feature_extractor, class_classifier = model_load(args.extractor_path, args.classifier_path)
    feature_extractor.eval()
    class_classifier.eval()

    for line, toki in zip(index_csv.iterrows(), tqdm(range(0,len(index_csv)))):
        _, row = line
        predict(row, feature_path, New_Features, feature_extractor, class_classifier, output)

    print("Prediction completed successfully!")

if __name__ == '__main__':
    main()