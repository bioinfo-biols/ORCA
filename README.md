# Orca: Omni RNA Modification Characterization and Annotation

Orca is a command-line toolkit for RNA modification analysis, featuring preprocessing (pileup & eventalign), feature extraction, prediction using pretrained models, and genomic annotation.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
  - [Step1. Basecalling & alignments](#step1-basecalling--alignments)
  - [Step2. Prediction](#step2-prediction)
  - [Step3. Annotation](#step3-annotation)
- [Outputs](#outputs)
  - [1. Prediction](#1-prediction)
  - [2. Annotation](#2-annotation)
- [Support](#support)
- [Author](#author)
- [License](#license)



## Installation

### Dependency

```
Softwares:
    minimap2 >= 2.21
    samtools >= 1.11
    f5c >= 1.11
    slow5tools >= 0.8.0
```

Recommended python version is python v3.10.

### Installation with wheel

One can create a virtual envirionment with conda and install Orca using wheel file provided in dist folder.

```bash
conda create -n Orca python=3.10
conda activate Orca
git clone https://bioinfo.ioz.ac.cn/git/dongh/Orca.git
cd Orca
pip install ./dist/ORCA-0.1.0-py3-none-any.whl
```

### Information of test files

To test Orca, please start from [Step2](#step2-prediction). We provide test files as shown in the chart below: 

| column | name                                      | for command                  | description                                                        |
| ------ | ----------------------------------------- | -----------------------------| ------------------------------------------------------------------ |
| 1      | MCF7.Example.eventalign                   | orca-pred_signal_feature_ext | Eventalign file of a human sample from f5c eventalign command      |
| 2      | MCF7.Example.pileup                       | orca-pred_bascal_feature_ext | Pileup file of a human sample from samtools mpileup command        |
| 3      | Answer_from_RMBase_and_DirectRMDB_NGS.csv | orca-annotation              | A csv file containing the genomic coordicates of RNA modifications |

Please click [Here](#https://bioinfo.ioz.ac.cn/files/share/7O9b7pxx) for download.

## Usage

### Step1. Basecalling & alignments

Perform basecalling on FAST5 files using GUPPY:

```bash
guppy_basecaller -i /path/to/FAST5 -s /path/to/output --config /path/to/configuration --fast5-out
```

Align FASTQ sequences to the transcriptome with minimap2 and samtools:

```bash
minimap2 -ax splice -N 0 -uf -k14 --cs -t threads <transcriptome> <fastq> | samtools sort -@ threads -o <bam>
samtools index -@ threads <bam>
```

Convert FAST5 files using slow5tools:

```bash
slow5tools f2s <fast5_dir> --allow -d <blow5_dir> -p threads
slow5tools merge <blow5_dir> -o merged_blow5 -t threads
```

Use f5c/nanopolish to align signals to reference sequences (example using f5c):

```bash
f5c index -t threads --slow5 blow5 <fastq>
f5c eventalign --rna --signal-index --scale-events --threads threads --slow5 blow5 --reads fastq --bam bam --secondary=no --collapse-events --genome transcriptome --summary summary_path > eventalign_file
```

### Step2. Prediction

While running Orca, make sure to run all commands in Step 2 and Step 3 with **the same `--work_dir` folder and the same `--prefix` string**.

1. **Signal Feature Extraction**  
    ```bash
    # Extract signal alignment features from the eventalign results:
    usage: orca-pred_signal_feature_ext [-h] [--n_processes N_PROCESSES] --eventalign EVENTALIGN [--chunk_size CHUNK_SIZE] [--prefix PREFIX] --work_dir WORK_DIR

    Extract signal alignment features from the eventalign results.

    options:
      -h, --help            show this help message and exit
      --n_processes N_PROCESSES
                            Number of parallel processes. Default: All available CPU cores
      --eventalign EVENTALIGN
                            Path to the eventalign file.
      --chunk_size CHUNK_SIZE
                            Chunk size for reading eventalign files for indexing. Default: 100000
      --prefix PREFIX       prefix of output file. Default: data
      --work_dir WORK_DIR   Working directory of your job.
    ```



2. **Basecalling Feature Extraction**  
    ```bash
    # Extract basecalling features from pileup results
    usage: orca-pred_bascal_feature_ext [-h] --pileup PILEUP [--prefix PREFIX] --work_dir WORK_DIR [--n_processes N_PROCESSES]

    options:
      -h, --help            show this help message and exit
      --pileup PILEUP       pileup FILE PATH from samtools mpileup
      --prefix PREFIX       prefix of output file, please keep it THE SAME AS the one used in previous steps. Default: data
      --work_dir WORK_DIR   Working directory of your job, please keep it THE SAME AS the one used in previous steps.
      --n_processes N_PROCESSES
                            Number of parallel processes. Default: All available CPU cores
    ```



3. **Feature Merge**  
    ```bash
    # Merge both types of features:
    usage: orca-pred_feature_merge [-h] [--prefix PREFIX] --work_dir WORK_DIR [--n_processes N_PROCESSES]

    options:
      -h, --help            show this help message and exit
      --prefix PREFIX       prefix of output file, please keep it THE SAME AS the one used in previous steps. Default: data
      --work_dir WORK_DIR   Working directory of your job, please keep it THE SAME AS the one used in previous steps.
      --n_processes N_PROCESSES
                            Number of parallel processes. Default: All available CPU cores
    ```

4. **Run Prediction**  
    ```bash
    # RNA modification sites prediction based on pretrained models
    usage: orca-prediction [-h] [--prefix PREFIX] --work_dir WORK_DIR [--extractor_path EXTRACTOR_PATH] [--classifier_path CLASSIFIER_PATH]

    options:
      -h, --help            show this help message and exit
      --prefix PREFIX       prefix of output file, default: data
      --work_dir WORK_DIR
                            DIRECTORY of output files
      --extractor_path EXTRACTOR_PATH
                            Path to the feature extractor model, default: orca/models/feature_extractor.pt
      --classifier_path CLASSIFIER_PATH
                            Path to the class classifier model, default: orca/models/class_classifier.pt
    ```

### Step3. Annotation



1. **Genomic Location**
    ```bash
    # Map transcriptomic to genomic coordinates:
    usage: orca-prediction [-h] [--prefix PREFIX] --work_dir WORK_DIR [--extractor_path EXTRACTOR_PATH] [--classifier_path CLASSIFIER_PATH]

    Run prediction on sample feature data using prediction models.

    options:
      -h, --help            show this help message and exit
      --prefix PREFIX       prefix of output file, please keep it THE SAME AS the one used in previous steps. Default: data
      --work_dir WORK_DIR   Working directory of your job, please keep it THE SAME AS the one used in previous steps.
      --extractor_path EXTRACTOR_PATH
                            Path to the feature extractor model, default: orca/models/feature_extractor.pt
      --classifier_path CLASSIFIER_PATH
                            Path to the class classifier model, default: orca/models/class_classifier.pt
    ```



2. **Filter for Annotation**
    ```bash
    usage: orca-anno_bascal_feature_ext [-h] [--prefix PREFIX] --work_dir WORK_DIR

    options:
      -h, --help           show this help message and exit
      --prefix PREFIX      prefix of output file, please keep it THE SAME AS the one used in previous steps. Default: data
      --work_dir WORK_DIR  Working directory of your job, please keep it THE SAME AS the one used in previous steps.
    ```


3. **Flatten Signal Features**
    ```bash
    usage: orca-anno_signal_feature_ext [-h] [--prefix PREFIX] --work_dir WORK_DIR

    options:
      -h, --help           show this help message and exit
      --prefix PREFIX      prefix of output file, please keep it THE SAME AS the one used in previous steps. Default: data
      --work_dir WORK_DIR  Working directory of your job, please keep it THE SAME AS the one used in previous steps.
    ```


4. **Annotation**:  

    The answer file is a 0-based modification annotation CSV file without a header, containing four columns: chromosome, position, strand, and modification type. See the test folder for an example.

    ```bash
    usage: orca-annotation [-h] --answer_path ANSWER_PATH --ref_path REF_PATH [--mod_num_threshold MOD_NUM_THRESHOLD] [--prefix PREFIX] --work_dir WORK_DIR

    options:
      -h, --help            show this help message and exit
      --answer_path ANSWER_PATH
                            Path to the NGS-based answers
      --ref_path REF_PATH   Path to the reference GENOME path
      --mod_num_threshold MOD_NUM_THRESHOLD
                            Minimum number of modifications required to retain
      --prefix PREFIX       prefix of output file, please keep it THE SAME AS the one used in previous steps. Default: data
      --work_dir WORK_DIR   Working directory of your job, please keep it THE SAME AS the one used in previous steps.
    ```

## Outputs

### 1. Prediction
  The prediction results are stored in the `your_prefix.preds.per.site` file, containing the following columns:

| column | name       | description                              |
| ------ | ---------- | ---------------------------------------- |
| 1      | id         | transcript ID                            |
| 2      | position   | 0-based transcriptome coordinate         |
| 3      | kmer       | 5-mers sequence centered at this position|
| 4      | depth      | sequencing depth                         |
| 5      | modScore   | RNA modification score                   |
| 6      | pred\_rate | predicted modification proportion        |

### 2. Annotation
  The annotation results are stored in the `your_prefix.annotation.per.site` file, containing the following columns:

| column | name          | description                              |
| ------ | ------------- | ---------------------------------------- |
| 1      | id            | transcript ID                            |
| 2      | position      | 0-based transcriptome coordinate         |
| 3      | kmer          |11-mer sequence centered at this position |
| 4      | contig        | chromosome name                          |
| 5      | gen\_position | 0-based genomic coordinate               |
| 6      | strand        | strand of the transcript                 |
| 7      | modification  | predicted modification                   |
| 8      | source        | source of the modification               |

      

## Support

If you encounter issues or have questions, please open an issue on our [GitHub repository]().

## Author

Authors: Han Dong(donghan@biols.ac.cn), Jinyang Zhang(zhangjinyang@biols.ac.cn), Fangqing Zhao(zhfq@biols.ac.cn)

Maintainer: Han Dong

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

*Last updated: June 10, 2025*