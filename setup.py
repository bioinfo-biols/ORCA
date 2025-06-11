from setuptools import setup, find_packages

setup(
    name='ORCA',
    version='0.1.0',
    author='Han DONG',
    author_email='donghan@biols.ac.cn',
    description='ORCA: Omni RNA modification characterization and annotation',
    packages=find_packages(include=['orca', 'orca.*']),
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'torchvision',
        'scikit-learn',
        'tqdm',
        'progressbar2',
        'matplotlib',
        'pysam==0.23.2',
    ],
    entry_points={
        'console_scripts': [
            'orca-pred_signal_feature_ext = orca.scripts.prediction_signal_prep:main',
            'orca-pred_feature_merge = orca.scripts.prediction_feature_merge:main',
            'orca-prediction = orca.scripts.prediction:main',
            'orca-pred_bascal_feature_ext = orca.scripts.prediction_bascal_prep:main',
            'orca-annotation = orca.scripts.annotation:main', 
            'orca-anno_signal_feature_ext = orca.scripts.annotation_signal_prep:main', 
            'orca-anno_bascal_feature_ext = orca.scripts.annotation_bascal_prep:main', 
            'orca-genomic_locator = orca.scripts.gen_write:main'
        ],
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.10',
)
