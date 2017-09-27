"""Install DLT2T."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='DLT2T',
    version='1.2.2',
    description='Dual Learning for Tensor2Tensor',
    author='Renqian Luo',
    author_email='lrqrichard@gmail.com',
    url='http://github.com/renqianluo/DLT2T',
    license='Apache 2.0',
    packages=find_packages(),
    package_data={
        'DLT2T.data_generators': ['test_data/*'],
        'DLT2T.visualization': [
            'attention.js',
            'TransformerVisualization.ipynb'
        ],
    },
    scripts=[
        'DLT2T/bin/dual-t2t-trainer',
        'DLT2T/bin/dual-t2t-datagen',
        'DLT2T/bin/dual-t2t-decoder',
        'DLT2T/bin/dual-t2t-make-tf-configs',
    ],
    install_requires=[
        'bz2file',
        'future',
        'numpy',
        'requests',
        'sympy',
        'six',
    ],
    extras_require={
        'tensorflow': ['tensorflow>=1.3.0'],
        'tensorflow_gpu': ['tensorflow-gpu>=1.3.0'],
        'tests': ['pytest', 'h5py', 'mock'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='tensorflow machine learning',)
