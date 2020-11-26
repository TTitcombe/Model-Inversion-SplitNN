from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name="dpsnn",
    version="0.1.0dev1",
    description="Researching differentially-private SplitNNs",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/TTitcombe/Differentially-Private-SplitNN',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    python_requires='>=3.6',
    install_requires=[
    'torch', 'torchvision', 'numpy', 'matplotlib', 'tensorboard', 'pytorch_lightning'],
)
