from setuptools import setup, find_packages
from os import path
from io import open

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='kaggle_twosigma_utils',
    version='0.0.1',
    description='Library with some basic functions for https://www.kaggle.com/c/two-sigma-financial',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/alex4321/kaggle_twosigma_utils',
    author='Alexander Pozharskiy',
    author_email='gaussmake@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        # Pick your license as you wish
        'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='kaggle',  # Optional
    packages=find_packages(),
    install_requires=['pandas', 'scikit-learn'],
    extras_require={  # Optional
        'dev': ['check-manifest'],
        'test': ['coverage'],
    },
    package_data={},
    entry_points={},
)
