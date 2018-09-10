# SigMa

[![Build Status](https://travis-ci.com/lrgr/sigma.svg?branch=master)](https://travis-ci.org/lrgr/sigma)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the source code for SigMa (<u>Sig</u>nature <u>Ma</u>rkov model) and related experiments. SigMa is a probabilistic model of the sequential dependencies among mutation signatures.

Below, we provide an overview of the SigMa model from the [corresponding paper](https://github.com/lrgr/sigma#references). "The input data consists of (**A**) a set of predefined signatures that form
an emission matrix _E_ (here, for simplicity, represented over six mutation types), and (**B**) a sequence of mutation
categories from a single sample and a distance threshold separating sky and cloud mutation segments. (**C**) The SigMa
model has two components: (top) a multinomial mixture model (MMM) for isolated sky mutations and (bottom) an
extension of a Hidden Markov Model (HMM) capturing sequential dependencies between close-by cloud mutations;
all model parameters are learned from the input data in an unsupervised manner. (**D**) SigMa finds the most likely
sequence of signatures that explains the observed mutations in sky and clouds."

<img src='https://github.com/lrgr/sigma/raw/master/src/assets/SigMa-overview.jpg'>

## Setup

### Dependencies
SigMa is written in Python 3. We recommend using [Conda](https://conda.io/docs/) to manage dependencies, which you can do directly using the provided `environment.yml` file:

    conda env create -f environment.yml
    source activate sigma-env

For windows replace last command with

    activate sigma-env

## Usage

We use [Snakemake](https://snakemake.readthedocs.io/en/stable/index.html) to manage the workflow of running SigMa on hundreds of tumor samples.

### Reproducing the experiments from the SigMa paper

First, download and preprocess the ICGC breast cancer whole-genomes and COSMIC mutation signatures. To do so, run:

    cd data && snakemake all

Second, run SigMa and a multinomial mixture model (MMM) on each sample, and perform leave-one-out cross-validation (LOOCV):

    snakemake all

This will create an `output/` directory, with two subdirectories: `models/` and `loocv/`. `models/` contains SigMa trained on each sample. `loocv/` contains the results of LOOCV with SigMa using different cloud thresholds.


### Configuration

To run the entire SigMa workflow on different mutation signatures or data, see the `Snakefile` for configuration options.

To train SigMa or MMM on individual mutation sequences, use the [`src/train_and_predict.py`](https://github.com/lrgr/sigma/blob/master/src/train_and_predict.py) script. To get a list of command-line arguments, run:

    python src/train_and_predict.py -h

## Support

Please report bugs and feature requests in the [Issues](https://github.com/lrgr/sigma/issues) tab of this GitHub repository.

For further questions, please email [Max Leiserson](mailto:mdml@cs.umd.edu) and [Itay Sason](itaysason@mail.tau.ac.il
) directly.

## References

Xiaoqing Huang*, Itay Sason*, Damian Wojtowicz*, Yoo-Ah Kim, Mark Leiserson^, Teresa M Przytycka^, Roded Sharan^. Hidden Markov Models Lead to Higher Resolution Maps of Mutation Signature Activity in Cancer. _bioRxiv_ (2018) [doi:10.1101/392639](https://doi.org/10.1101/392639).

\* equal first author contribution
^ equal senior author contribution
