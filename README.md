# SigMa

## Setup

Clone the project, create virtual enviornment, and activate it
```
git clone https://github.com/lrgr/sigma.git
cd sigma
conda env create -f environment.yml
source activate sigma-env
```
For windows replace last command with
```
activate sigma-env
```

## Run
### Data
To download the data execute the snakemake in the data directory
```
cd data
snakemake all
cd ..
```
### Activating models on data
```
snakemake all
```
This will create results directory, with 2 directories inside, one for signatures and one for the leave one out test.
