################################################################################
# SETUP
################################################################################
# Modules
from os.path import join
sys.path.append(os.path.join(os.getcwd(), 'src'))
from constants import *

# Configuration
config['signatures_file'] = config.get('signatures_file',
                                       'data/signatures/emissions_for_breast_cancer.npy') # default:
config['active_signatures'] = config.get('active_signatures',
                                         [1,2,3,5,6,8,13,17,18,20,26,30])
config['run_name'] = config.get('run_name', 'ICGC-R22-BRCA') # default:
config['mutations_file'] = config.get('mutations_file',
                                      'data/mutations/nik-zainal2016-wgs-brca-mutations-for-hmm.json')
config['models'] = config.get('models', MODEL_NAMES)
config['output_dir'] = OUTPUT_DIR = config.get('output_dir', join('output', config.get('run_name')))

if not ('samples' in config):
    import json
    with open(config.get('mutations_file'), 'r') as IN:
        config['samples'] = json.load(IN).get('samples')

    config['samples'] = ['PD4076']

config['models'] = [MMM_NAME]

config['random_seed'] = config.get('random_seed', 94781)
config['max_iter'] = config.get('max_iter', 100)
config['cloud_thresholds'] = config.get('cloud_thresholds', list(range(1000,10001,1000)))
config['chosen_cloud_threshold'] = config.get('chosen_cloud_threshold', 2000)
config['tolerance'] = config.get('tolerance', 1e-3)

# Directories
DATA_DIR = 'data'
SIGNATURES_DIR = join(DATA_DIR, 'signatures')
MUTATIONS_DIR = join(DATA_DIR, 'mutations')

LOOCV_DIR = join(OUTPUT_DIR, 'loocv')
TRAINED_MODEL_DIR = join(OUTPUT_DIR, 'models')

SRC_DIR = 'src'

# Files
SIGNATURES_FILE = config.get('signatures_file')
MUTATIONS_FILE = config.get('mutations_file')

TRAINED_MODEL_FMT = '%s/ct{threshold}/{model}-{sample}.json' % TRAINED_MODEL_DIR
LOOCV_MODEL_FMT = '%s/ct{threshold}/{model}-{sample}.json' % LOOCV_DIR

# Scripts
TRAIN_AND_PREDICT_PY = join(SRC_DIR, 'train_and_predict.py')

################################################################################
# RULES
################################################################################
# General rules
rule all:
    input:
        expand(TRAINED_MODEL_FMT, sample=config.get('samples'), model=config.get('models'), threshold=[config.get('chosen_cloud_threshold')]),
        expand(LOOCV_MODEL_FMT, sample=config.get('samples'), model=config.get('models'), threshold=config.get('cloud_thresholds'))

# Train models for each sample
rule train:
    input:
        mutations=config.get('mutations_file'),
        signatures=config.get('signatures_file')
    params:
        max_iter=config.get('max_iter'),
        random_seed=config.get('random_seed'),
        tolerance=config.get('tolerance')
    output:
        TRAINED_MODEL_FMT
    shell:
        'python {TRAIN_AND_PREDICT_PY} -mf {input.mutations} -sf {input.signatures} '\
        '-od {TRAINED_MODEL_DIR}/ct{wildcards.threshold} -mn {wildcards.model} '\
        '-sn {wildcards.sample} -mi {params.max_iter} -ct {wildcards.threshold} '\
        '-rs {params.random_seed} -tol {params.tolerance}'

# Perform LOOCV for each sample
rule loocv:
    input:
        mutations=config.get('mutations_file'),
        signatures=config.get('signatures_file')
    params:
        max_iter=config.get('max_iter'),
        random_seed=config.get('random_seed'),
        tolerance=config.get('tolerance')
    output:
        LOOCV_MODEL_FMT
    shell:
        'python {TRAIN_AND_PREDICT_PY} -mf {input.mutations} -sf {input.signatures} '\
        '-od {LOOCV_DIR}/ct{wildcards.threshold} -mn {wildcards.model} -sn {wildcards.sample} '\
        '-mi {params.max_iter} -ct {wildcards.threshold} -rs {params.random_seed} '\
        '-tol {params.tolerance} --cross-validation-mode'
