################################################################################
# SETUP
################################################################################
# Modules
from os.path import join
sys.path.append(os.path.join(os.getcwd(), 'src'))
from constants import *

# Configuration
config['signatures_file'] = config.get('signatures_file',
                                       'data/signatures/cosmic-signatures.tsv') # default: COSMIC
config['active_signatures'] = config.get('active_signatures',
                                         [1,2,3,5,6,8,13,17,18,20,26,30])
config['run_name'] = config.get('run_name', 'ICGC-R22-BRCA') # default:
config['mutations_file'] = config.get('mutations_file',
                                      'data/mutations/ICGC-BRCA-EU.RELEASE_22.SBS.renamed.sigma.json')
config['output_dir'] = OUTPUT_DIR = config.get('output_dir', join('output', config.get('run_name')))

if not ('samples' in config):
    import json
    with open(config.get('mutations_file'), 'r') as IN:
        config['samples'] = json.load(IN).get('samples')
elif type(config['samples']) != type([]):
    config['samples'] = [config['samples']]

config['random_seed'] = config.get('random_seed', 94781)
config['max_iter'] = config.get('max_iter', 100)
config['cloud_thresholds'] = config.get('cloud_thresholds', list(range(1000,10001,1000)))
config['chosen_cloud_threshold'] = config.get('chosen_cloud_threshold', 2000)
config['tolerance'] = config.get('tolerance', 1e-3)

if len(config.get('active_signatures')) == 0:
    ACTIVE_SIGNATURES_PARAM = ''
else:
    ACTIVE_SIGNATURES_PARAM = '-as %s' % ' '.join(map(str, config.get('active_signatures')))

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

TRAINED_MODEL_FMT = '%s/%s{threshold}/{model}-{sample}.json' % (TRAINED_MODEL_DIR, SIGMA_NAME)
MMM_LOOCV_MODEL_FMT = '%s/%s/%s-{sample}.json' % (LOOCV_DIR, MMM_NAME, MMM_NAME)
SIGMA_LOOCV_MODEL_FMT = '%s/%s{threshold}/%s-{sample}.json' % (LOOCV_DIR, SIGMA_NAME, SIGMA_NAME)

# Scripts
TRAIN_AND_PREDICT_PY = join(SRC_DIR, 'train_and_predict.py')

################################################################################
# RULES
################################################################################
# General rules
rule all:
    input:
        expand(TRAINED_MODEL_FMT, sample=config.get('samples'), model=MODEL_NAMES, threshold=[config.get('chosen_cloud_threshold')]),
        expand(SIGMA_LOOCV_MODEL_FMT, sample=config.get('samples'), threshold=config.get('cloud_thresholds')),
        expand(MMM_LOOCV_MODEL_FMT, sample=config.get('samples'))

# Train models for each sample
rule train:
    input:
        mutations=config.get('mutations_file'),
        signatures=config.get('signatures_file')
    params:
        max_iter=config.get('max_iter'),
        random_seed=config.get('random_seed'),
        tolerance=config.get('tolerance'),
        active_signatures=ACTIVE_SIGNATURES_PARAM
    output:
        TRAINED_MODEL_FMT
    shell:
        'python {TRAIN_AND_PREDICT_PY} -mf {input.mutations} -sf {input.signatures} '\
        '-od {TRAINED_MODEL_DIR}/sigma{wildcards.threshold} -mn {wildcards.model} '\
        '{params.active_signatures} -sn {wildcards.sample} -mi {params.max_iter} '\
        '-ct {wildcards.threshold} -rs {params.random_seed} -tol {params.tolerance}'

# Perform LOOCV for each sample
rule sigma_loocv:
    input:
        mutations=config.get('mutations_file'),
        signatures=config.get('signatures_file')
    params:
        max_iter=config.get('max_iter'),
        random_seed=config.get('random_seed'),
        tolerance=config.get('tolerance'),
        active_signatures=ACTIVE_SIGNATURES_PARAM,
    output:
        SIGMA_LOOCV_MODEL_FMT
    shell:
        'python {TRAIN_AND_PREDICT_PY} -mf {input.mutations} -sf {input.signatures} '\
        '-od {LOOCV_DIR}/sigma{wildcards.threshold} -mn {SIGMA_NAME} -sn {wildcards.sample} '\
        '{params.active_signatures} -mi {params.max_iter} -ct {wildcards.threshold} '\
        '-rs {params.random_seed} -tol {params.tolerance} --cross-validation-mode'

rule mmm_loocv:
    input:
        mutations=config.get('mutations_file'),
        signatures=config.get('signatures_file')
    params:
        max_iter=config.get('max_iter'),
        random_seed=config.get('random_seed'),
        tolerance=config.get('tolerance'),
        active_signatures=ACTIVE_SIGNATURES_PARAM,
    output:
        MMM_LOOCV_MODEL_FMT
    shell:
        'python {TRAIN_AND_PREDICT_PY} -mf {input.mutations} -sf {input.signatures} '\
        '-od {LOOCV_DIR}/mmm -mn {MMM_NAME} -sn {wildcards.sample} '\
        '{params.active_signatures} -mi {params.max_iter} -ct 0 '\
        '-rs {params.random_seed} -tol {params.tolerance} --cross-validation-mode'
