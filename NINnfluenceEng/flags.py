'''
Created on Jan 30, 2017

@author: atri
'''

GRID_SIZE_ROWS = 5
GRID_SIZE_COLS = 6 
NUM_INPUT_CHANNELS = 7
NUM_ENV_CHANNELS = 20
NUM_OUTPUT_CHANNELS = 2
BATCH_SIZE = 50
DECAY_STEP = 40000
NUM_EPOCHS = (200 // BATCH_SIZE) * 100
SEED = 66478
EVAL_BATCH_SIZE = 1
EVAL_FREQUENCY = 10
