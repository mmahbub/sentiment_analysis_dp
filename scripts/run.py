import os

os.system('./data_poison.py')

os.system('./train.py --gpus=4')