#-*- coding:utf- -*-
import numpy as np
from sys import argv
import os

def TestLabelGenerate(sequence_path, scalar_path, label_path):
    sequence = np.loadtxt(sequence_path, delimiter = ',', dtype = int)
    scalar = np.loadtxt(scalar_path, delimiter = ',', dtype = int)
    label = np.array([np.argmax(np.convolve(seq, np.ones(sca), mode = 'valid')) 
                      for seq, sca in zip(sequence, scalar)])
    if not os.path.exists(label_path):
        np.savetxt(label_path, label.reshape(-1, 1), fmt = '%d', delimiter = ',')

if __name__ == '__main__':
    if len(argv) >= 4:
        TestLabelGenerate(argv[1], argv[2], argv[3])