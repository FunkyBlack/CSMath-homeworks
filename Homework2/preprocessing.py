# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 13:33:40 2017

@author: FunkyBlack
"""
import numpy as np

def load_data(fname, digit=3):
    lines = open(fname).readlines()
    digit_datas = []
    digit_data = ''
    for i, line in enumerate(lines):
        if len(line)==33:
            digit_data += line.strip()
        if len(line)==3:
            if line[1]==str(digit):
                print (len(digit_data))
                digit_datas.append([int(x) for x in list(digit_data)])
            digit_data = ''
    digit_datas = np.array(digit_datas)
    return digit_datas

if __name__ == '__main__':
    fnames = ['dataset/optdigits-orig.cv',
              'dataset/optdigits-orig.tra',
              'dataset/optdigits-orig.wdep',
              'dataset/optdigits-orig.windep'
              ]
    digit = 3
    all_digit_data = []          
    for fname in fnames:
        data = load_data(fname, digit=digit)
        all_digit_data.append(data)
    all_digit_data = np.vstack(all_digit_data)
    print (all_digit_data.shape)
    np.save('all_digit_{}.npy'.format(digit), all_digit_data)