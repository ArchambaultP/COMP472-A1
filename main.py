#!/usr/bin/env python

import sklearn as sk
import csv
from pathlib import Path

def main():
    path = Path('./dataset')

    symbol_dict = {}
    samples = {}

    with open(path / 'info_1.csv') as file:
        reader = csv.reader(file, delimiter='\n') 
        for row in reader:
            try:
                row = row[0].split(',')
                symbol_dict[int(row[0])] = row[1]
                samples[int(row[0])] = 0
            except Exception as e:
                continue
    
    print(symbol_dict)

    with open(path / 'train_1.csv') as file:
        reader = csv.reader(file, delimiter='\n') 
        for row in reader:
            index = int(row[0].split(',')[-1])
            samples[index] += 1

    print(samples)
    
 

if __name__ == '__main__':
    main()
