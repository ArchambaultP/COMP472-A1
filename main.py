#!/usr/bin/env python

import csv
from pathlib import Path
import matplotlib.pyplot as plt

def main():
    path = Path('./dataset')

    symbol_dict = {}
    samples = {}
    training_data = []

    with open(path / 'info_1.csv') as file:
        reader = csv.reader(file, delimiter='\n') 
        for row in reader:
            try:
                row = row[0].split(',')
                symbol_dict[int(row[0])] = row[1]
                samples[int(row[0])] = 0
            except Exception as e:
                continue
    
    # print(symbol_dict)

    with open(path / 'train_1.csv') as file:
        reader = csv.reader(file, delimiter='\n') 
        for row in reader:
            split_row = row[0].split(',')
            index = int(split_row[-1])
            samples[index] += 1
            training_data.append([int(string) for string in split_row[:-1]])


    # print(samples)

    symbs = symbol_dict.values()
    vals = samples.values()
    fig, ax = plt.subplots()
    ax.bar(symbs, vals)

    plt.show()


if __name__ == '__main__':
    main()
