#!/usr/bin/env python

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import baseDT

def main():
    path = Path('./dataset')

    symbol_dict = {}
    samples = {}
    X_train = []
    Y_train = []

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
            X_train.append([int(string) for string in split_row[:-1]]) # training data, label in last position
            Y_train.append(index)

    # print(samples)

    symbs = symbol_dict.values()
    vals = samples.values()
    fig, ax = plt.subplots()
    ax.bar(symbs, vals)
    # plt.show()

    baseDT.train(X_train, Y_train)


    


if __name__ == '__main__':
    main()
