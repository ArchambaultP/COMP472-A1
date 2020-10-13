#!/usr/bin/env python

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import baseDT
import bestDT
import gnb
import bestMLP
import outFile

def main():
    path = Path('./dataset')

    symbol_dict = {}
    samples = {}
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
    X_test = []
    Y_test = []


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

    with open(path / 'val_1.csv') as file:
        reader = csv.reader(file, delimiter='\n') 
        for row in reader:
            split_row = row[0].split(',')
            index = int(split_row[-1])
            samples[index] += 1
            X_val.append([int(string) for string in split_row[:-1]]) # training data, label in last position
            Y_val.append(index)
    
    with open(path / 'test_with_label_1.csv') as file:
        reader = csv.reader(file, delimiter='\n') 
        for row in reader:
            split_row = row[0].split(',')
            index = int(split_row[-1])
            samples[index] += 1
            X_test.append([int(string) for string in split_row[:-1]]) # training data, label in last position
            Y_test.append(index)

    symbs = symbol_dict.values()
    vals = samples.values()
    # fig, ax = plt.subplots()
    # ax.bar(symbs, vals)
    # plt.show()

    # baseDT.train(X_train, Y_train)
    # Y_gnb = gnb.gaussianNB(X_train, Y_train, X_test)
    
    # bestMLP.bestMLP(X_train, Y_train, X_val, Y_val, X_test, Y_test)
    # outFile.createCSV(Y_gnb, Y_test, symbol_dict, "wtv")

    bestDT_pred = bestDT.bestDT(X_train, Y_train, X_val, Y_val, list(symbol_dict.keys()))


if __name__ == '__main__':
    main()
