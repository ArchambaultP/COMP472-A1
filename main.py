#!/usr/bin/env python

import csv
from pathlib import Path
import matplotlib.pyplot as plt
import baseDT
import bestDT
import gnb
import bestMLP
import baseMLP
import perceptron
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

    dataset_index = 1;

    with open(path / f'info_{dataset_index}.csv') as file:
        reader = csv.reader(file, delimiter='\n') 
        for row in reader:
            try:
                row = row[0].split(',')
                symbol_dict[int(row[0])] = row[1]
                samples[int(row[0])] = 0
            except Exception as e:
                continue
    
    # print(symbol_dict)

    with open(path / f'train_ {dataset_index}.csv') as file:
        reader = csv.reader(file, delimiter='\n') 
        for row in reader:
            split_row = row[0].split(',')
            index = int(split_row[-1])
            samples[index] += 1
            X_train.append([int(string) for string in split_row[:-1]]) # training data, label in last position
            Y_train.append(index)

    with open(path / f'val_{dataset_index}.csv') as file:
        reader = csv.reader(file, delimiter='\n') 
        for row in reader:
            split_row = row[0].split(',')
            index = int(split_row[-1])
            samples[index] += 1
            X_val.append([int(string) for string in split_row[:-1]]) # training data, label in last position
            Y_val.append(index)
    
    with open(path / f'test_with_label_{dataset_index}.csv') as file:
        reader = csv.reader(file, delimiter='\n') 
        for row in reader:
            split_row = row[0].split(',')
            index = int(split_row[-1])
            samples[index] += 1
            X_test.append([int(string) for string in split_row[:-1]]) # training data, label in last position
            Y_test.append(index)

    symbs = symbol_dict.values()
    vals = samples.values()
    fig, ax = plt.subplots()
    ax.bar(symbs, vals)
    plt.show()

    # baseD = baseDT.baseDT(X_train, Y_train, X_test, Y_test)
    # outFile.createCSV(baseD, Y_test, symbol_dict, "BASE-DT-DS1")

    # dt = bestDT.bestDT(X_train, Y_train, X_test, Y_test, list(symbol_dict.keys()))
    # outFile.createCSV(dt, Y_test, symbol_dict, "BEST-DT-DS1")

    # Y_gnb = gnb.gaussianNB(X_train, Y_train, X_test)
    # outFile.createCSV(Y_gnb, Y_test, symbol_dict, "GNB-DS1")

    # pcp = perceptron.perceptron(X_train,Y_train,X_test)
    # outFile.createCSV(pcp, Y_test, symbol_dict, "PER-DS1")

    # base_mlp = baseMLP.baseMLP(X_train,Y_train,X_test)
    # outFile.createCSV(base_mlp,Y_test,symbol_dict,"BASE-MLP-DS1")

    # best_mlp = bestMLP.bestMLP(X_train, Y_train, X_val, Y_val, X_test)
    # outFile.createCSV(best_mlp, Y_test, symbol_dict, "BEST_MLP-DS1")

    #bestDT_pred = bestDT.bestDT(X_train, Y_train, X_test, Y_test, list(symbol_dict.keys()))
    

if __name__ == '__main__':
    main()
