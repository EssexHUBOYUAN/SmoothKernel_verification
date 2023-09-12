# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 12:30:48 2023

@author: Doc Who
"""
#get_POPvsCORR_metric
import csv
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

'''______________read file_____________________'''
'''
read path/gender pop^corr.csv
get dictionary D_pix_arr[gender] = { 'TP':arr_TP, 'FN':arr_FN, 'FP':arr_FP }
w/ for each arr, pix as [pop,corr]
'''
def read(filename):
    D_pix_arr = dict()
    D_med = dict()
    L_pop = []
    L_corr = []
    with open(filename, mode='r', encoding='UTF-8-sig') as file:
    	data = csv.reader(file)
    	head = True
    	for line in data:
    	    if head:
                head = False
    	    else:
                pop = round(float(line[0]),1)
                corr = round(float(line[1]),1)
                sex = line[2]
                L_result = []
                if pop not in L_pop:
                    L_pop.append(pop)
                if corr not in L_corr:
                    L_corr.append(corr)
                try:
                    L_result = L_result + line[3].split('/')
                except TypeError:
                    L_result = L_result + [line[3]]
                if sex == 'female':
                    if line[5] == '':
                        L_result.append('TN')
                else:
                    if line[6] == '':
                        L_result.append('TN')
                D_med[pop, corr] = L_result
    L_pop.sort()
    L_corr.sort()
    arr_FN = np.zeros( (len(L_pop), len(L_corr)) )
    arr_TP = np.zeros( (len(L_pop), len(L_corr)) )
    arr_FP = np.zeros( (len(L_pop), len(L_corr)) )
    arr_TN = np.zeros( (len(L_pop), len(L_corr)) )
    filename_tail = filename.split('/')[-1]
    gender = filename_tail.split(' ')[0]
    for pop, corr in D_med:
    	i_pop = L_pop.index(pop)
    	i_corr = L_corr.index(corr)
    	if 'ND' in D_med[pop, corr]:
    	    arr_FN[i_pop, i_corr] = 1
    	if 'TP' in D_med[pop, corr]:
    	    arr_TP[i_pop, i_corr] = 1
    	'''
    	if 'neg' in D_med[pop, corr]:
    	    arr_FP[i_pop, i_corr] = 1 + arr_FP[i_pop, i_corr]
    	'''
    	if 'FP' in D_med[pop, corr]:
    	    arr_FP[i_pop, i_corr] = 1 + arr_FP[i_pop, i_corr]
    	if 'TN' in D_med[pop, corr]:
    	    arr_TN[i_pop, i_corr] = 1 + arr_TN[i_pop, i_corr]
    
    D_pix_arr[gender] = { 'TP':arr_TP, 'FN':arr_FN, 'FP':arr_FP, 'TN':arr_TN }
    return D_pix_arr

'''___________________combine info_____________________'''
def add_arr(D_pix_arr, D_main_arr):
    for gender in D_pix_arr:
    	try:
    	    D_main_arr[gender]['TP'] = D_main_arr[gender]['TP'] + D_pix_arr[gender]['TP']
    	    D_main_arr[gender]['FN'] = D_main_arr[gender]['FN'] + D_pix_arr[gender]['FN']
    	    D_main_arr[gender]['FP'] = D_main_arr[gender]['FP'] + D_pix_arr[gender]['FP']
    	    D_main_arr[gender]['TN'] = D_main_arr[gender]['TN'] + D_pix_arr[gender]['TN']
    	except KeyError:
    	    D_main_arr[gender] = D_pix_arr[gender]
    return D_main_arr

'''___________________main_____________________'''
def plot_figs(D_main_arr, L_title = ['sensitivity', 'specificity', 'precision', 'negative predictive value']):
    for gender in D_main_arr:
    	i = 0
    	for title in L_title:
            i = 1+i
            arr = np.transpose( D_main_arr[gender][title] )
            sns.heatmap(arr)
            plt.title('YEAR:' + gender + ' product test:' + title)
            plt.xticks(np.arange(8)+0.5, labels = [ '5.7','6.0','6.7','7.0','7.7','8.0','8.7','9.0' ] )
            plt.xlabel(xlabel='log population')
            plt.yticks(np.arange(9)+0.5, labels = [ '0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9' ] )
            plt.ylabel(ylabel='correlation cofficient')
            plt.show()
    return

def savefile(D_main_arr):
    return

def main(path, L_gensis_label, L_setind, filename_tail = ' pop^cor test.csv',str_med = '/set_3/testset'):
    D_main_arr = dict()
    N = 0
    for label in L_gensis_label:
    	for ind in L_setind:
    	    pathtopopcorrfile = path+'test_gensis_'+label+str_med+ind+'/'
    	    for gender in ['female','male']:
        		filename = pathtopopcorrfile + gender + filename_tail
        		D_arr = read(filename)
        		D_main_arr = add_arr(D_arr, D_main_arr)
        		N = N + 1
    for gender in D_main_arr:
    	D_main_arr[gender]['sensitivity'] = D_main_arr[gender]['TP']/(D_main_arr[gender]['TP'] + D_main_arr[gender]['FN'])
    	D_main_arr[gender]['specificity'] = D_main_arr[gender]['TN']/(D_main_arr[gender]['TN'] + D_main_arr[gender]['FP'])
    	D_main_arr[gender]['precision'] = D_main_arr[gender]['TP']/(D_main_arr[gender]['TP'] + D_main_arr[gender]['FP'])
    	D_main_arr[gender]['negative predictive value'] = D_main_arr[gender]['TN']/(D_main_arr[gender]['TN'] + D_main_arr[gender]['FN'])
    plot_figs(D_main_arr, ['TP'])
    return

path = 'G:/smoothkernel method verification/YEAR/'
L_gensis_label=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
L_setind = ['1','2','3']
main(path, L_gensis_label, L_setind)