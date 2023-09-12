# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:10:30 2023
this program reads the test sets (corr crossproduct pop) and form summary table for male amd female

@author: Admin
"""

import os
import shutil
import csv
import numpy as np
from scipy import stats

female_filename = 'Impect on female by ingredients in different countries.csv'
male_filename = 'Impect on male by ingredients in different countries.csv'
D_filename = dict()
D_filename['male'] = male_filename
D_filename['female'] = female_filename
def read_summary_file(filename):
    D_product = dict()
    with open(filename, mode='r', encoding = 'UTF-8-sig') as file:
        data = csv.reader(file)
        Head = True
        for line in data:
            if Head:
                Head = False
            else:
                #D_product[product name] : [list_slope, list_std, list_p_value, list_t_value, list_numbercountry]
                try:
                    D_product[line[0]].append([line[i+2] for i in range(len(line)-2)])
                except KeyError:
                    D_product[line[0]] = [[line[i+2] for i in range(len(line)-2)]]
    return D_product

def test_n(list_n):
    try:
        n = float(list_n[40])
    except ValueError:
        return False
    if n <= 10:
        return False
    else:
        return True
    
def test_min_p(list_p):
    list_p = [list_p[i+10] for i in range(len(list_p)-10)]
    '''
    wide = False
    L_50 = []
    for p in list_p:
        if float(p) < 0.05:
            L_50.append(1)
        else:
            L_50.append(0)
    #print('num of observ p s:', np.sum(np.array(L_10)))
    m = np.sum(np.array(L_50))
    if m >= 50:
        wide = True
    else:
        wide = False
    
    wist = False
    L_70 = []
    for p in list_p:
        if float(p) < 0.25:
            L_70.append(1)
        else:
            L_70.append(0)
    #print('num of observ p s:', np.sum(np.array(L_10)))
    u = np.sum(np.array(L_70))
    if u >= 66:
        wist = True
    else:
        wist = False
    '''
    base = False
    L_70 = []
    for p in list_p:
        if float(p) < 0.1:
            L_70.append(1)
        else:
            L_70.append(0)
    #print('num of observ p s:', np.sum(np.array(L_10)))
    k = np.sum(np.array(L_70))
    if k >= 60:
        base = True
    else:
        base = False
    
    if base:
        L_20 = []
        for p in list_p:
            if float(p) < 0.01:
                L_20.append(1)
            else:
                L_20.append(0)
        #print('num of observ p s:', np.sum(np.array(L_10)))
        n = np.sum(np.array(L_20))
        if n >= 16:
            return True , n
        else:
            return False , n
    else:
        return False , 0

def test_min_p_0(list_p):
    L_10 = []
    for p in list_p:
        if float(p) < 0.01:
            L_10.append(1)
        else:
            L_10.append(0)
    #print('num of observ p s:', np.sum(np.array(L_10)))
    n = np.sum(np.array(L_10))
    if n >= 20:
        return True, n
    else:
        return False, n

def sum_t_values(list_t, list_p):
    sum_t = 0
    for p in list_p:
        if float(p) < 0.01:
            sum_t = sum_t + float( list_t[list_p.index(p)] )
        else:
            pass
    return sum_t

def test_localslope(list_A):
    L_bumpedslope = []
    list_a = [float(a) for a in list_A]
    list_alpha = [list_a[0],list_a[1],list_a[2]]
    for i in range(len(list_a)-6):
        mid_ind = i+3
        L = [ list_a[mid_ind-3],list_a[mid_ind-2],list_a[mid_ind-1],list_a[mid_ind+3],list_a[mid_ind+2],list_a[mid_ind+1] ]
        t, p = stats.ttest_1samp(L, list_a[mid_ind])
        if p < 0.1:
            L_bumpedslope.append(mid_ind)
            list_alpha.append(list_a[mid_ind])
            #print(mid_ind, 'is bumped!')
        else:
            list_alpha.append(np.average(np.array(L)))
    list_alpha = list_alpha + [list_a[-3],list_a[-2],list_a[-1]]
    return list_alpha, L_bumpedslope

def replacebumps(L, L_bumped):
    arr_L = np.array(L)
    for ind in L_bumped:
        arr_L[ind] = 1
    return list(arr_L)
def replacenanp21(l_p):
    l = []
    for p in l_p:
        if float(p) == float(p):
            l.append(float(p))
        else:
            l.append(1)
    return l

def output_productlist(D_product):
    with open('elected product.csv', mode='w', newline='') as file:
        data = csv.writer(file)
        for gender in D_product:
            l_product = D_product[gender]
            data.writerow([gender] + l_product)
    return
def takesecond(l):
    return l[1]

def get_summary(dirstr, corr, pop, medfile = 'summary over all'):
    D_product = dict()
    for gender in D_filename:
        filename = dirstr +'/' +'pop'+pop +'/'+ 'cor'+corr +'/'+ medfile +'/'+ D_filename[gender]
        L_product = []
        D_data = read_summary_file(filename)
        for product in D_data:
            l_slope = D_data[product][0]
            l_p = replacenanp21(D_data[product][2])
            arr_l_P = np.array([float(p) for p in l_p])
            minp = np.min(arr_l_P)
            l_n = D_data[product][4]
            l_t = D_data[product][3]
            #print( test_n(l_n), l_n[40], test_min_p(l_p) )
            if test_n(l_n) and test_min_p(l_p)[0]:
                #print('suspect found')
                sum_t = sum_t_values(l_t, l_p)
                n_obs = test_min_p(l_p)[1]
                l_a,l_bump = test_localslope(l_slope)
                if len(l_bump) == 0:
                    L = product.split('&')
                    product = L[-1]
                    L_product.append([product, n_obs, sum_t])
                else:
                    l_p_new = replacebumps(l_p,l_bump)
                    if test_min_p(l_p_new)[0]:
                        n_obs = test_min_p(l_p)[1]
                        L = product.split('&')
                        product = L[-1]
                        L_product.append([product, n_obs, sum_t])
                    else:
                        pass
            else:
                pass
        #L_product.sort(key=takesecond, reverse = True)
        D_product[pop, corr, gender] = L_product #[ [product name, n, sum_t] ]
    return D_product

def result_classification(key, value):
    if len(value) == 0:
        return 'ND'
    else:
        s = []
        for product in value:
            name = product[0]
            sum_t = product[2]
            if name in key and sum_t > 0:
                s.append('TP')
            elif sum_t > 0:
                s.append('FP')
            else:
                s.append('neg')
        result = ''
        for note in s:
            result = result + note + '/'
        result = result.strip('/')
        return result

def save_file(D_product, filehead = '', filetail = ' pop^cor test.csv'):
    headline = ['pop', 'corr', 'gender', 'TP/FP/ND/neg','TP', 'number of obs male', 'number of obs female', 'sum_t male', 'sum_t female' ]
    for gender in ['female', 'male']:
        with open(filehead +gender+filetail, mode='w', newline='') as file:
            data = csv.writer(file)
            data.writerow(headline)
            for key in D_product:
                if gender in key:
                    line = [key[0],key[1],key[2]]
                    value = D_product[key]
                    line.append( result_classification(key, value) )
                    if 'TP' in result_classification(key, value):
                        line.append(1)
                    else:
                        line.append(0)
                    if len(value) == 1:
                        product_res = value[0]
                        if product_res[0] == 'male':
                            line = line+[product_res[1],'',product_res[2],'']
                        else:
                            line = line+['',product_res[1],'',product_res[2]]
                    elif len(value) == 2:
                        arr = np.zeros(4)
                        for product_res in value:
                            if product_res[0] == 'male':
                                arr[0] = product_res[1]
                                arr[2] = product_res[2]
                            else:
                                arr[1] = product_res[1]
                                arr[3] = product_res[2]
                        line = line + [i for i in arr]
                    else:
                        line = line+['','','','']
                    data.writerow(line)
                else:
                    pass
    return 
    

#set_pointer = 'C:/Users/Admin/Desktop/validation'
#L_corr =[str(round(0.1*(i+1),1)) for i in range(9)]
#L_pop = ['5.7', '6.0', '6.7', '7.0', '7.7', '8.0', '8.7', '9.0']
def get_settable(set_pointer, L_corr=[str(round(0.1*(i+1),1)) for i in range(9)], L_pop= ['5.7', '6.0', '6.7', '7.0', '7.7', '8.0', '8.7', '9.0']):
    D_test_result = dict()
    for pop in L_pop:
        for corr in L_corr:
            D_product = get_summary(set_pointer, corr, pop)
            D_test_result = {**D_test_result, **D_product}
    save_file(D_test_result, filehead = set_pointer+'/', filetail = ' pop^cor test.csv')
    return
