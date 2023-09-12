# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 14:28:40 2023

@author: Doc Who
"""

import csv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import event_finding as evf

p = 0.2
q = 0.8
f_sd = np.sqrt(p*q)
threthold = 301

#L_countryrace = ['Australia','Austria','Bulgaria','Canada','CostaRica','Croatia','Cyprus','Czech','Denmark','Estonia','Iceland','Ireland','Israel','Japan','Lithuania','Malta','Netherlands','NewZealand','Norway','Slovakia','Slovenia','UK-England','UK-Scotland','UK-NorthernIreland']
#L_countryrace = ['Bulgaria','CostaRica','Croatia','Cyprus','Czech','Estonia','Iceland','Ireland','Japan','UK-England(old)','UK-NorthernIreland(old)','UK-Scotland(old)']
#L_countryrace = ['UK-3areas']
#L_countryrace = ['China']
#L_datatype = ['pop']
#L_age_period = [i*5 for i in range(18)]

'''
note:
    only small file canbe converted, file with 0~85-age terms (86) can bot be converted!
'''

def get_5_num(N):
    if N == 0:
        L = [0,0,0,0,0]
        return L
    else:
        if N < threthold:
            L = []
            D_L = {0:0, 1:0, 2:0, 3:0,4:0}
            try:
                choice = np.random.choice([i for i in range(5)],N)
            except ValueError:
                print(N)
                choice = np.random.choice([i for i in range(5)],N)
            for i in choice:
                D_L[i] = D_L[i]+1
            for l in D_L:
                L.append(D_L[l])
        else:
            mean = N*p
            sd = np.sqrt(N)*f_sd
            L = np.random.normal(mean,sd,5)
            Norm = evf.SUM(L)
            L = [round(l*N/Norm) for l in L]
        #print(L)
        #print(evf.SUM(L))      
        return L

def readfile(country,datatype):
    filename = country + '_' + datatype + '.csv'
    D_file = dict()
    ''''
    {country_race, gender, year, age}: number
    number is count/pop
    '''
    with open(filename, mode = 'r', encoding='UTF-8-sig') as file:
        data = csv.reader(file)
        HEAD = True
        for line in data:
            if HEAD:
                HEAD = False
            else:
                for i in range(len(line)):
                    if i >= 3:
                        try:
                            number = int(float(line[i]))
                        except ValueError:
                            number = line[i]
                            number = number.replace('\xa0','')
                            number = round(np.abs(float(number)))
                            
                        D_file[line[0],line[1],line[2],(i-3)] = number
                    else:
                        pass
        
    return D_file

def readfile_raw(country,datatype):
    filename = country + '_' + datatype + '.csv'
    D_file = dict()
    ''''
    {country_race, gender, year, age}: number
    number is count/pop
    '''
    with open(filename, mode = 'r', encoding='UTF-8-sig') as file:
        data = csv.reader(file)
        HEAD = True
        for line in data:
            if HEAD:
                HEAD = False
            else:
                for i in range(len(line)):
                    if i >= 3:
                        D_file[line[0],line[1],line[2],(i-3)*5] = round(np.abs( float(line[i])))
                    else:
                        pass
        
    return D_file

def listproduct(L1,L2):
    L = []
    for item1 in L1:
        for item2 in L2:
            try:
                L.append(item1 + [item2])
            except TypeError:
                L.append([item1,item2])
    return L

def takeSecond(elem):
    return elem[1]

def outputfile(country,datatype,D_data):
    if datatype == 'rate':
        filename = country + '_' + datatype + '.csv'
    else:
        filename = country + '_' + datatype + '_86.csv'
    headline = ['region','gender','year']+[i for i in range(86)]
    D_dataline = dict()
    # transform from "point" type to list type
    for datapoint in D_data:
        country_race, gender, year, age = datapoint
        # data point is a list of 5 numbers
        try:
            D_dataline[country_race, gender, year].append([D_data[datapoint],age])
        except KeyError:
            D_dataline[country_race, gender, year] = [[D_data[datapoint],age]]
    for line in D_dataline:
        D_dataline[line].sort(key=takeSecond)
        L = D_dataline[line]
        L = [l[0] for l in L]
        K = []
        for l in L:
            K = K+l
        D_dataline[line] = K
        
    with open(filename, mode='w', newline='') as file:
        data = csv.writer(file,dialect = ("excel"))
        data.writerow(headline)
        for line in D_dataline:
            data.writerow(list(line)+D_dataline[line])
    print('file: ', filename, ', is done')
    return

def dot2point(N, age):
    if age == 85:
        return [N]
    else:
        return get_5_num(N)

def five_year_2_each_yaer_counts(L_countryrace,L_datatype):
    file_counts = listproduct(L_countryrace,L_datatype)
    for item in file_counts:
        D_counts = readfile_raw(item[0],item[1])
        for dot in D_counts:
            N = D_counts[dot]
            D_counts[dot] = dot2point(N,dot[3])
        outputfile(item[0],item[1],D_counts)
    return


def five_year_2_each_yaer_pop(L_countryrace,L_datatype):
    file_counts = listproduct(L_countryrace,L_datatype)
    for item in file_counts:
        D_counts = readfile_raw(item[0],item[1])
        D_updatecounts = dict()
        for dot in D_counts:
            country_race, gender, year, age = dot
            N2 = D_counts[dot]
            if age == 85:
                D_updatecounts[dot] = [N2]
            elif age == 0:
                N3 = D_counts[country_race, gender, year,5]
                e = 3*N2/25 + 2*N3/25
                c = N2/5
                d = (e+c)/2
                a = 2*N2/5-e
                b = 2*N2/5-d
                D_updatecounts[dot] = [a,b,c,d,e]
            else:
                N1 = D_counts[country_race, gender, year,age - 5]
                N3 = D_counts[country_race, gender, year,age + 5]
                e = 3*N2/25 + 2*N3/25
                a = 3*N2/25 + 2*N1/25
                c = N2*2/5 - a/2 - e/2
                K = N2 - a - e - 3*c
                b = (a+c-K)/2
                d = (e+c-K)/2
                D_updatecounts[dot] = [a,b,c,d,e]
        outputfile(item[0],item[1],D_updatecounts)
        del(D_updatecounts)
        del(D_counts)

def getrate(L_countryrace):
    for country in L_countryrace:
        D_lrate = dict()
        D_pop = readfile(country,'pop_86')
        D_counts = readfile(country,'counts_86')
        for item in D_counts:
            try:
                pop = D_pop[item]
                counts = D_counts[item]
                D_lrate[item] = [counts/pop*100000]
            except KeyError:
                pass
        outputfile(country, 'rate', D_lrate)
    return

#1st five_year_2_each_yaer_pop()
#2nd five_year_2_each_yaer_counts()
#3rd getrate()
    