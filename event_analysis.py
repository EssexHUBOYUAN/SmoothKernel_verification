# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 19:49:10 2023

@author: Doc Who
"""

import csv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import event_finding as evf


'''
special varbile:        ['Block sample size']
candidate fix varibles: ['region','product', 'race', 'age']
invesited varibles:     ['slope', 'R2']
'''
#L_age = [i*5 for i in range(18)]
L_age = [i for i in range(86)]
UB = 1000
LB = -1000

#L_guide = ['blankguide','piecewise','smooth','piecewise&smooth']
#L_guide = ['blank','linear','smooth','linearandsmooth']
#L_guide = ['blank']

#L_country = ['Australia','Austria','Bulgaria','Canada','CostaRica','Croatia','Czech','Denmark','Estonia','Ireland','Israel','Japan','Lithuania','Netherlands','NewZealand','Norway','Slovakia','Slovenia','UK']
#L_country = ['Australia', 'Austria', 'Bulgaria', 'Canada']

mag = False


filename_sumN = '-analysis_corr_block.csv'
filename_sumNV = '-analysis_corr_block'

#plotfillter = 'gender'

def read_market(filename = 'test_market'):
    L_region = set()
    L_product = set()
    with open(filename+".csv", mode = 'r', encoding='UTF-8-sig') as file:
        marketdata = csv.reader(file)
        Headline = True
        for line in marketdata:
            if Headline:
                Headline = False
            else:
                L_region.add(line[0])
                L_product.add(line[1])
        file.close()
    L_region = list(L_region)
    L_product = list(L_product)
    #print(L_region, L_product)
    return L_region, L_product

def filename(guide,region,product, strdir=''):
    filename = strdir + guide + '_corr_' + region +'_' + product + '.csv'
    return filename

def readcorr(filename):
    '''
    D_head: {['region'] : 'USA' }
    ''' 
    D_head = dict()
    D_index_head = dict()
    D_data = dict()
    L = filename.split('.csv')
    L = L[0].split("_")
    product = L[-1]
    
    with open(filename, mode='r', encoding='UTF-8-sig') as file:
        L = filename.split('_')
        guide = L[0]
        try:
            L = guide.split('/')
            guide = L[-1]
        except:
            pass
        #product,csv = L[-1].split('.')
        data = csv.reader(file)
        HEAD = True
        for line in data:
            if HEAD:
                i = 0
                for i in range(len(line)-2):
                    D_index_head[i] = line[i]
                    D_head[line[i]] = set()
                    i = 1+i
                HEAD = False
            else:
                for i in range(len(line)):
                    try:
                        key = D_index_head[i]
                        D_head[key].add(line[i])
                    except KeyError:
                        pass
                    D_data[line[0], guide, product, line[1],line[2],line[3]] = [line[4],line[5],line[6]]
                    # Block sample size, guide, product, race, gender, age = a, R2, a_std
                
        file.close()
    for head in D_head:
        D_head[head] = list(D_head[head])
        L = []
        INT = False
        for a in D_head[head]:
            try:
                L.append(int(float(a)))
                INT = True
            except ValueError:
                pass
        if INT:
            D_head[head] = L
            INT = False
        else:
            pass
        D_head[head].sort()
        
    del(D_index_head)
    #print(D_head)
    return D_head, D_data

# lay out some basic math functions
'''
def get_L_R2N2(L_N,L_R2):
    #print('LN:',len(L_N),'LR2:', len(L_R2), ' are the length of Ns and R2s')
    A = 1/evf.SUM([1/N**2 for N in L_N])
    L_R2N2 = []
    for i in range(len(L_N)):
        L_R2N2.append(A*L_R2[i]/L_N[i]**2)
    
    return L_R2N2
'''
def get_L_R2N2(L_N,L_R2):
    #print('LN:',len(L_N),'LR2:', len(L_R2), ' are the length of Ns and R2s')
    A = 1/evf.SUM([1/N for N in L_N])
    L_R2N2 = []
    for i in range(len(L_N)):
        L_R2N2.append(A*L_R2[i]/L_N[i])
    
    return L_R2N2

def get_L_magni(L_R2N2):
    M = 9/(evf.SUM([np.exp(x) for x in L_R2N2]))
    L_magni = []
    for R2N2 in L_R2N2:
        L_magni.append(M*np.exp(R2N2))
    
    return L_magni

def get_L_a_age(L_a,L_R2N2,L_m):
    s = 0
    for i in range(len(L_a)):
        s = s + L_a[i]*L_R2N2[i]*L_m[i]
    return s

def takeSecond(elem):
    return elem[1]

def IntD_over_N(D_data ,mag = False):
    '''
    input:
    D_data is a dictionary with 
    keys = (Block sample size, guide, product, race, gender, age)
    values = [a,R2] 
    
    output:
    D_sumN has
    keys = guide, region, product, race
    values = [a|age=0, a|age=1, ..., a|age=85]
    
    mag: magnification coffcient, found by function(get_L_magni())
    '''
    
    D_prep = dict()
    '''
    {(guide, region, product, race, age): L_N, L_a, L_R2 }
    '''
    for N, guide, product, race, gender, age in D_data:
        #print(product, race, gender, age)
        try:
            D_prep[guide, product, race, gender, age][0].append(int(float(N)))
            D_prep[guide, product, race, gender, age][1].append(float(D_data[N, guide, product, race, gender, age][0]))
            D_prep[guide, product, race, gender, age][2].append(float(D_data[N, guide, product, race, gender, age][1]))
        except KeyError:
            D_prep[guide, product, race, gender, age]=[[int(float(N))],[float(D_data[N, guide, product, race, gender, age][0])],[float(D_data[N, guide, product, race, gender, age][1])]]
    #print(D_prep)
    
    D_output = dict()
    for k_a in D_prep:
        guide, product, race, gender, age = k_a
        L_N,L_a,L_R2 = D_prep[k_a]
        L_R2N2 = get_L_R2N2(L_N, L_R2)
        if mag:
            L_m = get_L_magni(L_R2N2)
        else:
            L_m = [1 for i in range(len(L_N))]
        v_a = get_L_a_age(L_a,L_R2N2,L_m)
        
        try:
            D_output[guide, product, race, gender].append([v_a,int(age)])
        except KeyError:
            D_output[guide, product, race, gender] = [[v_a,int(age)]]
    
    for k_a in D_output:
        D_output[k_a].sort(key=takeSecond)
        D_output[k_a] = np.array([[a[0],0] for a in D_output[k_a]])
        
    del(D_prep)
    return D_output

def IntD_over_age(D_data):
    '''
    input:
    D_data is a dictionary with 
    keys = (Block sample size == 1(of raw), guide, product, race, gender, age)
    values = [a,R2, a_std] 
    
    output:
    D_sumN has
    keys = guide, region, product, race
    values = [a|age=0, a|age=1, ..., a|age=85]
    
    mag: magnification coffcient, found by function(get_L_magni())
    '''
    
    D_prep = dict()
    '''
    {(guide, region, product, race, age): L_N, L_a, L_R2, L_a_std }
    '''
    for N, guide, product, race, gender, age in D_data:
        #print(product, race, gender, age)
        try:
            D_prep[guide, product, race, gender, age][0].append(int(float(N)))
            D_prep[guide, product, race, gender, age][1].append(float(D_data[N, guide, product, race, gender, age][0]))
            D_prep[guide, product, race, gender, age][2].append(float(D_data[N, guide, product, race, gender, age][1]))
            D_prep[guide, product, race, gender, age][3].append(float(D_data[N, guide, product, race, gender, age][3]))
        except KeyError:
            D_prep[guide, product, race, gender, age]=[[int(float(N))],[float(D_data[N, guide, product, race, gender, age][0])],[float(D_data[N, guide, product, race, gender, age][1])],[float(D_data[N, guide, product, race, gender, age][2])]]
    #print(D_prep)
    
    D_output = dict()
    for k_a in D_prep:
        guide, product, race, gender, age = k_a
        L_N,L_a,L_R2,L_std = D_prep[k_a]
        v_std = np.sqrt(evf.SUM([std**2 for std in L_std]))
        v_a = get_L_a_age(L_a,[1 for i in range(len(L_a))],[1 for i in range(len(L_a))])
        try:
            D_output[guide, product, race, gender].append([v_a,int(age),v_std])
        except KeyError:
            D_output[guide, product, race, gender] = [[v_a,int(age),v_std]]

    for k_a in D_output:
        D_output[k_a].sort(key=takeSecond)
        D_output[k_a] = np.array([[a[0],a[2]] for a in D_output[k_a]])
        # 2D array with [[a, std], ... ]
    del(D_prep)
    #print(D_output)
    return D_output

def listproduct(L1,L2):
    L = []
    for item1 in L1:
        for item2 in L2:
            try:
                L.append(item1 + [item2])
            except TypeError:
                L.append([item1,item2])
    return L

def listintuple(L,T):
    R = True
    if type(L) == list:
        for item in L:
            if item not in T:
                R = False
            else:
                pass
    else:
        if L in T:
            pass
        else:
            R = False
    return R

def clearinf(A):
    a, b = np.shape(A)
    for i in range(a):
        for j in range(b):
            if A[i,j] > UB:
                A[i,j] = np.nan
            elif A[i,j] < LB:
                A[i,j] = np.nan
            else:
                pass
    return A

def IntD_over_var(D_sumN, D_head_N, L_var):
    
    '''
    Input:
    L_var list the varibles to be sumed
    the other varibles in D_head_N would be remained, contains: {guide, region, product, race}
    D_sumN = {guide, race, gender) : [a|age=0, a|age=1, ..., a|age=85]}
    
    output:
    D_sumNVs = {(fixed varibles): [a|age=0, a|age=1, ..., a|age=85] }
    '''
    L_order = ['guide','product','race','gender']
    C_normal = 1
    for head in D_head_N:
        if head in L_var:
            print(head)
            C_normal = C_normal*len(D_head_N[head])
        else:
            pass
    print('C_normal is: ',C_normal)
    for head in L_var:
        L_order.remove(head)
    
    D_sumNVs = dict()
    
    # get the list for keys
    size = len(L_order)
    L1 = list(D_head_N[L_order[0]])
    while size > 1:
        L2 = list(D_head_N[L_order[1]])
        L1 = listproduct(L1, L2)
        L_order.pop(0)
        size = size - 1
    #print(L1,'head of D_sumNVs')
    
    
        
    for key in L1:
        #print(key)
        a_s = np.zeros([2,86])
        for k_a in D_sumN:
            #print(k_a)
            if listintuple(key,k_a):
                #a_s = a_s + clearinf( np.transpose(D_sumN[k_a]) )
                a_s = a_s + np.transpose(D_sumN[k_a])
                #print(key, a_s)
            else:
                pass
        #print(a_s)
        if type(key) == str:
            D_sumNVs[key]= a_s/C_normal  
        else:
            D_sumNVs[tuple(key)]=  a_s/C_normal 
    '''        
    for item in D_sumN:
        print(np.transpose(D_sumN[item])[1])
    '''
    del(L1)
    return D_sumNVs


def CorrD_given_var(D_sumNVs, D_head, var):
    
    return 

def output_IntD_N(D_sumN, filename_sumN):
    with open(filename_sumN, mode='w', newline='') as file:
        data = csv.writer(file,dialect = ("excel"))
        headline = ['guide','product','race','gender'] + L_age
        data.writerow(headline)
        for item in D_sumN:
            guide, product, race, gender = item
            line1 = [guide, product, race, gender, 'slope']+list(np.transpose(D_sumN[item][0]))
            data.writerow(line1)
            line2 = [guide, product, race, gender, 'var']+list(np.transpose(D_sumN[item][1]))
            data.writerow(line2)
        file.close()
    print(filename_sumN, ' is saved')
    return

def output_IntD_V(D_sumNVs, filename_sumNV, L_var):
    s = ' '
    headline = ['guide','product','race','gender']
    for word in L_var:
        s = s + '$' + word
        headline.remove(word) 
    headline = headline + ['value_type'] + L_age
    
    with open(filename_sumNV + 'sumover' + s + '.csv', mode='w', newline='') as file:
        data = csv.writer(file,dialect = ("excel"))
        data.writerow(headline)
        for item in D_sumNVs:
            if type(item) == tuple:
                L_heads = list(item)
            else:
                L_heads = [item]
            line1 = L_heads+ ['slope'] + list(D_sumNVs[item][0])
            line2 = L_heads+ ['std'] + list(D_sumNVs[item][1])
            data.writerow(line1)
            data.writerow(line2)
        file.close()
    print(filename_sumNV + 'sumover' + s + '.csv', ' is saved')
    return

def plot_IntD_N(D_sumN):
    plt.xlabel('age')
    plt.ylabel('Slope of C73 Rate vs Market Price')
    for item in D_sumN:
        label = item[0]+' '+item[1]+' '+item[2]
        plt.plot(L_age, list(D_sumN[item]), label = label)
        #plt.legend()
    plt.show()
    print('D_sumN is ploted')
    return

def plot_IntD_V(D_sumNV):
    plt.xlabel('age')
    plt.ylabel('Slope of C73 Rate vs Market Price')
    for item in D_sumNV:
        label = ''
        for word in item:
            label = label + ' ' + word
        plt.plot(L_age, list(D_sumNV[item]), label = label)
        plt.title('males')
        plt.rcParams.update({'font.size':7})
        #plt.legend()
    plt.show()


def analysis_for_each_country(country, marketinfo, L_guide, strdir_evf = ''):
    marketfile = country +'_'+ marketinfo
    L_region, L_product = read_market(marketfile)
    
    D_filehead = dict()
    D_filehead['region'] = L_region
    D_filehead['product'] = L_product
    D_filehead['guide'] = L_guide
    
    L_filename = listproduct(L_guide, L_region)
    #print(L_filename)
    L_filename = listproduct(L_filename, L_product)
    #print(L_filename)
    
    D_allData = dict()
    for guide,region,product in L_filename:
        D_head, D_data = readcorr(filename(guide,region,product, strdir_evf))
        D_allData = {**D_allData,**D_data}
    
    D_head = {**D_head,**D_filehead}
    for head in D_head:
        print(head, 'contains: ', D_head[head])
    
    ext_filename_sumN = country + filename_sumN
    D_sumN = IntD_over_age(D_allData)
    output_IntD_N(D_sumN, ext_filename_sumN)
    
    #plot_IntD_N(D_sumN)
    
    del D_head['Block sample size']
    
    ext_filename_sumNV = country + filename_sumNV
    D_sumNV_G = IntD_over_var(D_sumN, D_head, ['guide'])
    output_IntD_V(D_sumNV_G, ext_filename_sumNV, ['guide'])
    #plot_IntD_V(D_sumNV_G)
    del(D_sumNV_G)
    
     
    '''
    D_sumNV_GR = IntD_over_var(D_sumN, D_head, ['guide','race'])
    output_IntD_V(D_sumNV_GR, filename_sumNV, ['guide','race'])
    plot_IntD_V(D_sumNV_GR)
    del(D_sumNV_GR)
    '''
    '''
    D_sumNV_GGn = IntD_over_var(D_sumN, D_head, ['guide','gender'])
    output_IntD_V(D_sumNV_GGn, filename_sumNV, ['guide','gender'])
    plot_IntD_V(D_sumNV_GGn)
    del(D_sumNV_GGn)
    '''
    '''
    D_sumNV_all = IntD_over_var(D_sumN, D_head, ['guide','race','product'])
    output_IntD_V(D_sumNV_all, filename_sumNV, ['guide','race','product'])
    plot_IntD_V(D_sumNV_all)
    '''
    return
def main(L_country, kernel, strdir_evf, marketinfo = 'ingredients'):
    L_guide = [kernel]
    marketinfo = 'ingredients'
    for country in L_country:
        analysis_for_each_country(country, marketinfo, L_guide ,strdir_evf)
    return

#main()
