# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:58:20 2023

@author: Doc Who
"""


import csv
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from tkinter import _flatten

'''probable initial values for age_year curves'''
#L_initis = [[0.05, 0.0001, 66, 7],[0.01, 0.0001, 66, 7],[0.005, 0.00006, 66, 7],[0.005, 0.00006, 63, 7]] # for SEER17
L_initis = [[0.001,0.0000025,70,10],[0.05,0.00001,76,3],[0.01,0.00001,80,3],[0.05,0.00001,50,10],[0.05,0.00001,90,3],[0.0025,0.00006,90,10],[0.001,0.00001,60,7],[0.005,0.01,100,8],[0.006,0.07,95,5]]



#L_country = ['Australia','Austria','Bulgaria','Canada','CostaRica','Croatia','Czech','Denmark','Estonia','Ireland','Israel','Japan','Lithuania','Netherlands','NewZealand','Norway','Slovakia','Slovenia','UK']
#L_country = ['Australia', 'Austria', 'Bulgaria', 'Canada']
marketvalue = 'ingredients'
L_guide = ['blank','linear','smooth','linearandsmooth']
#L_guide = ['blank']

rawtag = 'raw'

log = True

def read_blocks(filename):
    # return dictionary (male, white, 1):array([ [r_2000, r_2001, r_2002 ... ]_age=0,[ ],[ ],[ ] ... ])
    with open( filename + '.csv', mode = 'r', encoding='UTF-8-sig') as file:
        L = filename.split('/')
        filename = L[-1]
        country,guide = filename.split('_')
        
        blockdata = csv.reader(file)
        RACE = True
        YEAR = True
        
        set_Race = set()
        set_Year = set()
        D_Race = dict()
        D_Year = dict()
        D_Block = dict()
        D_Block_array = dict()
        
        for line in blockdata:
            if RACE:
                # get a dictionary:  D_Race = { index : [race,gender] }
                for i in range(len(line)):
                    head = line[i]
                    if head != '':
                        set_Race.add(line[i])
                        try:
                            race,gender = head.split('-')
                        except ValueError:
                            try:
                                L = country.split('/')
                                race = L[-1]
                            except:
                                race = country
                            gender = head
                        D_Race[i] = [race,gender]
                    else:
                        pass
                    
                RACE = False
                #print(D_Race)
            elif YEAR:
                # get a dictionary: D_Year = {index: year}
                for i in range(len(line)):
                    try:
                        D_Year[i] = float(line[i])
                        set_Year.add(int(line[i]))
                    except ValueError:
                        pass
                YEAR = False
                #print(D_Year)
            else:
                # first 2 cols: | n X n or '' | 'age' |
                # get a dictionary: D_Block[N, age, race, gender]
                if line[1] == '0':
                    n, m = line[0].split('X')
                    N = int(n)
                    for i in range(len(line)):
                        if i > 1:
                            try:
                                L = D_Block[N,int(line[1]),D_Race[i][0],D_Race[i][1]]
                                L.append(float(line[i]))
                                D_Block[N,int(line[1]),D_Race[i][0],D_Race[i][1]] = L
                            except KeyError:
                                D_Block[N,int(line[1]),D_Race[i][0],D_Race[i][1]] = [np.abs(float(line[i]))]
                        else:
                            pass
                else:
                    #print(line)
                    for i in range(len(line)):
                        if i > 1:
                            try:
                                L = D_Block[N,int(line[1]),D_Race[i][0],D_Race[i][1]]
                                L.append(float(line[i]))
                                D_Block[N,int(line[1]),D_Race[i][0],D_Race[i][1]] = L
                            except KeyError:
                                D_Block[N,int(line[1]),D_Race[i][0],D_Race[i][1]] = [np.abs(float(line[i]))]
                        else:
                            pass
                
        for item in D_Block:
            n, age, race, gender = item
            try:
                L = D_Block_array[n,race,gender]
                L.append(D_Block[item])
                D_Block_array[n,race,gender] = L
                #print(len(D_Block[item]))
            except KeyError:
                D_Block_array[n,race,gender] = [D_Block[item]]
                #print(len(D_Block[item]))
        
        for Block in D_Block_array:
            D_Block_array[n,race,gender] = np.array(D_Block_array[n,race,gender])
            #print(np.ndim(D_Block_array[n,race,gender]))
        
    return D_Block_array, D_Race, D_Year, set_Race, set_Year


def read_market(filename):
    # file form:
    # years ...
    # country | type of product | number (market size) ...
    with open(filename+".csv", mode = 'r', encoding='UTF-8-sig') as file:
        marketdata = csv.reader(file)
        
        L_year = []
        D_market = dict()
        
        Head = True
        for line in marketdata:
            if Head:
                for year in line:
                    try:
                        y = float(year)
                        L_year.append(y)
                    except ValueError:
                        pass
                Head = False
            else:
                for i in range(len(line)):
                    if i <= 1:
                        pass
                    else:
                        try:
                            try:
                                money = float(line[i])
                                if money != 0:
                                    if log:
                                        money = np.log10(money)
                                    else:
                                        pass
                                    D_market[line[0],line[1]].append(money)
                                else:
                                    D_market[line[0],line[1]].append(np.nan)
                            except ValueError:
                                D_market[line[0],line[1]].append(np.nan)
                        except KeyError:
                            try:
                                money = float(line[i])
                                if money != 0:
                                    if log:
                                        money = np.log10(money)
                                    else:
                                        pass
                                    D_market[line[0],line[1]] = [money]
                                else:
                                    D_market[line[0],line[1]] = [np.nan]
                            except ValueError:
                                D_market[line[0],line[1]] = [np.nan]

    return D_market,L_year

def output_block(D_Block):
    Begin = True
    Sec = True
    for item in D_Block:
        if True:
            with open("test_block.csv", mode='w', newline = '')as w:
                writer = csv.writer(w,dialect = ("excel"))
                for row in D_Block[item]:
                    writer.writerow(row)
            
            #plot_3DBlock(D_Block[item],item)
            
            Begin = False
            
        else:
            if Sec:
                # put printing part here if need second block
                Sec = False
            else:
                pass
    return
                
def output_corr(D_corr, D_market, typename, head=''):
    '''save a .csv file for corrilation coiff'''
    for country, product in D_market:
        filename = head + typename + '_corr_' + country +'_' + product + '.csv'
        Head = ['Block sample size','race','gender','age','slope','R2', 'slope_var']
        with open(filename, mode = 'w', newline= '') as w:
            writer = csv.writer(w,dialect = ("excel"))
            writer.writerow(Head)
            for item in D_corr:
                L = list(item)
                if country == L[0] and product == L[1]:
                    L.pop(0)
                    L.pop(0)
                    L.append(D_corr[item][0][0])
                    L.append(D_corr[item][1])
                    L.append(D_corr[item][2])
                    writer.writerow(L)
                else:
                    pass
        #print(filename, ' is done')
    
    return
def save_D_filted(D_2Darray, year1, year2, country, tail='2Dreg', head = ''):
    '''
    Dicttype:
        (n,race,gender): 2D array with shape (? years, 86 years of age)
    save as file:
        2 lines of heads:
        |     |      | country-gender
        | block_size |  age\year  | year1~year2 ...
        | nXn | age | data over age year1~year2
    '''
    
    
    
    headline1 = ['','']
    headline2 = ['block_size','age\year']
    #print(D_2Darray)
    D_ensambled = dict()
    filename = ''
    for key in D_2Darray:
        n,race,gender = key
        
        filename = head + country + '_' + tail +'.csv'
        race_gender = race + '-' + gender
        try:
            D_ensambled[n].append([D_2Darray[key],race_gender])
            headline1 = headline1 + [race_gender for i in range(year2-year1+1)]
            headline2 = headline2 + [i+year1 for i in range(year2-year1+1)]
        except KeyError:
            headline1 = headline1 + [race_gender for i in range(year2-year1+1)]
            headline2 = headline2 + [i+year1 for i in range(year2-year1+1)]
            D_ensambled[n] = [[D_2Darray[key],race_gender]]
    
    with open(filename, mode = 'w', newline = '') as file:
        data = csv.writer(file)
        data.writerow(headline1)
        data.writerow(headline2)
    
        for n in D_ensambled:
            A = D_ensambled[n]
            A.sort(key = takesecond, reverse=True)
            L_A = tuple([A[i][0] for i in range(len(A))])
            A = np.concatenate(L_A, axis=1)
            print('shape of med array is:', np.shape(A))
            age = 0
            for row in A:
                if age == 0:
                    line = [ str(n)+'X'+str(n) , age] + list(row)
                else:
                    line = ['',age] + list(row)
                data.writerow(line)
                age = age+1
        file.close()
    print(filename, 'is saved')
    del(D_2Darray)
    return

def get_sub_array(D_array):
    D_subblock = dict()
    for key in D_array:
        a, b = np.shape(D_array[key])
        #print(a, b)
        A = []
        Arr_block = D_array[key]
        #print(type(Arr_block),Arr_block)
        for i in range(18):
            L = Arr_block[5*i]
            A.append(L)
        A = np.array(A)
        #print('shape of sub block is ', np.shape(A))
        D_subblock[key] = A
    return D_subblock
''' get weight of Block '''
def takesecond(L):
    return L[1]
def Nlinear(lx, a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15):
    try:
        a,b = np.shape(lx)
        if a < 15:
            Zero_array = np.zeros((15-a, b))
            lx = np.concatenate((lx,Zero_array), axis=0)
        else:
            pass
    except ValueError:
        a = np.shape(lx)
        a = int(a[0])
        if a < 15:
            Zero_array = np.zeros((15-a,))
            lx = np.concatenate((lx,Zero_array), axis=0)
    
    return lx[0]*a1 + lx[1]*a2 + lx[2]*a3 + lx[3]*a4 + lx[4]*a5 + lx[5]*a6 + lx[6]*a7 + lx[7]*a8 + lx[8]*a9 + lx[9]*a10 + lx[10]*a11 + lx[11]*a12 + lx[12]*a13 + lx[13]*a14 + lx[14]*a15

def N5linear(x, a1,a2,a3,a4,a5):
    '''
    length = np.shape(lx)
    lweight = [lweight[i] for i in range(len(length))]
    '''
    return x[0]*a1 + x[1]*a2 + x[2]*a3 + x[3]*a4 + x[4]*a5

def N1linear(x, a1):
    return a1*x

def get_weightofBlock(D_aim_array, D_sample_array, year2008, year2012, pop):
    D_weight = dict() # (race, gender) : [weight, n],[weight, n]
    D_med = dict() # (race, gender) : [array, n], [array, n]
    #L_initi = [1.1]
    L_initi = [0 for i in range(15)]
    
    for key in D_aim_array:
        N, race, gender = key
        D_med[race, gender] = [[np.array(D_aim_array[key]), N]]
    for key in D_sample_array:
        N, race, gender = key
        D_med[race, gender].append([D_sample_array[key], N])
    for KEY in D_med:
        D_med[KEY].sort(key=takesecond)
        L_lxT = []
        for Block in D_med[KEY]:
            A, N = Block
            a,b = np.shape(A)
            if N == 0:
                #print('Aim is found')
                A = A[:,year2008:year2012+1]
                #A = np.transpose(A)
                arr_ly = A.flatten()
            elif N <= 5:
                #print('sample',N,'is ensambled')
                A = A[:,year2008:year2012+1]
                #A = np.transpose(A)
                L_lxT.append(list(A.flatten()))
            else:
                
                A = A[:,year2008:year2012+1]
                #A = np.transpose(A)
                L_lxT.append(list(A.flatten()))
                
                pass
            # now N is the number of parameters need to be fitted

        arr_lx = np.array(L_lxT)
        
        #arr_lx = arr_lx[0]
        #print(arr_lx)
        #print(arr_ly)
        try:
            arr_weight, arr_corr = curve_fit(Nlinear, arr_lx, arr_ly, L_initi)
        except RuntimeError:
            arr_weight, arr_corr = curve_fit(Nlinear, arr_lx, arr_ly)
        #arr_weight, arr_corr = curve_fit(N5linear, arr_lx, arr_ly)
        
        D_weight[KEY] = [[arr_weight[i], i+1] for i in range(len(arr_weight))]
        ''' plot fig to check 
        fig = plt.figure(figsize=(16,6))
        ax1 = fig.add_subplot(1,2,1) # left side, avergaes of slopes and std
        ax2 = fig.add_subplot(1,2,2) # right side, p values, in log, reversed
        ax1.plot(np.arange(90),arr_ly)
        for line in L_lxT:
            ax2.plot(np.arange(90), np.array(line))
            #ax1.plot(np.arange(18), np.array(line))
        a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15 = arr_weight
        #a1,a2,a3,a4,a5 = arr_weight
        lye = [Nlinear(lx, a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15) for lx in np.transpose(arr_lx)]
        #lye = [N5linear(lx, a1,a2,a3,a4,a5) for lx in np.transpose(arr_lx)]
        ax1.plot(np.arange(90),np.array(lye))
        '''
        a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15 = arr_weight
        lye = [Nlinear(lx, a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15) for lx in np.transpose(arr_lx)]
        R2 = R_square(list(arr_ly), lye)
        print(KEY, 'has R2', R2)
        #print(D_weight[KEY])
    #plt.show()
    #print(D_weight)
    del(D_med)
    return D_weight

def get_combinedBlock(D_weight, D_Block):
    D_combined = dict()
    D_med = dict()
    for key in D_Block:
        n,race, gender = key
        try:
            D_med[race,gender].append([D_Block[key],n])
        except KeyError:
            D_med[race,gender] = [[D_Block[key],n]]
    for race, gender in D_med:
        D_med[race, gender].sort(key = takesecond)
        N = len(D_med[race, gender])
        D_weight[race, gender].sort(key = takesecond)
        a = len(D_weight[race, gender])
        while a < N:
            a = a+1
            D_weight[race, gender].append([0,a+1])
        else:
            pass
        for i in range(N):
            #print(np.array(D_med[race, gender][i][0]))
            #print(type(D_weight[race, gender][i][0]))
            L_newBlock = [np.array(D_med[race, gender][i][0])*D_weight[race, gender][i][0] for i in range(N)]
        D_combined[1, race, gender] = np.abs(sum(L_newBlock))
        #print(np.shape(D_combined[1, race, gender]))
        '''
        plot_3DBlock(D_combined[1, race, gender],[race,gender])
        '''
    return D_combined

def age_curve(x, a,b,c,d):
    return -a/b*(np.log(np.sqrt(b)*x+1)-np.sqrt(b)*x)/(np.exp((x-c)/d)+1)

def age_fit(lx,ly):
    for candidate in L_initis:
        TryNext = False
        NotBest = True
        try:
            initis = candidate
            fitParams, fitCovariances = curve_fit(age_curve, lx, ly, initis)
            a,b,c,d = fitParams
            if a < 0  or c > 125 or c < 12 or d < 0:
                #print('try next initis for age curve fitting')
                TryNext = True
            else:
                pass
            if TryNext:
                pass
            else:
                #print(fitParams,' is found')
                NotBest = False
                break
        except RuntimeError:
            #print('try next initis for age curve fitting')
            pass
    if NotBest:
        print('finally', fitParams,' is found')
    else:
        pass
    #print(fitParams)
    return fitParams

def get_expbyage(D_Block, initis = [0.05, 0.0001, 66, 7]):
    D_Block_exparray = dict()
    D_col_para = dict()
    for item in D_Block:
        matrix = D_Block[item]
        matrixT = np.transpose(matrix)
        L_age = [x for x in range(86)]
        exp_matrix = []
        for row in matrixT:
            try:
                a,b,c,d = age_fit(L_age,row) #fitParams 
            except UnboundLocalError:
                a,b,c,d = [np.nan, np.nan, np.nan, np.nan]
                print(item, ' wrong initis error, fix initis!')
            new_row = [age_curve(x, a,b,c,d) for x in L_age]
            D_col_para[item] = a,b,c,d
            exp_matrix.append(new_row)
        exp_matrixT = np.transpose(np.array(exp_matrix))
        D_Block_exparray[item] = exp_matrixT
    return D_Block_exparray, D_col_para

def get_max_slop(D_exp_Block):
    D_diff = get_diffarray_age(D_exp_Block)
    output_block(D_diff)
    D_max_slop = dict()
    for item in D_diff:
        a,b = np.shape(D_diff[item])
        #print(b)
        L_dmax = []
        for i in range(b):
            dmax = D_diff[item][:,i].max()
            arr_dpos = np.where(dmax == D_diff[item][:,i])
            #print(len(arr_dpos[0]))
            dpos = arr_dpos[0][0]
            L_dmax.append([(dpos,i),dmax])
        #print(L_dmax)
        D_max_slop[item] = L_dmax
    return D_max_slop

def get_max_peak(D_exp_Block):
    D_max_peak = dict()
    for item in D_exp_Block:
        a,b = np.shape(D_max_peak[item])
        #print(b)
        L_pmax = []
        for i in range(b):
            pmax = D_exp_Block[item][:,i].max()
            arr_ppos = np.where(pmax == D_exp_Block[item][:,i])
            print(len(arr_ppos[0]))
            ppos = arr_ppos[0][0]
            L_pmax.append([(ppos,i),pmax])
        D_max_peak[item] = L_pmax
    return D_max_peak
            
def get_diffarray_agerate(D_Block, D_Block_exparray):
    '''get difference between measure and exp'''
    D_diff = dict()
    for item in D_Block:
        M_measure = D_Block[item]
        M_expect = D_Block_exparray[item]
        D = (M_measure - M_expect)
        ZeroNan(D)
        D_diff[item] = D
        
    return D_diff

def get_diffarray_age(D_Block, d=1):
    '''get difference across d years difference'''
    D_diff = dict()
    for item in D_Block:
        M = D_Block[item]
        Fore = M
        Hind = M
        for i in range(d):
            Fore = np.delete(Fore,-1,axis=0)
            Hind = np.delete(Hind,0,axis=0)
        D = (Hind - Fore)
        ZeroNan(D)
        D_diff[item] = D
        #print(np.shape(D_diff[item]))
    return D_diff

def get_diffarray_year(D_Block, d=1):
    '''get difference across d years difference'''
    D_diff = dict()
    for item in D_Block:
        M = D_Block[item]
        Fore = M
        Hind = M
        for i in range(d):
            Fore = np.delete(Fore,-1,axis=1)
            Hind = np.delete(Hind,0,axis=1)
        D = (Hind - Fore)
        ZeroNan(D)
        D_diff[item] = D
        #print(np.shape(D_diff[item]))
    return D_diff

def get_diffarray_ageyear(D_Block, r=1, lam = 1):
    '''get difference across r years difference, and considering variation in age 
    as the propendicular projection of a_r (lambda)'''
    D_diff = dict()
    for item in D_Block:
        M = D_Block[item]
        Fore = M
        Hind_age = M
        Hind_year = M
        for i in range(r):
            Fore = np.delete(np.delete(Fore,-1,axis=1),-1, axis=0)
            Hind_age = np.delete(np.delete(Hind_age,0,axis=1),0, axis=0)
            Hind_year = np.delete(np.delete(Hind_year,0,axis=1),-1, axis=0)
        D = (lam*Hind_age + (1-lam)*Hind_year) - Fore
        #print(np.shape(D))
        D_diff[item] = D
    
    return D_diff
    
def ZeroNan(A):
    a, b = np.shape(A)
    for j in range(a):
        for i in range(b):
            if A[j,i] > -1 and A[j,i] < 11:
                pass
            elif A[j,i] >= 1:
                A[j,i] = 1
            elif A[j,i] <= -1:
                A[j,i] = -1
            else:
                A[j,i] = 0
    return 

def plot_test(fitParams,lx,ly):
    a,b,c,d = fitParams
    y2 = []
    for x in lx:
        y2.append(age_curve(x, a,b,c,d))
    
    plt.plot(lx, ly, y2)
    plt.show()
    return


def linear(x,a,b):
    return a*x+b

def SUM(List):
    S = 0
    for item in List:
        S = S + item
    return S

def Var(l_y):
    Sum = SUM(l_y)
    Ave = Sum/len(l_y)
    SS = 0
    for y in l_y:
        SS = SS + (y-Ave)**2
    return SS/len(l_y)

def R_square(l_y, l_ye):
    if len(l_y) == len(l_ye):
        pass
    else:
        print('non homo lists in R_square calc')
        return
    Var_y = Var(l_y)
    Diff = 0
    for i in range(len(l_y)):
        y = l_y[i]
        ye = l_ye[i]
        Diff = Diff + (y - ye)**2
    #print ('suare of difference is ', Diff/len(l_y))
    #print ('variance is ', Var_y)
    return 1 - (Diff/len(l_y)/Var_y)

def BlockVsTrend(D_Block, D_market, D_year, L_year, Range=2):
    ''' D_Block[N, race, gender] = 2Darray_[(age, year)]
        D_market[country, product] = list_[market price]
        D_year[index #2+] = year
        L_year = [year] from market info
        Range := extracted number of itterations
        
        return D_corr_results['USA', 'Beauty and Personal Care', 1.0, 'black', 'male', age=84] = fit_param_1Darray, R2
    '''
    D_corr_results = dict()
    
    L_yr, begin_Block = common_years(D_year, L_year)
    #print(L_yr, begin_Block)
    for product in D_market:
        for array in D_Block:
            age = 0
            L_R2=[]
            L_age=[]
            L_a = []
            for age_line in D_Block[array]:
                L_x = []
                for i in range(len(age_line)):
                    if i >= begin_Block and len(L_x) < len(L_yr):
                        L_x.append(age_line[i])
                    else:
                        pass
                L_y = []
                for j in range(len(D_market[product])):
                    if len(L_y) < len(L_yr):
                        L_y.append(D_market[product][j])
                    else:
                        pass

                #head off the uncapable parts of x,y pair
                L_x,L_y = headofflxly(L_x,L_y)
                '''
                curve_fit and get a,b, R^2, kai^2; 
                note: the lable of L_x and L_y are reversed
                '''
                #print('x values are rates: ',L_x)
                #print('y values are market: ',L_y)
                
                L_initis = [[0,0],[0.1,0],[-0.1,0]]
                if len(L_x) > 2:
                    try:
                        for initis in L_initis:
                            try:
                                fit_param, fitCovariances = curve_fit(linear, np.array(L_y), np.array(L_x), initis)
                                break
                            except RuntimeError:
                                #print('linear slope fitting error, try next one')
                                pass
                        a,b = fit_param
                        std_a = fitCovariances[0,0]
                        L_ye = [linear(x,a,b) for x in L_y]
                        R2 = R_square(L_x, L_ye)
                    except RuntimeError or ValueError:
                        #print(product, 'has time error in BVTr')
                        fit_param = [np.nan,np.nan]
                        a,b = fit_param
                        R2 = [np.nan]
                    #print(age, a, R2.round(2))
                else:
                    fit_param = [np.nan,np.nan]
                    std_a = np.nan
                    a,b = fit_param
                    R2 = np.nan
                    
                L_age.append(age)
                L_R2.append(R2)
                L_a.append(a)
                
                '''
                store in dictionary
                '''
                key = product + array + tuple([age])
                D_corr_results[key] = fit_param, R2, std_a
                
                age = age+1
                #print(age)
            #print('corr with market info with ', product + array, ' is done' )
            '''
            s = ''
            for info in array:
                s = s + str(info) + ' '
            for info in product:
                s = s + str(info) + ' '
            plt.plot(L_age, L_R2)
            plt.title(s)
            plt.show()
            '''
            
            
    #print(D_corr_results)
    return D_corr_results

def common_years(D_year, L_year):
    L = set()
    for index in D_year:
        try:
            I = L_year.index(D_year[index])
            L.add(L_year[I])
        except ValueError:
            pass
    L=list(L)
    L.sort()
    y0 = L[0]
    for index in D_year:
        if D_year[index] == y0:
            begin_Block = index
            break
        else:
            pass
            
    #print(y0)
    return L, begin_Block-2

def headofflxly(L_x,L_y):
    lx = []
    ly = []
    for i in range(len(L_x)):
        if L_x[i] == L_x[i] and L_y[i] == L_y[i]:
            lx.append(L_x[i])
            ly.append(L_y[i])
        else:
            pass
    del(L_x)
    del(L_y)
    return lx, ly

def hight(age, year, Block_array):
    #print(type(Block_array))
    a = len(age)
    b = len(year)
    M = []
    i = 1
    for line in Block_array:
        while len(line) < a:
            line = np.hstack((line,0))
            #print(i)
            i = i+1
        else:
            M.append(line)
    
    return np.transpose(np.array(M))

def plot_3DBlock(Block_array,item, what=''):
    s = ''
    for word in item:
        s = s + str(word) + ' '
    s = s + what
    
    Block_array = np.array(Block_array)
    #print(M)
    '''setup color'''
    Facecolor = 'antiquewhite'
    #Malecolor = 'navy'
    #Femalecolor = 'tomato'
    #Neutrocolor = 'cadetblue'
    #Widthofline = 2
    #Maleline = 1
    #Femaleline = 1
    #MaleaSD = 0.3
    #FemaleaSD = 0.3
    #S = 10
    
    plt.rcParams['axes.facecolor'] = Facecolor
    fig = plt.figure(figsize=(8, 8), facecolor=Facecolor)
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.array([int(x) for x in range(86)])
    y = np.array([int(y) for y in range(86)])
    #print(type(x), type(y))
    age, year = np.meshgrid(x, y)
    high = hight(age, year, Block_array)
    plt.title(s)
    
    ax.plot_surface(age, year, high, alpha=0.9, cstride=1, rstride = 1, cmap='rainbow')
    plt.show()
    return
    

def fitting_test():
    with open('test.csv', mode='r', encoding = 'UTF-8-sig') as file:
        xy = csv.reader(file)
        i = 0
        for line in xy:
            if i == 0:
                lx = [float(u) for u in line]
                i = i+1
            else:
                ly = [float(u) for u in line]
    
    fitParams = age_fit(lx,ly)
    plot_test(fitParams,lx,ly)
    return

def find_overlap(L1,L2):
    L1.sort()
    L2.sort()
    begin = max(L1[0],L2[0])
    end = min(L1[-1],L2[-1])
    return begin, end

def for_each_country_main(country, country_tail, marketvalue, strdir, pop, kernel):
    guide = kernel
    blockfile = country + '_' + guide
    rawfile = country + '_' + rawtag
    marketfile = country + '_' + marketvalue
    
    D_Block, D_Race, D_Year, set_Race, set_Year = read_blocks(blockfile)
    D_market,L_year_market = read_market(marketfile)
    D_Block_raw = read_blocks(rawfile)[0]
    L_year_block = list(set_Year)
    
    begin_year, end_year = find_overlap(L_year_block, L_year_market)
    
    
    for i in D_Year:
        #print(D_Year[i])
        if D_Year[i] == begin_year:
            ind_begin_block = i-2
            break
        else:
            pass
    for i in D_Year:
        if D_Year[i] == end_year:
            ind_end_block = i-2
            break
        else:
            pass
        
    ''' take D_black and D_block_raw find corresponding subblocks '''
    #D_subBlock = get_sub_array(D_Block)
    ''' find weight by regression (can be modified with wight by age and year) '''
    #D_weight = get_weightofBlock(D_Block_raw,D_subBlock, ind_begin_block, ind_end_block, pop)
    ''' get combined result by weight'''
    #D_combined = get_combinedBlock(D_weight, D_Block)
    #plot(D_combined)
    #save_D_filted(D_combined, L_year_block[0], L_year_block[-1], country_tail, head=strdir)
    
    #D_Block_exparray, D_col_para = get_expbyage(D_combined)
    #D_diff_array = get_diffarray_agerate(D_Block, D_Block_exparray)
    
    #D_diff_year = get_diffarray_year(D_Block)
    #D_L_dmax = get_max_slop(D_Block_exparray)
    #D_res_lam = get_diffarray_ageyear(D_diff_array, r=1, lam=1)
    #output_block(D_Block_exparray)
        
    '''
    for item in D_market:
        print(item)
        print(D_market[item])
    '''
    #print(D_Year)
    
    D_corr_Block = BlockVsTrend(D_Block, D_market, D_Year, L_year_market)
    #print(len(D_corr_Block))
    output_corr(D_corr_Block, D_market, guide, head = strdir)
    del(D_corr_Block)
    
    '''
    D_corr_Block_ageexp = BlockVsTrend(D_Block_exparray, D_market, D_Year, L_year_market)
    output_corr(D_corr_Block_ageexp, D_market, guide, head = strdir)
    del(D_corr_Block_ageexp)
    '''
    '''
    D_corr_Block_residue = BlockVsTrend(D_res_lam, D_market, D_Year, L_year_market)
    output_corr(D_corr_Block_residue, D_market, guide)
    del(D_corr_Block_residue)
    '''
    return

def main(L_country,poptail,L_country_tails,strdir, kernel, marketvalue = 'ingredients'):
    l = poptail.split('p')
    pop = l[-1]
    for country in L_country:
        country_tail = L_country_tails[ L_country.index(country) ]
        for_each_country_main(country, country_tail, marketvalue,strdir, pop, kernel)
    return
#main()
