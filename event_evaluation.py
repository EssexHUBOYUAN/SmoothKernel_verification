# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 14:21:40 2023

@author: Doc Who
"""

'''
this file reads the guide sum over file over different countries and do summary beased on different rating
------------------------------
read and find P values for each pair af (paroduct, race-country) for given gender, for all ages
    provide sorted list of products based on
        1. slope mean values
        2. P values
        3. slope X R^2
output country

combine all countries in the list
    provide sorted list of products based on
        1. slope mean value (std_country provided)
        2. t values (based on P)
        3. SUM of (slpoe X R^2 / popualtion partition)
output combined
'''
import csv
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

L_country = ['Australia','Austria','Bulgaria','Canada','CostaRica','Croatia','Czech','China','Denmark','Estonia','Ireland','Israel','Japan','Lithuania','Netherlands','NewZealand','Norway','Slovakia','Slovenia','UK','USA']

#L_country = ['USA','Japan']
L_varible = ['guide']

#L_special_product = ['Parabens', 'Homosalate', 'Butylated Hydroxylanisole (BHA)', 'Butylated Hydroxyltoluene (BHT)', 'Phosphate Esters (Surfactants)', 'Phosphate Esters (Antifoams)','Benzophenones', 'Ethylhexyl Methoxycinnamate']
L_special_product = ['Phosphate Esters (Antifoams)','Homosalate']
D_continent = {'Australia':'Oceania','Austria':'Europe','Bulgaria':'Europe','Canada':'America','China':'Asia','CostaRica':'America','Croatia':'Europe','Czech':'Europe','Denmark':'Europe','Estonia':'Europe','Ireland':'Europe','Israel':'Asia','Japan':'Asia','Lithuania':'Europe','Netherlands':'Europe','NewZealand':'Oceania','Norway':'Europe','Slovakia':'Europe','Slovenia':'Europe','UK':'Europe','USA':'America'}
marketinfo = 'ingredients'

Continent_country = []
for country in L_country:
    if D_continent[country] == 'Europe':
        Continent_country.append(country)
degfree = 4
logedrate=False

'''________________________________read files_______________________________'''
def takesecond(L):
    return L[1]

def filename(country, L_varible=['guide']):
    file = country + '-' + 'analysis_corr_blocksumover'
    for varible in L_varible:
        file = file + ' $' + varible
    return file

def read(country):
    name = filename(country, L_varible=L_varible)
    D_med = dict()
    with open(name+'.csv', mode='r', encoding='UTF-8-sig') as file:
        data = csv.reader(file)
        Head = True
        for line in data:
            if Head:
                Head = False
            else:
                L = []
                for i in range(len(line)):
                    try: 
                        L.append(float(line[i]))
                    except ValueError:
                        pass
                try:
                    D_med[line[2], line[0], country, line[1]].append([L,line[3]])
                except KeyError:
                    D_med[line[2], line[0], country, line[1]] = [[L,line[3]]]
    
    for item in D_med:
        D_med[item].sort(key=takesecond)
        #print(D_med[item])
        D_med[item] = [ D_med[item][0][0],D_med[item][1][0] ]
        if D_med[item][1] == [np.nan for i in range(86)]:
            D_med[item] = np.array([[np.nan for i in range(86)],[np.nan for i in range(86)]])
        else:
            D_med[item] = np.array(D_med[item])
    # here D_med = { gender, product, country, race : 2Darray(2,86) [ [slope,slope,slope ] , [std, std, std ]]  }
    return D_med

'''________________________________functions_______________________________'''

def find_P_value(u1, sd1, sd2, N=degfree, u2=0):
    (t, p) = stats.ttest_ind_from_stats(mean1=u1, std1=sd1*np.sqrt((N+1)/N), nobs1=N, mean2=u2, std2=sd2*np.sqrt((N+1)/N), nobs2=N)
    return p, t

def find_sum_slope(A):
    #A in 2D, and first row is slope
    a,b = np.shape(A)
    for j in range(b):
        if not A[0,j] == A[0,j]:
            #print(A[i,j])
            A[0,j] = 0
        else:
            pass
    return np.sum(A, axis=1)[0]

def find_max_t(A, i=3):
    #A in 2D and fourth row is t
    a,b = np.shape(A)
    for j in range(b):
        if not A[i,j] == A[i,j]:
            #print(A[i,j])
            A[i,j] = 0
        else:
            pass

    return np.max(A, axis=1)[i]

def find_sum_t(A, i=3):
    #A in 2D and fourth row is t
    a,b = np.shape(A)
    for j in range(b):
        if not A[i,j] == A[i,j]:
            #print(A[i,j])
            A[i,j] = 0
        else:
            pass
    #print(np.sum(A, axis=1)[i])
    return np.sum(A, axis=1)[i]

def find_summary_array(L_A):
    # take onlu slop values, and do statistics based on them (for each age)
    # return a summary array(4,86)
    #N = len(L_A)
    arr_slope = []
    for A in L_A:
        arr_slope.append(A[0])
    arr_slope = np.array(arr_slope)
    # stats.describe(a) ==> nobs=10, minmax=(0, 9), mean=4.5, variance=9.166666666666666, skewness=0.0, kurtosis=-1.2242424242424244
    summary = []
    acrossage = np.transpose(arr_slope)
    for age in acrossage:
        age = array_strip_nan(age)
        try:
            nobs, minmax, mean, var, skew, kurtosis = stats.describe(age)
            sd = np.sqrt(var)
            p,t = find_P_value(mean, sd, sd, N=nobs)
            summary.append([float(mean), float(sd), float(p), float(t), int(nobs)])
        except ValueError:
            nobs = np.nan
            minmax = (np.nan, np.nan)
            mean = np.nan
            var = np.nan
            skew = np.nan
            kurtosis = np.nan
            sd = np.nan
            p = np.nan
            t = np.nan
            summary.append([float(mean), float(sd), float(p), float(t), float(nobs)])
    summary = np.array(summary)
    return np.transpose(summary)

def array_strip_nan(A):
    L = []
    for a in A:
        if not a == a:
            pass
        else:
            L.append(a)
    return np.array(L)

def get_P_value_vs0(D_data):
    D_vs0_data = dict()
    for item in D_data:
        L = []
        A_slope_std = D_data[item]
        A_slope_std = np.transpose(A_slope_std)
        for pair in A_slope_std:
            p, t = find_P_value(pair[0], pair[1], pair[1])
            L.append([pair[0], pair[1], p, t])
        D_vs0_data[item] = np.transpose(np.array(L))
    
    # here D_med = { gender, product, country, race : 2Darray(4,86) [ [slope,slope,slope ] , [std, std, std ], [p,p,p], [t,t,t]]  }
    return D_vs0_data

def get_P_value_vs_genders(D_data):
    D_vsgender_data = dict()
    for item in D_data:
        gender, product, country, race = item
        try:
            D_vsgender_data[product, country, race].append([D_data[item], gender])
        except KeyError:
            D_vsgender_data[product, country, race]= [[D_data[item], gender]]
        
    for item in D_vsgender_data:
        D_vsgender_data[item].sort(key=takesecond)
        L = D_vsgender_data[item]
        # female at first, male at second
        Arr_L = np.transpose(  np.concatenate([L[0][0],L[1][0]])  )
        newL = []
        for pair in Arr_L:
            u1 = pair[0]
            sd1 = pair[1]
            u2 = pair[2]
            sd2 = pair[3]
            t,p = find_P_value(u1, sd1, sd2, N=degfree, u2 = u2)
            newL.append([u1, sd1, u2, sd2, t, p, int(u1>u2)])
        D_vsgender_data[item] = np.transpose(np.array(newL))
    # D_vsgender_data { product, country, race : 2Darray(7,86)[ female[slope,slope,slope ] , female[std, std, std ], male[slope,slope,slope ] , male[std, std, std ], [p,p,p], [t,t,t], [1 is True for female > male, 0 is False]]  }
    return D_vsgender_data

def get_summary_info_for_product(D_data):
    # here, D_data is D_vs0, { gender, product, country, race : array(2,86) slope,std vs age0~86 }
    D_summary_female = dict()
    D_summary_male = dict()
    set_country = set()
    set_product = set()
    set_countryrace = set()
    set_race = set()
    for item in D_data:
        gender, product, country, race = item
        if country == race:
            countryrace = country + '-' + D_continent[country]
        else:
            countryrace = country + '-' + race
        set_countryrace.add(countryrace)
        set_product.add(product)
        set_country.add(country)
        if gender == 'female':
            try:
                D_summary_female[gender, product].append([D_data[item], countryrace])
            except KeyError:
                D_summary_female[gender, product] = [[D_data[item], countryrace]]
        else:
            #gender = 'male'
            try:
                D_summary_male[gender, product].append([D_data[item], countryrace])
            except KeyError:
                D_summary_male[gender, product] = [[D_data[item], countryrace]]
    for product_key in D_summary_female:
        D_summary_female[product_key].sort(key = takesecond)
        L_array_of_countries = [obj[0] for obj in D_summary_female[product_key]]
        Arr_summary = find_summary_array(L_array_of_countries)
        D_summary_female[product_key] = Arr_summary
            
    for product_key in D_summary_male:
        D_summary_male[product_key].sort(key = takesecond)
        L_array_of_countries = [obj[0] for obj in D_summary_male[product_key]]
        Arr_summary = find_summary_array(L_array_of_countries)
        D_summary_male[product_key] = Arr_summary
        
    return D_summary_female, D_summary_male

'''________________________________save files___________________________________'''

def save_sort_by_sumslope():
    return
def save_sort_by_maxt(filename, D, i):
    if i == 3:
        vs0 = True
    else:
        # i == 5
        vs0 = False
        
    L_sort = []
    for item in D:
        T = find_max_t(D[item],i)
        L_sort.append( [item, T] )
    L_sort.sort(key=takesecond, reverse=True)
    with open(filename+'.csv', mode= 'w', newline='') as file:
        data = csv.writer(file)
        if vs0:
            data.writerow(['gender', 'product', 'country', 'race','valuetype'] + [a for a in range(86)])
            for l_item in L_sort:
                item = l_item[0]
                data.writerow( list(item) + ['slope'] + list(D[item][0]) )
                data.writerow( list(item) + ['std'] + list(D[item][1]) )
                data.writerow( list(item) + ['p_value'] + list(D[item][2]) )
                data.writerow( list(item) + ['t_value'] + list(D[item][3]) )
        else:
            data.writerow(['gender', 'product', 'country', 'race','valuetype'] + [a for a in range(86)])
            for l_item in L_sort:
                item = l_item[0]
                data.writerow( ['female'] + list(item) + ['slope'] + list(D[item][0]) )
                data.writerow( ['female']  + list(item) + ['std'] + list(D[item][1]) )
                data.writerow( ['male'] + list(item) + ['slope'] + list(D[item][2]) )
                data.writerow( ['male'] + list(item) + ['std'] + list(D[item][3]) )
                data.writerow( ['compare'] + list(item) + ['p_value'] + list(D[item][4]) )
                data.writerow( ['compare']+ list(item) + ['t_value'] + list(D[item][5]) )
                data.writerow( ['compare']+ list(item) + ['female sensetive'] + list(D[item][6]) )
    print(filename , ' is saved')
    return

def save_sort_by_sumt(filename, D, i):
    if i == 3:
        vs0 = True
    else:
        # i == 5
        vs0 = False
        
    L_sort = []
    for item in D:
        T = find_sum_t(D[item],i)
        L_sort.append( [item, T] )
    L_sort.sort(key=takesecond, reverse=True)
    with open(filename+'.csv', mode= 'w', newline='') as file:
        data = csv.writer(file)
        if vs0:
            data.writerow(['gender', 'product', 'country', 'race','valuetype'] + [a for a in range(86)])
            for l_item in L_sort:
                item = l_item[0]
                data.writerow( list(item) + ['slope'] + list(D[item][0]) )
                data.writerow( list(item) + ['std'] + list(D[item][1]) )
                data.writerow( list(item) + ['p_value'] + list(D[item][2]) )
                data.writerow( list(item) + ['t_value'] + list(D[item][3]) )
        else:
            data.writerow(['gender', 'product', 'country', 'race','valuetype'] + [a for a in range(86)])
            for l_item in L_sort:
                item = l_item[0]
                data.writerow( ['female'] + list(item) + ['slope'] + list(D[item][0]) )
                data.writerow( ['female']  + list(item) + ['std'] + list(D[item][1]) )
                data.writerow( ['male'] + list(item) + ['slope'] + list(D[item][2]) )
                data.writerow( ['male'] + list(item) + ['std'] + list(D[item][3]) )
                data.writerow( ['compare'] + list(item) + ['p_value'] + list(D[item][4]) )
                data.writerow( ['compare']+ list(item) + ['t_value'] + list(D[item][5]) )
                data.writerow( ['compare']+ list(item) + ['female sensetive'] + list(D[item][6]) )
    print(filename , ' is saved')
    return

def save_summary_countries_file(filename, D_summary):
    headline = ['product', 'valuetype'] + [a for a in range(86)]
    with open(filename + '.csv', mode='w', newline = '') as file:
        data = csv.writer(file)
        data.writerow(headline)
        for key_product in D_summary:
            line1 = [key_product[1], 'ave_slope'] + list(D_summary[key_product][0])
            line2 = [key_product[1], 'std'] + list(D_summary[key_product][1])
            line3 = [key_product[1], 'p_value'] + list(D_summary[key_product][2])
            line4 = [key_product[1], 't_value'] + list(D_summary[key_product][3])
            line5 = [key_product[1], 'number of countries'] + list(D_summary[key_product][4])
            data.writerow(line1)
            data.writerow(line2)
            data.writerow(line3)
            data.writerow(line4)
            data.writerow(line5)
    return

def save_special_list(filename, D_data, L_special):
    with open(filename + '.csv', mode='w', newline = '') as file:
        data = csv.writer(file)
        data.writerow(['gender', 'product', 'country', 'race','valuetype'] + [a for a in range(86)])
        for key in D_data:
            for product in L_special:
                if product == key[1]:
                    data.writerow( list(key) + ['slope'] + list(D_data[key][0]) )
                    data.writerow( list(key) + ['std'] + list(D_data[key][1]) )
                    data.writerow( list(key) + ['p_value'] + list(D_data[key][2]) )
                    data.writerow( list(key) + ['t_value'] + list(D_data[key][3]) )
                else:
                    pass
    return
    
def main(L_country, L_country_tails, strdir=''):
    D_all_country_data = dict()
    for country in L_country:
        country_tail = L_country_tails[ L_country.index(country) ]
        D_data = read(country)
        '''
        '''
        D_vs0_data = get_P_value_vs0(D_data)
        name = country + '-slope_t_vs0 (sort maxt)'
        save_sort_by_maxt(name, D_vs0_data, 3)
        #del(D_vs0_data)
        
        D_vsgender_data = get_P_value_vs_genders(D_data)
        name = country + '-slope_t_vsGender (sort sumt)'
        save_sort_by_sumt(name, D_vsgender_data, 5)
        #del(D_vsgender_data)
        
        D_all_country_data = {**D_all_country_data, **D_data}
        
        #del(D_vs0_data)
        #del(D_vsGender_data)
        #del(D_data)
        
        print(country, ' files saved')

    D_summary_female, D_summary_male = get_summary_info_for_product(D_all_country_data)
    name = strdir + 'Impect on female by ' + marketinfo + ' in different countries'
    save_summary_countries_file(name, D_summary_female)
    name = strdir + 'Impect on male by ' + marketinfo + ' in different countries'
    save_summary_countries_file(name, D_summary_male)
    
    '''
    name = 'Specialy noticed products over all countries'
    save_special_list(name, D_all_country_data, L_special_product)
    '''
    print('summary files saved')
    return

