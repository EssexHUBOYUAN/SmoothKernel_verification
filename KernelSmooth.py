# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 22:05:52 2023
this file aims to develop a kernel smoothing method for 2D image
@author: Admin

1. read rates file

2.1 smooth by age dependency (kernel of 1Darray_x and y)
2.2 smooth by year dependency (kernel of 1Darray_x and y)
2.3 smooth by dual (kernal of 2Darray_x and y)

3. save output file by standerd blocks

4. read filted results
4.1 

"""
import csv
import numpy as np
import event_finding as evf
#from gunicorn.util import write_error
'''
1. read rates file
'''
def readblock(filename):
    # file : 
    # return dictionary (1, white, male):array([ [r_2000, r_2001, r_2002 ... ]_age=0,[ ],[ ],[ ] ... ])
    with open( filename + '.csv', mode = 'r', encoding='UTF-8-sig') as file:
        filename = filename.split('/')[-1]
        country,guide = filename.split('_')
        blockdata = csv.reader(file)
        head = True
        D_med = dict()
        L_year = []
        for line in blockdata:
            if head:
                head = False
            else:
                if int(line[2]) not in L_year:
                    L_year.append( int(line[2]) )
                D_med[line[1], line[0], 1, int(line[2])] = [float(line[i+3]) for i in range(len(line)-3)]
    D_block = dict()
    for item in D_med:
        gender, country, N, year = item
        try:
            D_block[N, country, gender].append([ D_med[item], year ])
        except KeyError:
            D_block[N, country, gender] = [[ D_med[item], year ]]
    for KEY in D_block:
        D_block[KEY].sort(key = takesecond)
        D_block[KEY] = np.transpose( np.array( [subvalue[0] for subvalue in D_block[KEY]] ) )
    L_year.sort()
    return D_block, L_year

'''
2.0 kernelsmoothing
'''
def test_image(image):
    '''
    Parameters
    ----------
    kernel : array like
        to be averaged item
    weight : str
        the weight type to average the kernel
    Returns
    -------
    array : numpy array type
        to be averaged array.
    N_dims : int
        number of dimentions.

    '''
    if type(image) == list:
        image = np.array(image)
        N_dims = np.ndim(image)
        return image, N_dims
    elif type(image) == np.ndarray:
        N_dims = np.ndim(image)
        return image, N_dims
    else:
        print('TypeError: kernel', type(image))
        return

def smoothing1D(image, r):
    '''
    image: the object to be smoothed
    r: the size of smoothing
    '''
    hr = round(r/2)
    a = np.shape(image)[0]
    newimage = np.zeros(a)
    for i in range(a):
        if i <= hr or (a-i)-1 <= hr:
            if i == 0 or i == a-1:
                newimage[i] = image[i]
            else:
                hspan = min(i, (a-i)-1)
                L = []
                for s in range(hspan*2+1):
                    L.append(image[s+i-hspan])
                arr = np.array(L)
                newimage[i] = np.average(arr)
        elif int(r/2) < r/2:
            L = []
            for s in range(hr*2+1):
                L.append(image[s+i-hspan])
            arr = np.array(L)
            newimage[i] = np.average(arr)
        else:
            L = []
            for s in range(hr*2+1):
                L.append(image[s+i-hspan])
            L = L + L
            L.pop(0)
            L.pop(-1)
            arr = np.array(L)
            newimage[i] = np.average(arr)
    return newimage

def smoothing2D(image, a, b):
    hr = int(a/2)
    hc = int(b/2)
    nrow, ncol = np.shape(image)
    newimage = np.zeros((nrow, ncol))
    for i in range(nrow):
        for j in range(ncol):
            if i <= hr or (nrow-i)-1 <= hr:
                if i == 0 or i == nrow-1:
                    newline = smoothing1D(image[i], b)
                    newimage[i,j] = newline[j]
                    subarr = None
                elif j <= hc or (ncol-j)-1 <= hc:
                    hrspan = min(i, (nrow-i)-1)
                    hcspan = min(j, (ncol-j)-1)
                    if hrspan == i:
                        init_a = True
                    else:
                        init_a = False
                    if hcspan == j:
                        init_b = True
                    else:
                        init_b = False
                    if init_a and init_b:
                        subarr = image[0:2*hrspan+1,0:2*hcspan+1]
                    elif init_a and not init_b:
                        subarr = image[0:2*hrspan+1,ncol-2-2*hcspan:-1]
                    elif not init_a and init_b:
                        subarr = image[nrow-2-2*hrspan:-1,0:2*hcspan+1]
                    elif not init_a and not init_b:
                        subarr = image[nrow-2-2*hrspan:-1,ncol-2-2*hcspan:-1]
                    else:
                        print('WrongCornerSamplingError')
                    newimage[i,j] = np.average(subarr)
                else:
                    hrspan = min(i, (nrow-i)-1)
                    subarr = image[i-hrspan:i+hrspan, j-hc:j+hc]
                    newimage[i,j] = np.average(subarr)
            else:
                if j <= hc or (ncol-j)-1 <= hc:
                    hcspan = min(j, (ncol-j)-1)
                    if hcspan == j:
                        init_b = True
                    else:
                        init_b = False
                    if init_b:
                        subarr = image[ i-hr:i+hr ,0:2*hcspan+1]
                    else:
                        subarr = image[ i-hr:i+hr ,ncol-2-2*hcspan:-1]
                    newimage[i,j] = np.average(subarr)
                else:
                    subarr = image[i-hr:i+hr, j-hc:j+hc]
                    newimage[i,j] = np.average(subarr)
    return newimage

def KernelSmooth(image, r = 4, b= 4):
    '''

    Parameters
    ----------
    image : list or numpy array form (1D or 2D)
        the image to be smoothed,
    r : int, optional
        sampleing size (row axis for 2D case). The default is 4.
    b : int, optional
        sampleing size (colomn axis for 2D case). The default is 4.
    Returns
    -------
    newimage : np.array
        smoothed result image.

    '''
    image, Nd = test_image(image)
    if Nd == 1:
        newimage = smoothing1D(image, r)
    elif Nd == 2:
        newimage = smoothing2D(image, r, b)
    return newimage

'''
2.1 smooth by age dependency (kernel of 1Darray_x and y)
'''
def smoothbyage(D_block_array, r = 4):
    for key in D_block_array:
        a,b = np.shape( D_block_array[key] )
        #print(a,b)
        new_block = []
        for i in range(a):
            L_raw = [rate for rate in D_block_array[key][i]]
            arr_new = KernelSmooth(L_raw, r = r)
            new_block.append(arr_new)
        D_block_array[key] = np.array(new_block)
    return D_block_array
'''
2.2 smooth by year dependency (kernel of 1Darray_x and y)
'''
def smoothbyyear(D_block_array, r = 4):
    for key in D_block_array:
        tr_array = np.transpose(D_block_array[key] )
        a,b = np.shape( tr_array )
        #print(a,b)
        new_block = []
        for i in range(a):
            L_raw = [rate for rate in tr_array[i]]
            arr_new = KernelSmooth(L_raw, r = r)
            new_block.append(arr_new)
        D_block_array[key] = np.transpose( np.array(new_block) )
        #print(np.shape(D_block_array[key]))
    return D_block_array

'''
2.3 smooth by dual (kernal of 2Darray_x and y)
'''
def smoothin2D(D_block_array, a = 4, b = 4):
    for key in D_block_array:
        D_block_array[key] = KernelSmooth(D_block_array[key], r = a, b= b)
    return D_block_array

'''
3. save file
'''
def takesecond(l):
    return l[1]

def save_D_filted(D_2Darray, year1, year2, guide, headpath):
    '''
    Dicttype:
        (n,gender,country): 2D array with shape (? years, 86 years of age)
    save as file:
        2 lines of heads:
        |     |      | country-gender
        | block_size |  age\year  | year1~year2 ...
        | nXn | age | data over age year1~year2
    '''
    
    
    
    headline1 = ['',''] + ['male' for i in range(year2-year1+1)] + ['female' for i in range(year2-year1+1)]
    headline2 = ['block_size','age\year'] + [i+year1 for i in range(year2-year1+1)] + [i+year1 for i in range(year2-year1+1)]
    
    #print(D_2Darray)
    D_ensambled = dict()
    filename = ''
    for key in D_2Darray:
        n,country,gender = key
        
        filename = headpath + country + '_' + guide +'.csv'
        
        try:
            D_ensambled[n].append([D_2Darray[key],gender])
        except KeyError:
            D_ensambled[n] = [[D_2Darray[key],gender]]
    
    
    with open(filename, mode = 'w', newline = '') as file:
        data = csv.writer(file)
        data.writerow(headline1)
        data.writerow(headline2)
    
        for n in D_ensambled:
            A = D_ensambled[n]
            A.sort(key = takesecond, reverse=True)
            
            A = np.concatenate((A[0][0],A[1][0]),axis=1)
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

'''
main function
'''
def main(L_country, ave_axi, path):
    for country in L_country:
        filename = country + '_rate'
        D_Block_array, l_year = readblock(filename)
        year1 = int(l_year[0])
        year2 = int(l_year[-1])
        
        if ave_axi == 'age':
            D_Block_age = smoothbyage(D_Block_array, r = 5)
            save_D_filted(D_Block_age, year1, year2, 'age', path)
            del(D_Block_age)
        elif ave_axi == 'year':
            D_Block_year = smoothbyyear(D_Block_array, r = 5)
            save_D_filted(D_Block_year, year1, year2, 'year', path)
            del(D_Block_year)
        elif ave_axi == 'dual':
            D_Block_dual = smoothin2D(D_Block_array, a = 5, b = 5)
            save_D_filted(D_Block_dual, year1, year2, 'dual', path)
            del(D_Block_dual)
        else:
            print('wrong ave_axi type, choose one among "age", "year", "dual":', ave_axi)
    return
