# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 20:38:30 2022

@author: Admin
"""
from scipy.optimize import curve_fit
import numpy as np
#import cv2
import matplotlib.pyplot as plt
import csv

dict_sex = dict()
dict_sex['0']='male'
dict_sex['1']='female'

dict_race = dict()
dict_race['0']='white'
dict_race['1']='black'
dict_race['2']='asian&pacific'
dict_race['3']='native_american'
dict_race['4']='ALL'

#L_countryrace = ['Australia','Austria','Bulgaria','Canada','CostaRica','Croatia','Cyprus','Czech','Denmark','Estonia','Iceland','Ireland','Israel','Japan','Lithuania','Malta','Netherlands','NewZealand','Norway','Slovakia','Slovenia','UK-3areas']
#L_countryrace = ['China']
L_guide = ['blank','smooth','linear','linearandsmooth']
N = 255

#L_initis = [[0.05, 0.0001, 66, 7],[0.01, 0.0001, 66, 7],[0.005, 0.00006, 66, 7],[0.005, 0.00006, 63, 7]]

#L_initis = [[0.001,0.0000025,70,10],[0.05,0.00001,76,3],[0.01,0.00001,80,3],[0.05,0.00001,50,10],[0.05,0.00001,90,3]]
L_initis = [[0.001,0.0000025,70,10],[0.05,0.00001,76,3],[0.01,0.00001,80,3],[0.05,0.00001,50,10],[0.05,0.00001,90,3],[0.0025,0.00006,90,10],[0.001,0.00001,60,7],[0.005,0.01,45,12],[0.006,0.07,95,5]]  

def imread(filename):
    print(dict_sex,dict_race)
    dict_8all = dict()
    with open(filename, mode='r',encoding='UTF-8-sig') as file:
        all8 = csv.reader(file)
        label = ''
        A = []
        for line in all8:
            try:
                labelkey = float(line[0])
                #print(labelkey)
                for item in line:
                    if float(item) > 1999:
                        L = []
                    else:
                        L.append(float(item))
                A.append(L)
            except:
                labelkey = line[0]
                #print(labelkey)
                if labelkey == 'Sex':
                    #initialise array  A
                    A = []
                    label1 = dict_sex[line[1]]
                    label = label1
                elif labelkey == 'Race':
                    label2 = dict_race[line[1]]
                    label = label + '-' + label2
                else:
                    print(label,'_image produced')
                    array = np.array(A)
                    dict_8all[label]=array
                    
    return dict_8all #8 label:ndarray

def ifcontain(Str, label):
    if label in Str:
        return True
    else:
        return False
    

def normaliztion(img):
    cum = 0
    for row in img:
        for ele in row:
            cum = cum + ele
    norimg = img*(N/cum)
    #print(cum)
    
    return norimg, cum

def antinormaliztion(img,cum):
    newcum = 0
    for row in img:
        for ele in row:
            newcum = newcum + ele
    atnimg = img*(cum/newcum)
    return atnimg

def boxfilter(img, r):
    (rows, cols) = img.shape
    imDst = np.zeros_like(img)

    imCum = np.cumsum(img, 0)
    imDst[0 : r+1, :] = imCum[r : 2*r+1, :]
    imDst[r+1 : rows-r, :] = imCum[2*r+1 : rows, :] - imCum[0 : rows-2*r-1, :]
    imDst[rows-r: rows, :] = np.tile(imCum[rows-1, :], [r, 1]) - imCum[rows-2*r-1 : rows-r-1, :]

    imCum = np.cumsum(imDst, 1)
    imDst[:, 0 : r+1] = imCum[:, r : 2*r+1]
    imDst[:, r+1 : cols-r] = imCum[:, 2*r+1 : cols] - imCum[:, 0 : cols-2*r-1]
    imDst[:, cols-r: cols] = np.tile(imCum[:, cols-1], [r, 1]).T - imCum[:, cols-2*r-1 : cols-r-1]

    return imDst


def guidedfilter(I, p, r, eps):
    (rows, cols) = I.shape
    N = boxfilter(np.ones([rows, cols]), r)

    meanI = boxfilter(I, r) / N
    meanP = boxfilter(p, r) / N
    meanIp = boxfilter(I * p, r) / N
    covIp = meanIp - meanI * meanP

    meanII = boxfilter(I * I, r) / N
    varI = meanII - meanI * meanI

    a = covIp / (varI + eps)
    b = meanP - a * meanI

    meanA = boxfilter(a, r) / N
    meanB = boxfilter(b, r) / N

    q = meanA * I + meanB
    return q
'''
if __name__ == '__main__':
    #img = cv2.imread('dasda.png', -1)
    #guide_map = cv2.imread('DR_ck_0001.tiff', -1)
'''
def smooth(guide_map,img, name, n = 1):    
    #maxx=np.max(guide_map)
    #maxx_img = np.max(img)
    out = guidedfilter(guide_map, img, n, 32)
    a=np.max(out)
    #b=np.min(out)
    b = 0
    out = (N*(out-b)/(a-b)).astype(np.uint64)
    #cv2.imwrite(name + ' fillted.png', out)
    '''
    plt.subplot(2, 3, 1), plt.title(name)
    plt.imshow(img, cmap='gray'), plt.axis('off')
    plt.subplot(2, 3, 2), plt.title('guide')
    plt.imshow(guide_map, cmap='gray'), plt.axis('off')
    plt.subplot(2, 3, 3), plt.title(name + ' fillted')
    plt.imshow(out, cmap='gray'), plt.axis('off')
    plt.show()
    '''
    return out

def save_img_as_file(D_img):
    filename = 'SEER17_age_year_filted_1119_sample0_eph30_blank.csv'
    L_out = []
    for img in D_img:
        L_out.append([img])
        for row in D_img[img]:
            L_out.append(row)
    with open(filename, mode = 'w', newline='') as outfile:
        output = csv.writer(outfile, dialect = ("excel"))
        for line in L_out:
            output.writerow(line)
    return

def SEER_main():
    D_in = imread('SEER17_age_year_smooth+piecewise.csv')
    D_out = dict()
    for tup_img in D_in:
        if ifcontain(tup_img, 'male'):
            print('male start')
            #begin the transformation for males
            if  ifcontain(tup_img, 'ALL'):
                print(tup_img)                
                guideimg = D_in[tup_img]
                norguide = normaliztion(guideimg)[0]
                cum_guide = normaliztion(guideimg)[1]
                D_out[tup_img] = guideimg
            else:
                img = D_in[tup_img]
                print(tup_img)
                #black//asian//american
                norimg = normaliztion(img)[0]
                cum_img = normaliztion(img)[1]
                nor_outimg = smooth(norguide,norimg, tup_img)
                outimg = antinormaliztion(nor_outimg, cum_img)
                print(normaliztion(outimg)[1])
                D_out[tup_img] = outimg
                
        else:
            print('female start')
            #begin the transformation for females
            if  ifcontain(tup_img, 'ALL'):
                guideimg = D_in[tup_img]
                norguide = normaliztion(guideimg)[0]
                cum_guide = normaliztion(guideimg)[1]
                D_out[tup_img] = guideimg
                #print(tup_img)
            else:
                img = D_in[tup_img]
                #black//asian//american
                norimg = normaliztion(img)[0]
                cum_img = normaliztion(img)[1]
                nor_outimg = smooth(norguide,norimg, tup_img)
                outimg = antinormaliztion(nor_outimg, cum_img)
                D_out[tup_img] = outimg
                
    save_img_as_file(D_out)
    return
'''--------------------------------FOR SEER DATA USE-----------------------------------------'''

def takeSecond(L):
    return L[1]

def readcountryfile(filename):
    D_dataline = dict()
    D_img = dict()
    
    with open(filename, mode = 'r', encoding = 'UTF-8-sig') as file:
        data = csv.reader(file)
        HEAD = True
        s_year = set()
        for line in data:
            if HEAD:
                HEAD = False
            else:
                D_dataline[line[1],int(line[2])] = [float(line[i+3]) for i in range(len(line)-3)]
                s_year.add(line[2])
        for key_dataline in D_dataline:
            gender, year = key_dataline
            try:
                D_img[gender].append([D_dataline[key_dataline],year])
            except KeyError:
                D_img[gender] = [[D_dataline[key_dataline],year]]
        file.close()
        del(D_dataline)
        
    for gender in D_img:
        L = D_img[gender]
        L.sort(key=takeSecond)
        D_img[gender] = np.transpose(np.array([l[0] for l in L]))
        #print('dim of array is:', np.ndim(D_img[gender]))
    L_year = list(s_year)
    L_year.sort()
    #print(L_year)
    return D_img, int(L_year[0]), int(L_year[-1])

def age_curve(x, a,b,c,d):
    return -a/b*(np.log(np.sqrt(b)*x+1)-np.sqrt(b)*x)/(np.exp((x-c)/d)+1)

def age_fit(lx,ly):
    for candidate in L_initis:
        TryNext = False
        try:
            initis = candidate
            fitParams, fitCovariances = curve_fit(age_curve, lx, ly, initis)
            a,b,c,d = fitParams
            for i in fitParams:
                if i < 0 or i > 100 or c < 20 or d < 1:
                    print('try next initis for age curve fitting')
                    TryNext = True
                else:
                    pass
            if TryNext:
                pass
            else:
                print(fitParams,' is found')
                break
        except RuntimeError:
            print('try next initis for age curve fitting')
            pass

    #print(fitParams)
    return fitParams

def smooth_guidemaker(D_img):
    D_guideimg = dict()
    for gender in D_img:
        A = D_img[gender]
        u,v = np.shape(A)
        #print(u,v)
        ly = np.transpose(np.sum(A,axis=1))/v
        #print(ly)
        L_age = [i for i in range(u)]
        a,b,c,d = age_fit(L_age,ly)
        
        lye = [age_curve(x,a,b,c,d) for x in L_age]
        Ae = np.transpose(np.array([lye for i in range(v)]))
        D_guideimg[gender] = Ae
        #print(np.shape(Ae),Ae)
    return D_guideimg

def piecewise_linear(x, x0, x1, b, k1, k2, k3):
    x0 = 2009
    x1 = 2015
    condlist = [x < x0, (x >= x0) & (x < x1), x >= x1]
    funclist = [lambda x: k1*x + b, lambda x: k1*x + b + k2*(x-x0), lambda x: k1*x + b + k2*(x-x0) + k3*(x - x1)]
    print(k1,k2,k3,b)
    return np.piecewise(x, condlist, funclist)

def linear(x,a,b):
    return a*x+b

def quadratic(x,a,b,c):
    return a*x**2+b*x+c

def cubic(x, a,b,c,d):
    return a*x**3+b*x**2+c*x +d

def quartic(x,a,b,c,d,e):
    return a*x**4+b*x**3+c*x**2 +d*x +e

def quintic(x,a,b,c,d,e,f):
    return a*x**5+b*x**4+c*x**3 +d*x**2 +e*x+f

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

def linears_guidemaker(D_img):
    D_guideimg = dict()
    for gender in D_img:
        A = D_img[gender]
        u,v = np.shape(A)
        #print(u,v)
        ly = np.transpose(np.sum(A,axis=0))/u
        #print(ly)
        L_year = np.array([i for i in range(v)])
        
        L_lye = []
        L_R2 = []
        
        # linear
        L_param, Arr_corr =  curve_fit(linear, L_year, ly)
        lye = [linear(x,L_param[0],L_param[1]) for x in L_year]
        L_lye.append(lye)
        R2 = R_square(ly, lye)
        L_R2.append(R2)
        
        # quadratic
        L_param, Arr_corr =  curve_fit(quadratic, L_year, ly)
        a,b,c = L_param
        lye = [quadratic(x,a,b,c) for x in L_year]
        L_lye.append(lye)
        R2 = R_square(ly, lye)
        L_R2.append(R2)
        
        # cubic
        L_param, Arr_corr =  curve_fit(cubic, L_year, ly)
        a,b,c,d = L_param
        lye = [cubic(x,a,b,c,d) for x in L_year]
        L_lye.append(lye)
        R2 = R_square(ly, lye)
        L_R2.append(R2)
        
        #quartic
        L_param, Arr_corr =  curve_fit(quartic, L_year, ly)
        a,b,c,d,e = L_param
        lye = [quartic(x,a,b,c,d,e) for x in L_year]
        L_lye.append(lye)
        R2 = R_square(ly, lye)
        L_R2.append(R2)
        
        #quintic
        L_param, Arr_corr =  curve_fit(quintic, L_year, ly)
        a,b,c,d,e,f = L_param
        lye = [quintic(x,a,b,c,d,e,f) for x in L_year]
        L_lye.append(lye)
        R2 = R_square(ly, lye)
        L_R2.append(R2)
        
        arr_info = np.array(L_R2)/np.array([1,2,3,4,5])
        #print('R2 are', L_R2)
        #print('info are', arr_info)
        
        maxinfo = np.max(arr_info)
        index = np.where(arr_info == maxinfo)
        
        print('pick the function:',index[0][0])
        lye_best = L_lye[index[0][0]]
        
        Ae = np.array([lye_best for i in range(u)])
        D_guideimg[gender] = Ae
        
        #print(np.shape(Ae), type(Ae))
    #print(D_guideimg)
    return D_guideimg

def linearandsmooth_guidemaker(D_img):
    D_guide_linear = linears_guidemaker(D_img)
    D_guide_smooth = smooth_guidemaker(D_img)
    D_guide_combined = dict()
    
    for gender in D_guide_linear:
        Arr_linears = D_guide_linear[gender]
        # catch a row of array, for each element in row it should be the cummulative value of new collum
        L_cum = Arr_linears[0]
        
        Arr_smooth = D_guide_smooth[gender]
        a,b = np.shape(Arr_smooth)
        # find the cummulative value across age
        L_oldcum = np.sum(Arr_smooth, axis = 0)
        #print(np.ndim(L_cum),np.ndim(L_oldcum))
        
        # renormalize them into value found above
        # form a new 2D array
        Arr_norm = L_cum/L_oldcum * Arr_smooth
        '''
        plt.subplot(2, 3, 1), plt.title('smooth')
        plt.imshow(Arr_smooth, cmap='gray'), plt.axis('off')
        plt.subplot(2, 3, 2), plt.title('linears')
        plt.imshow(Arr_linears, cmap='gray'), plt.axis('off')
        plt.subplot(2, 3, 3), plt.title('combined')
        plt.imshow(Arr_norm, cmap='gray'), plt.axis('off')
        plt.show()
        '''
        # save into dictionary
        D_guide_combined[gender] = Arr_norm
    return D_guide_combined

def readguide(country):
    D_guide={'male': np.array(country),'female': np.array(country) }
    return D_guide

def save_D_filted(D_2Darray, year1, year2, guide):
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
        n,gender,country = key
        
        filename = country + '_' + guide +'.csv'
        
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
            A.sort(key = takeSecond, reverse=True)
            
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

def othercountry_main(L_countryrace, guide='blank'):
    for country in L_countryrace:
        ''' read images, form guide(blank guide) '''
        filename = country + '_rate.csv'
        D_tupimg, year1, year2 = readcountryfile(filename)
        
        """need to save guide file, later on"""
        if guide == 'smooth':
            D_guide = smooth_guidemaker(D_tupimg)
        elif guide == 'linear':
            D_guide = linears_guidemaker(D_tupimg)
        elif guide == 'linearandsmooth':
            D_guide = linearandsmooth_guidemaker(D_tupimg)
        else:
            pass
        
        D_guidefilted = dict() #
        for gender in D_tupimg:
            if guide == 'blank':
                guideimg = np.ones_like(D_tupimg[gender])
                D_guide = {'male':guideimg,'female':guideimg}
            else:
                print('guide fitter:', guide)
            '''try all Ns possible(while loop and try)'''
            n_is_on = True
            n = 1
            # n is the step length of sampling brick
            while n_is_on:
                name = country + ' ' + gender
                img = D_tupimg[gender]
                guideimg = D_guide[gender]
                nor_img = normaliztion(img)[0]              
                cum_img = normaliztion(img)[1]
                
                nor_guideimg = guideimg/cum_img*N
                
                ''' smooth image by 'guide-fillter'''
                try:
                    nor_outimg = smooth(nor_guideimg, nor_img, name, n)
                    # normalize the result
                    outimg = antinormaliztion(nor_outimg, cum_img)
                    D_guidefilted[n,gender,country] = outimg
                    #print('n is:' , n)
                    n = n+1
                except ValueError:
                    n_is_on = False
                    break
        '''save file in std form for each country'''
        save_D_filted(D_guidefilted, year1, year2, guide)
    return
def form_guidances(L_country, L_guide):
    for guide in L_guide:
        othercountry_main(L_country, guide)
   
''' 
filename = 'China_rate.csv'
D_tupimg, year1, year2 = readcountryfile(filename)
#smooth_guidemaker(D_tupimg)
#linears_guidemaker(D_tupimg)
linearandsmooth_guidemaker(D_tupimg)
'''