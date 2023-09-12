# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:00:53 2023
this file aims to concreat all other program modules and form 2 summary files seperately for male and female
@author: Admin
"""
import os
import shutil
import five_year_to_each_year_age as five2age
import event_fillblank as evf
import event_analysis as eva
import event_evaluation as eve
import Guided_filter as gif
import KernelSmooth as KS
import test_sets_summary as setable

'''_____________________________________________________________________________________________'''
'''get estimated rate files, move them into raw file dictionary '''
def pop_file_86(L_country, L_datatype = ['pop']):
    L_age_period = [i*5 for i in range(18)]
    five2age.five_year_2_each_yaer_pop(L_country, L_datatype)
    return
def count_file_86(L_country, L_datatype = ['counts']):
    L_age_period = [i*5 for i in range(18)]
    five2age.five_year_2_each_yaer_counts(L_country, L_datatype)
    return
def rate_file_86(L_country):
    five2age.getrate(L_country)
    return
def move_files_to_rawdir(cor_pop_grid):
    guidancefile = cor_pop_grid + guidancetail
    os.mkdir(guidancefile)
    L_files = os.listdir(cor_pop_grid+popcountail)
    for file in L_files:
        if 'rate' in file:
            shutil.move(cor_pop_grid+popcountail +'/'+ file,guidancefile)
    return
def prepare_rates(L_country, cor_pop_grid):
    pop_file_86(L_country)
    count_file_86(L_country)
    rate_file_86(L_country)
    try:
        move_files_to_rawdir(cor_pop_grid)
    except FileExistsError:
        pass
    return

'''_____________________________________________________________________________________________'''
'''get the smoothed image from KernelSmooth'''
def get_guides(L_country_tails, cor, pop_tail, cor_pop_grid, kernel):
    path = cor_pop_grid + guidancetail + '/'
    L_country = [cor_pop_grid + guidancetail + '/' + cor+'^'+pop_tail+'^'+ country_tail for country_tail in L_country_tails ]
    KS.main(L_country, kernel, path)
    return

def move_guides(cor_pop_grid, guidancetail, kernel):
    L_files = os.listdir(cor_pop_grid+guidancetail)
    for file in L_files:
        if kernel in file:
            try:
                shutil.move(cor_pop_grid+guidancetail +'/'+ file, cor_pop_grid+rawtail)
            except:
                pass
        else:
            pass
    return
def prepare_guides(L_country_tails, guidancetail, cor, pop_tail, cor_pop_grid, kernel):
    get_guides(L_country_tails, cor, pop_tail, cor_pop_grid, kernel)
    move_guides(cor_pop_grid, guidancetail, kernel)
    return
'''_____________________________________________________________________________________________'''
'''
find event package: event fillblank --> event analysis --> event evaluation
event analysis: form guide summary for each country, summary results get into summary file, 'summary of countries'
event evalution: form summary file for each country, and then the overall summary, the overall summary moved into 'summary corr^pop'
'''
def form_country_files(cor_pop_grid, filename = 'raw'):
    try:
        os.mkdir(cor_pop_grid+'/'+filename)
    except:
        pass
    return cor_pop_grid+'/'+filename+'/'

def get_blockvstrend(strdir, L_country_tails, cor, pop_tail, kernel):
    L_country = [strdir + cor+'^'+pop_tail+'^'+ country_tail for country_tail in L_country_tails]
    evf.main(L_country, pop_tail, L_country_tails, strdir, kernel)
    return

def get_guide_summary(strdir, L_country_tails, cor, pop_tail, kernel):
    L_country = [strdir + cor+'^'+pop_tail+'^'+ country_tail for country_tail in L_country_tails]
    eva.main(L_country, kernel ,strdir+'/')
    return

def move_guide_summary(strdir, cor_pop_grid ,filename = 'summary by country'):
    try:
        os.mkdir(cor_pop_grid+'/'+filename)
    except FileExistsError:
        pass
    strdir_summary = cor_pop_grid+'/'+filename
    strdir = strdir.strip('/')
    L_files = os.listdir(strdir)
    for file in L_files:
        if '$guide' in file:
            try:
                shutil.move(strdir+'/'+file, strdir_summary)
            except:
                pass
        else:
            pass
    return strdir_summary

def evaluate_guide_summary(strdir_summary, cor_pop_grid, L_country_tails, cor, pop_tail, filename = 'summary over all'):
    try:
        os.mkdir(cor_pop_grid+'/'+filename)
    except FileExistsError:
        pass
    L_country = [strdir_summary +'/' + cor+'^'+pop_tail+'^'+ country_tail for country_tail in L_country_tails]
    eve.main(L_country, L_country_tails, cor_pop_grid+'/'+filename+'/')
    return

def prepare_SummaryOfImpact(cor_pop_grid, L_country_tails, cor, pop_tail, kernel, filename_eventfinding = 'raw'):
    for country_tail in L_country_tails:
        try:
            shutil.move(cor_pop_grid+'/popcounts/'+cor+'^'+pop_tail+'^'+ country_tail+'_ingredients.csv', cor_pop_grid+'/raw')
        except:
            pass
    strdir_evf = form_country_files(cor_pop_grid, filename = filename_eventfinding)
    get_blockvstrend(strdir_evf, L_country_tails, cor, pop_tail, kernel)
    get_guide_summary(strdir_evf, L_country_tails, cor, pop_tail, kernel)
    strdir_summary = move_guide_summary(strdir_evf, cor_pop_grid)
    evaluate_guide_summary(strdir_summary, cor_pop_grid, L_country_tails, cor, pop_tail)
    return
'''_____________________________________________________________________________________________'''


def prepare_TableOfDescovery(setfilepointer):
    setable.get_settable(setfilepointer)
    return

'''_____________________________________________________________________________________________'''
def flowwork(pop_tail, src_str, kernel):
    for cor in L_cor:
        cor_pop_grid = src_str+'/'+cor
        '''1st'''
        L_country = [cor_pop_grid + popcountail+'/' + cor+'^'+pop_tail+'^'+ country_tail for country_tail in L_country_tails]
        prepare_rates(L_country, cor_pop_grid)
        
        '''2nd'''
        prepare_guides(L_country_tails, guidancetail, cor, pop_tail, cor_pop_grid, kernel)
        
        '''3rd'''
        prepare_SummaryOfImpact(cor_pop_grid, L_country_tails, cor, pop_tail, kernel)
        
    return

'''_____________________________________________________________________________________________'''
popcountail = '/popcounts' #contain pop counts files
rawtail = '/raw' # contain raw and ingredients .csv files
guidancetail = '/guidances'
L_country_tails = ['country'+str(i+1) for i in range(24)]

L_cor = ['cor'+str(round(0.1*(i+1),1)) for i in range(9)]

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
L_pop_tail = [ 'pop5.7','pop6.0','pop6.7','pop7.0','pop7.7','pop8.0','pop8.7','pop9.0' ]
head = 'G:/smoothkernel method verification/YEAR/test_gensis_H/set_3/testset'
tail = '/'
kernel = 'year'
L_setpointer = [ head+str(i+1)+tail for i in range(3) ]
for setpointer in L_setpointer:
    '''
    for pop_tail in L_pop_tail:
        src_str = setpointer + pop_tail
        flowwork(pop_tail, src_str, kernel)
    4th'''
    setpointer = setpointer.strip('/')
    prepare_TableOfDescovery(setpointer)
    
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
