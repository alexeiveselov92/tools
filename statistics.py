#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import math as mth
from scipy.stats import f_oneway

# Получаем границы доверительного интервала + выборку полученных средних (результат - словарь)
def get_bootstrap_ci(data, n_samples, func = np.mean, alpha = 0.05):
    '''
    data - column (pd.Series type)
    n_samples - bootstrap n_samples 
    func - statistic, default - np.mean
    alpha - the critical level of significance, default - 0.05
    
    return {'ci':[...], 'scores':[...]}
    '''
    indices = np.random.randint(0, len(data.values), (n_samples, len(data.values)))
    samples = data.values[indices]
    scores = list(map(func, samples))
    ci = np.percentile(scores, [100 * alpha / 2., 100 * (1 - alpha / 2.)]).round(3)
    return {'ci':ci, 'scores':scores}
# Получаем доверительный интервал для доли
def get_fraction_ci(successes, n, conf_level = 0.95):
    '''
    conf_level - 3 variants - 0.9,0.95,0.99
    return - confidence_interval value, list of left and right borders
    document with formula and description - http://math-info.hse.ru/f/2017-18/ps-ms/confint.pdf
    '''
    conf_level_dict = {0.9:1.65,0.95:1.96,0.99:2.58}
    if conf_level in conf_level_dict.keys():
        n_errors = conf_level_dict[conf_level]
        error = np.sqrt((successes/n)*(1-successes/n)/n) 
        ci_value = error*n_errors
        ci = [successes/n - ci_value,successes/n + ci_value]
        return ci_value, ci
    else:
        print('Please, choose one of 3 values of conf_level: 0.9, 0.95 or 0.99')
# стат. тестирование методом бутстрепа для двух выборок с графиком и таблицей результатов
def get_bootstrap(
    data_column_1, # числовые значения первой выборки
    data_column_2, # числовые значения второй выборки
    n_samples = 1000, # количество бутстрэп-подвыборок
    statistic_func = 'mean', # интересующая нас статистика
    percentile_value = 50,
    bootstrap_conf_level = 0.95, # уровень значимости
    chart = True,
    printing = True):
    '''
    statistic_func: 'mean', 'percentile'. If 'percentile', then percentile param default = 50
    return dict {"boot_data": boot_data,"quants": quants,"p_value": p_value,"results": results}
    '''
    boot_len = max([len(data_column_1), len(data_column_2)])
    indices = np.random.randint(0, len(data_column_1), (n_samples, boot_len))
    samples_1 = data_column_1.values[indices]
    indices = np.random.randint(0, len(data_column_2), (n_samples, boot_len))
    samples_2 = data_column_2.values[indices]
    
    if statistic_func=='mean':
        boot_data = list(map(lambda x: np.mean(x), samples_1-samples_2))    
    elif statistic_func=='percentile':
        boot_data = list(map(lambda x: np.percentile(x, q=percentile_value), samples_1-samples_2))    

    pd_boot_data = pd.DataFrame(boot_data)   
    left_quant = (1 - bootstrap_conf_level)/2
    right_quant = 1 - (1 - bootstrap_conf_level) / 2
    quants = pd_boot_data.quantile([left_quant, right_quant])
        
    p_1 = st.norm.cdf(
        x = 0, 
        loc = np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_2 = st.norm.cdf(
        x = 0, 
        loc = -np.mean(boot_data), 
        scale = np.std(boot_data)
    )
    p_value = min(p_1, p_2) * 2
        
    # Визуализация
    if chart == True:
        _, _, bars = plt.hist(pd_boot_data[0], bins = 50)
        for bar in bars:
            if abs(bar.get_x()) <= quants.iloc[0][0] or abs(bar.get_x()) >= quants.iloc[1][0]:
                bar.set_facecolor('red')
            else: 
                bar.set_facecolor('grey')
                bar.set_edgecolor('black')

        plt.style.use('ggplot')
        plt.vlines(quants,ymin=0,ymax=50,linestyle='--')
        plt.xlabel('boot_data')
        plt.ylabel('frequency')
        plt.title("Histogram of boot_data")
        plt.show()
    
    # printing results
    results = pd.DataFrame()
    alpha = 1 - bootstrap_conf_level 
    results.loc[0, 'alpha'] = alpha
    results.loc[0, 'pvalue'] = p_value
    if printing==True:
        print('p-значение: ', p_value)
    if (p_value < alpha):
        results.loc[0, 'the_null_hypothesis'] = 'reject'
        if printing==True:
            print("Отвергаем нулевую гипотезу: различия статистически значимы")
    else:
        results.loc[0, 'the_null_hypothesis'] = 'fail to reject'
        if printing==True:
            print("Не получилось отвергнуть нулевую гипотезу")        
    return {"boot_data": boot_data, 
            "quants": quants, 
            "p_value": p_value,
            "results": results}
# Т-тест
def t_test_ind(x, y, alpha = 0.05, printing = True):
    results = pd.DataFrame()
    pvalue = st.ttest_ind(x, y)[1]
    # saving_results
    results.loc[0, 'alpha'] = alpha 
    results.loc[0, 'pvalue'] = pvalue
    if printing==True:
        print('p-значение: ', pvalue)
    if (pvalue < alpha):
        results.loc[0, 'the_null_hypothesis'] = 'reject'
        if printing==True:
            print("Отвергаем нулевую гипотезу: различия статистически значимы")
    else:
        results.loc[0, 'the_null_hypothesis'] = 'no reject'
        if printing==True:
            print("Не получилось отвергнуть нулевую гипотезу") 
    return results
# Z-тест (для долей)
def z_test(successes_1, successes_2, n1, n2, alpha = .05, printing=True):
    results = pd.DataFrame()
    # пропорция успехов в первой группе:
    p1 = successes_1/n1
    # пропорция успехов во второй группе:
    p2 = successes_2/n2
    # пропорция успехов в комбинированном датасете:
    p_combined = (successes_1 + successes_2) / (n1 + n2)
    # разница пропорций в датасетах
    difference = p1 - p2

    # считаем статистику в ст.отклонениях стандартного нормального распределения
    z_value = difference / mth.sqrt(p_combined * (1 - p_combined) * (1/n1 + 1/n2))
    # задаем стандартное нормальное распределение (среднее 0, ст.отклонение 1)
    distr = st.norm(0, 1) 
    pvalue = (1 - distr.cdf(abs(z_value))) * 2
    # saving_results
    results.loc[0, 'alpha'] = alpha 
    results.loc[0, 'pvalue'] = pvalue
    if printing==True:
        print('p-значение: ', pvalue)
    if (pvalue < alpha):
        results.loc[0, 'the_null_hypothesis'] = 'reject'
        if printing==True: print("Отвергаем нулевую гипотезу: между долями есть значимая разница")
    else:
        results.loc[0, 'the_null_hypothesis'] = 'fail to reject'
        if printing==True: print("Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными") 
    return results
# фильтр Хэмпеля - удаление выбросов (заменяем им на np.nan)
def filter_hampel(x):
    x_copy = x.copy()    
    difference = np.abs(x_copy.median()-x_copy)
    median_abs_deviation = difference.median()
    threshold = 3 * median_abs_deviation
    outlier_idx = difference > threshold
    x_copy[outlier_idx] = np.nan
    return(x_copy)
# ANOVA
def get_anova_results(list_of_arrays, alpha = 0.05, printing=True):
    results = pd.DataFrame()
    stat_anova, p_anova = f_oneway(*list_of_arrays)
    results.loc[0, 'alpha'] = alpha 
    results.loc[0, 'pvalue'] = p_anova
    results.loc[0, 'stat'] = stat_anova
    if (p_anova < alpha):
        results.loc[0, 'the_null_hypothesis'] = 'reject'
        if printing==True: print("Отвергаем нулевую гипотезу: между выборками есть значимая разница")
    else:
        results.loc[0, 'the_null_hypothesis'] = 'fail to reject'
        if printing==True: print("Не получилось отвергнуть нулевую гипотезу, нет оснований считать выборки разными") 
