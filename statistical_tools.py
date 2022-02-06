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
def bootstrap_test(
    values_1, # числовые значения первой выборки
    values_2, # числовые значения второй выборки
    n_samples = 1000, # количество бутстрэп-подвыборок
    statistic_func = 'mean', # интересующая нас статистика
    percentile_value = 50,
    bootstrap_conf_level = 0.95, # уровень значимости
    stratified = None,
    categories_1 = None,
    categories_2 = None,
    chart = True,
    chart_type = 'compare',
    print_results = True,
    returns = 'results'):
    '''
    statistic_func: 'mean', 'percentile'. If 'percentile', then percentile param default = 50
    for using percentile func watch video with shifting estimates: https://www.youtube.com/watch?v=p_5YzShN4sg
    Смещение выборки для решения проблемы высокого bias (для медианы, квантиля и т.д.):
    1. объединяем обе выборки и считаем в них параметр (среднее, медиану, дисперсию, кванитиль и т.д.) и записываем
    2. считаем дельту между двумя параметрами и записываем 
    3. сдвигаем для выборок статистику таким образом, чтобы параметр был такой же, как и у объединенной группы 
    4. бутстрепим в обеих выборках и считаем разницу 
    5. считаем случаи, когда разница бутстреп-статистик была равна или больше ранее записанной дельты. это и будет p-value

    returns: str - 'boot_data', 'quants', 'p_value', 'results'
    '''
    if stratified == True:
        if type(categories_1) not in [pd.core.series.Series, list, np.ndarray] or type(categories_2) not in [pd.core.series.Series, list, np.ndarray]:
            print('Error! For stratified bootstrap you need to use categories_1 and categories_2 as pd.Series, list of np.array of categorial values')
            return
        else:
            data = pd.DataFrame()
            for values_n in [1,2]:
                if values_n == 1: df = pd.DataFrame({'values':values_1, 'categories':categories_1})
                if values_n == 2: df = pd.DataFrame({'values':values_2, 'categories':categories_2})
                df['values_n'] = values_n
                data = pd.concat([data, df])
            max_values_in_categories = pd.concat([
                categories_1.value_counts().rename('count'),
                categories_2.value_counts().rename('count')
            ]).reset_index().rename(columns = {'index':'category'}).groupby('category')['count'].max().reset_index()
            for index, row in max_values_in_categories.iterrows():
                boot_len = row['count']
                category = row['category']
                cat_values_1 = data.query('values_n==1 and categories==@category')['values']
                cat_values_2 = data.query('values_n==2 and categories==@category')['values']

                indices = np.random.randint(0, len(cat_values_1), (n_samples, boot_len))
                if index == 0: samples_1 = cat_values_1.values[indices]
                if index != 0: samples_1 = np.concatenate([samples_1, cat_values_1.values[indices]], axis = 1)

                indices = np.random.randint(0, len(cat_values_2), (n_samples, boot_len))
                if index == 0: samples_2 = cat_values_2.values[indices]
                if index != 0: samples_2 = np.concatenate([samples_2, cat_values_2.values[indices]], axis = 1)
    else:           
        boot_len = max([len(values_1), len(values_2)])
        indices = np.random.randint(0, len(values_1), (n_samples, boot_len))
        samples_1 = values_1.values[indices]
        indices = np.random.randint(0, len(values_2), (n_samples, boot_len))
        samples_2 = values_2.values[indices]
    
    if statistic_func=='mean':
        boot_data = list(map(lambda x: np.mean(x), samples_1-samples_2))
        boot_mean_data_1 = list(map(lambda x: np.mean(x), samples_1))
        boot_mean_data_2 = list(map(lambda x: np.mean(x), samples_2))
        bootstrapped_value_1 = np.mean(boot_mean_data_1)
        bootstrapped_value_2 = np.mean(boot_mean_data_2)
    else:
        boot_data = list(map(lambda x: np.percentile(x, q=percentile_value), samples_1-samples_2))
        boot_mean_data_1 = list(map(lambda x: np.percentile(x, q=percentile_value), samples_1))
        boot_mean_data_2 = list(map(lambda x: np.percentile(x, q=percentile_value), samples_2))
        bootstrapped_value_1 = np.mean(boot_mean_data_1, q = percentile_value)  
        bootstrapped_value_2 = np.mean(boot_mean_data_2, q = percentile_value)  
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
        if chart_type == 'all' or chart_type == 'compare':
            sns.distplot(boot_mean_data_1, label = 'distribution_1')
            sns.distplot(boot_mean_data_2, label = 'distribution_2')
            plt.title("Histograms of ")
            plt.xlabel('statistic')
            plt.ylabel('frequency')
            plt.legend()
            plt.show()
            
        if chart_type == 'all' or chart_type == 'differents':
            _, _, bars = plt.hist(pd_boot_data[0], bins = 50)
            for bar in bars:
                if abs(bar.get_x()) <= quants.iloc[0][0] or abs(bar.get_x()) >= quants.iloc[1][0]:
                    bar.set_facecolor('red')
                else: 
                    bar.set_facecolor('grey')
                    bar.set_edgecolor('black')
            plt.vlines(quants,ymin=0,ymax=50,linestyle='--', color = 'blue')
            plt.xlabel('boot_data')
            plt.ylabel('frequency')
            plt.title("Histogram of boot_data")
            plt.show()
    
    # printing results
    results = pd.DataFrame()
    alpha = 1 - bootstrap_conf_level 
    results.loc[0, 'alpha'] = alpha
    results.loc[0, 'bootstrapped_value_1'] = bootstrapped_value_1
    results.loc[0, 'bootstrapped_value_2'] = bootstrapped_value_2
    results.loc[0, 'pvalue'] = p_value
    if print_results==True:
        print('p-значение: ', p_value)
    if (p_value < alpha):
        results.loc[0, 'the_null_hypothesis'] = 'reject'
        if print_results==True:
            print("Отвергаем нулевую гипотезу: различия статистически значимы")
    else:
        results.loc[0, 'the_null_hypothesis'] = 'fail to reject'
        if print_results==True:
            print("Не получилось отвергнуть нулевую гипотезу")        
    return {"boot_data": boot_data, 
            "quants": quants, 
            "p_value": p_value,
            "results": results}[returns]
# Т-тест
def t_test(x, y, alpha = 0.05, alternative='two-sided', print_results = True):
    '''
    alternative: string - 'two-sided', 'less', 'greater'
    '''
    results = pd.DataFrame()
    pvalue = st.ttest_ind(x, y)[1]
    # saving_results
    results.loc[0, 'alpha'] = alpha 
    results.loc[0, 'pvalue'] = pvalue
    if print_results==True:
        print('p-значение: ', pvalue)
    if (pvalue < alpha):
        results.loc[0, 'the_null_hypothesis'] = 'reject'
        if print_results==True:
            print("Отвергаем нулевую гипотезу: различия статистически значимы")
    else:
        results.loc[0, 'the_null_hypothesis'] = 'no reject'
        if print_results==True:
            print("Не получилось отвергнуть нулевую гипотезу") 
    return results
# тест Манна - Уитни
def mannwhitneyu_test(x, y, alpha = 0.05, alternative = 'two-sided', print_results = True):
    '''
    alternative: string - 'two-sided', 'less', 'greater'
    '''
    results = pd.DataFrame()
    pvalue = st.mannwhitneyu(x, y)[1]
    # saving_results
    results.loc[0, 'alpha'] = alpha 
    results.loc[0, 'pvalue'] = pvalue
    if print_results==True:
        print('p-значение: ', pvalue)
    if (pvalue < alpha):
        results.loc[0, 'the_null_hypothesis'] = 'reject'
        if print_results==True:
            print("Отвергаем нулевую гипотезу: различия статистически значимы")
    else:
        results.loc[0, 'the_null_hypothesis'] = 'no reject'
        if print_results==True:
            print("Не получилось отвергнуть нулевую гипотезу") 
    return results
# Z-тест (для долей)
def z_test(successes_1, successes_2, n1, n2, alpha = .05, print_results = True):
    results_df = pd.DataFrame()
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
    results_df.loc[0, 'alpha'] = alpha 
    results_df.loc[0, 'pvalue'] = pvalue
    if print_results==True:
        print('p-значение: ', pvalue)
    if (pvalue < alpha):
        results_df.loc[0, 'the_null_hypothesis'] = 'reject'
        if print_results==True: print("Отвергаем нулевую гипотезу: между долями есть значимая разница")
    else:
        results_df.loc[0, 'the_null_hypothesis'] = 'fail to reject'
        if print_results==True: print("Не получилось отвергнуть нулевую гипотезу, нет оснований считать доли разными") 
    return results_df
# ANOVA
def anova_test(list_of_arrays, alpha = 0.05, print_results = True):
    results = pd.DataFrame()
    stat_anova, p_anova = f_oneway(*list_of_arrays)
    results.loc[0, 'alpha'] = alpha 
    results.loc[0, 'pvalue'] = p_anova
    results.loc[0, 'stat'] = stat_anova
    if print_results==True:
        print('p-значение: ', p_anova)
    if (p_anova < alpha):
        results.loc[0, 'the_null_hypothesis'] = 'reject'
        if print_results==True: print("Отвергаем нулевую гипотезу: между выборками есть значимая разница")
    else:
        results.loc[0, 'the_null_hypothesis'] = 'fail to reject'
        if print_results==True: print("Не получилось отвергнуть нулевую гипотезу, нет оснований считать выборки разными") 
    return results
# binominal test   
def binom_test(successes, n, p = 0.5, alternative = 'two-sided', alpha = 0.05, print_results = True):
    '''
    alternative: string - 'two-sided', 'greater', 'less'
    '''
    results_df = pd.DataFrame()
    pvalue = st.binom_test(successes, n, p, alternative = alternative)
    # saving_results
    results_df.loc[0, 'alpha'] = alpha 
    results_df.loc[0, 'pvalue'] = pvalue
    if print_results == True:
        print('p-значение: ', pvalue)
    if (pvalue < alpha):
        results_df.loc[0, 'the_null_hypothesis'] = 'reject'
        if print_results == True: print("Отвергаем нулевую гипотезу")
    else:
        results_df.loc[0, 'the_null_hypothesis'] = 'fail to reject'
        if print_results == True: print("Не получилось отвергнуть нулевую гипотезу") 
    return results_df
# Хи квадрат - критерий согласия Пирсона https://www.coursera.org/lecture/stats-for-data-analysis/kritierii-soghlasiia-pirsona-khi-kvadrat-Z1Noq
def chisquare_test(observed_frequences, expected_frequences, ddof = 0, alpha = 0.05, print_results = True):
    results_df = pd.DataFrame()
    pvalue = st.chisquare(observed_frequences, expected_frequences, ddof = ddof).pvalue
    # saving_results
    results_df.loc[0, 'alpha'] = alpha 
    results_df.loc[0, 'pvalue'] = pvalue
    if print_results == True:
        print('p-значение: ', pvalue)
    if (pvalue < alpha):
        results_df.loc[0, 'the_null_hypothesis'] = 'reject'
        if print_results == True: print("Отвергаем нулевую гипотезу")
    else:
        results_df.loc[0, 'the_null_hypothesis'] = 'fail to reject'
        if print_results == True: print("Не получилось отвергнуть нулевую гипотезу") 
    return results_df
# фильтр Хэмпеля - удаление выбросов (заменяем их на np.nan)
def filter_hampel(x):
    x_copy = x.copy()    
    difference = np.abs(x_copy.median()-x_copy)
    median_abs_deviation = difference.median()
    threshold = 3 * median_abs_deviation
    outlier_idx = difference > threshold
    x_copy[outlier_idx] = np.nan
    return(x_copy)
# расчет размера выборки для провычисления пропорций категориальных данных (дискретный случай)
def get_sample_size_proportion(expected_proportion, min_detectable_effect, conf_level = 0.95, print_results = True):
    '''
    conf_level - 3 variants - 0.9,0.95,0.99
    '''
    conf_level_dict = {0.9:1.65,0.95:1.96,0.99:2.58}
    if conf_level in conf_level_dict.keys():
        min_sample_size = int((expected_proportion * (1 - expected_proportion) * conf_level_dict[conf_level]**2) / min_detectable_effect**2)
        if print_results == True: print(f'Минимальный размер выборки составляет {min_sample_size}.')
        return min_sample_size
    else:
        print('Please, choose one of 3 values of conf_level: 0.9, 0.95 or 0.99')
# расчет размера выборки для вычисления среднего значения (непрерывный случай)
def get_sample_size_mean_value(std_dev, min_detectable_effect, conf_level = 0.95, print_results = True):
    '''
    std_dev - standard deviation or (max - min) / 5 
    conf_level - 3 variants - 0.9,0.95,0.99
    '''
    conf_level_dict = {0.9:1.65,0.95:1.96,0.99:2.58}
    if conf_level in conf_level_dict.keys():
        min_sample_size = int((std_dev**2 * conf_level_dict[conf_level]**2) / min_detectable_effect**2)
        if print_results == True: print(f'Минимальный размер выборки составляет {min_sample_size}.')
        return min_sample_size
    else:
        print('Please, choose one of 3 values of conf_level: 0.9, 0.95 or 0.99') 
# расчет размера выборки для вычисления среднего значения (непрерывный случай) - версия два 
def get_sample_size_mean_value_2(values, min_detectable_effect_pct = 10, power = 0.8, alpha = 0.05):
    from statsmodels.stats.power import TTestIndPower
    std = np.std(values)
    mean = np.mean(values)
    effect = mean * (min_detectable_effect_pct / 100) / std
    analysis = TTestIndPower()
    size = int(analysis.solve_power(effect, power = power, alpha = alpha))
    results = pd.DataFrame()
    results.loc[0, 'alpha'] = alpha
    results.loc[0, 'power'] = power
    results.loc[0, 'std'] = std
    results.loc[0, 'mean'] = mean
    results.loc[0, 'min_detectable_effect_pct'] = min_detectable_effect_pct
    results.loc[0, 'min_size'] = size
    return results