#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
# bootstrap от авито
def bootstrap_avito(control, test, test_type='absolute'):
    # Функция от средних, которую надо посчитать на каждой выборке.
    absolute_func = lambda C, T: T - C
    relative_func = lambda C, T: T / C - 1
    
    boot_func = absolute_func if test_type == 'absolute' else relative_func
    stat_sample = []
    
    batch_sz = 100
    
    # В теории boot_samples_size стоить брать не меньше размера выборки. Но на практике можно и меньше.
    boot_samples_size = len(control)
    for i in range(0, boot_samples_size, batch_sz):
        N_c = len(control)
        N_t = len(test)
        # Выбираем N_c элементов с повторением из текущей выборки. 
        # И чтобы ускорить этот процесс, делаем это сразу batch_sz раз
        # Вместо одной выборки мы получим batch_sz выборок
        control_sample = np.random.choice(control, size=(len(control), batch_sz), replace=True)
        test_sample    = np.random.choice(test, size=(len(test), batch_sz), replace=True)

        C = np.mean(control_sample, axis=0)
        T = np.mean(test_sample, axis=0)
        assert len(T) == batch_sz
        
        # Добавляем в массив посчитанных ранее статистик batch_sz новых значений
        # X в статье – это boot_func(control_sample_mean, test_sample_mean)
        stat_sample += list(boot_func(C, T))

    stat_sample = np.array(stat_sample)
    # Считаем истинный эффект
    effect = boot_func(np.mean(control), np.mean(test))
    left_bound, right_bound = np.quantile(stat_sample, [0.025, 0.975])
    
    ci_length = (right_bound - left_bound)
    # P-value - процент статистик, которые лежат левее или правее 0.
    pvalue = 2 * min(np.mean(stat_sample > 0), np.mean(stat_sample < 0))
    return ExperimentComparisonResults(pvalue, effect, ci_length, left_bound, right_bound)
# post normed bootstrap avito
def post_normed_bootstrap_avito(control, test, control_before, test_before, test_type='absolute'):
    # Функция от средних, которую надо посчитать на каждой выборке.
    absolute_func = lambda C, T, C_b, T_b: T - (T_b / C_b) * C
    relative_func = lambda C, T, C_b, T_b: (T / C) / (T_b / C_b) - 1
    
    boot_func = absolute_func if test_type == 'absolute' else relative_func
    stat_sample = []
    
    batch_sz = 100
    
    #В теории boot_samples_size стоить брать не меньше размера выборки. Но на практике можно и меньше.
    boot_samples_size = len(control)
    for i in range(0, boot_samples_size, batch_sz):
        N_c = len(control)
        N_t = len(test)
        # Надо помнить, что мы семплируем именно юзеров
        # Поэтому, если мы взяли n раз i элемент в выборке control
        # То надо столько же раз взять i элемент в выборке control_before
        # Поэтому будем семплировать индексы
        control_indices = np.arange(N_c)
        test_indices = np.arange(N_t)
        control_indices_sample = np.random.choice(control_indices, size=(len(control), batch_sz), replace=True)
        test_indices_sample    = np.random.choice(test_indices, size=(len(test), batch_sz), replace=True)

        C   = np.mean(control[control_indices_sample], axis=0)
        T   = np.mean(test[test_indices_sample], axis=0)
        C_b = np.mean(control_before[control_indices_sample], axis=0)
        T_b = np.mean(test_before[test_indices_sample], axis=0)
        assert len(T) == batch_sz
        stat_sample += list(boot_func(C, T, C_b, T_b))

    stat_sample = np.array(stat_sample)
    # считаем истинный эффект
    effect = boot_func(np.mean(control), np.mean(test), np.mean(control_before), np.mean(test_before))
    left_bound, right_bound = np.quantile(stat_sample, [0.025, 0.975])
    
    ci_length = (right_bound - left_bound)
    # P-value - процент статистик, которые лежат левее или правее 0.
    pvalue = 2 * min(np.mean(stat_sample > 0), np.mean(stat_sample < 0))
    return ExperimentComparisonResults(pvalue, effect, ci_length, left_bound, right_bound)
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
        bootstrapped_value_1 = np.mean(boot_mean_data_1)  
        bootstrapped_value_2 = np.mean(boot_mean_data_2)  
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
    results.loc[0, 'dif_left_bound'] = np.quantile(boot_data, alpha/2)
    results.loc[0, 'dif_right_bound'] = np.quantile(boot_data, 1-alpha/2)
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
# tukey hsd test https://www.statology.org/tukey-test-python/ - надо сделать функцию
# применяем после ANOVA чтобы провести попарное сравнение и найти отличающиеся пары

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
# criterion functions
def absolute_ttest(control, test, alpha = 0.05):
    import scipy.stats as sps
    from collections import namedtuple
    ExperimentComparisonResults = namedtuple('ExperimentComparisonResults', 
                                        ['pvalue', 'effect', 'ci_length', 'left_bound', 'right_bound'])
    mean_control = np.mean(control)
    mean_test = np.mean(test)
    var_mean_control  = np.var(control) / len(control)
    var_mean_test  = np.var(test) / len(test)
    
    difference_mean = mean_test - mean_control
    difference_mean_var = var_mean_control + var_mean_test
    difference_distribution = sps.norm(loc=difference_mean, scale=np.sqrt(difference_mean_var))

    left_bound, right_bound = difference_distribution.ppf([alpha/2, 1 - alpha/2])
    ci_length = (right_bound - left_bound)
    pvalue = 2 * min(difference_distribution.cdf(0), difference_distribution.sf(0))
    effect = difference_mean
    return ExperimentComparisonResults(pvalue, effect, ci_length, left_bound, right_bound)
def relative_ttest(control, test, alpha = 0.05):
    import scipy.stats as sps
    from collections import namedtuple
    ExperimentComparisonResults = namedtuple('ExperimentComparisonResults', 
                                        ['pvalue', 'effect', 'ci_length', 'left_bound', 'right_bound'])
    mean_control = np.mean(control)
    var_mean_control  = np.var(control) / len(control)

    difference_mean = np.mean(test) - mean_control
    difference_mean_var  = np.var(test) / len(test) + var_mean_control
    
    covariance = -var_mean_control

    relative_mu = difference_mean / mean_control
    relative_var = difference_mean_var / (mean_control ** 2) \
                    + var_mean_control * ((difference_mean ** 2) / (mean_control ** 4))\
                    - 2 * (difference_mean / (mean_control ** 3)) * covariance
    relative_distribution = sps.norm(loc=relative_mu, scale=np.sqrt(relative_var))
    left_bound, right_bound = relative_distribution.ppf([alpha/2, 1 - alpha/2])
    
    ci_length = (right_bound - left_bound)
    pvalue = 2 * min(relative_distribution.cdf(0), relative_distribution.sf(0))
    effect = relative_mu
    return ExperimentComparisonResults(pvalue, effect, ci_length, left_bound, right_bound)
# relative bootstrap
def relative_bootstrap(control, test, alpha = 0.05, n_samples = 1000):
    import scipy.stats as sps
    from collections import namedtuple
    ExperimentComparisonResults = namedtuple('ExperimentComparisonResults', 
                                        ['pvalue', 'effect', 'ci_length', 'left_bound', 'right_bound'])
    boot_len = max([len(control), len(test)])
    indices = np.random.randint(0, len(control), (n_samples, boot_len))
    if type(control) == np.ndarray: control_samples = control[indices]
    if type(control) == pd.core.series.Series: control_samples = control.values[indices]
    indices = np.random.randint(0, len(test), (n_samples, boot_len))
    if type(control) == np.ndarray: test_samples = test[indices]
    if type(control) == pd.core.series.Series: test_samples = test.values[indices]
    
    control_data = list(map(lambda x: np.mean(x), control_samples))
    test_data = list(map(lambda x: np.mean(x), test_samples))
    boot_data = (np.array(test_data) - np.array(control_data)) / np.array(control_data)
  
    left_quant = alpha/2
    right_quant = 1 - alpha/2
    left_bound, right_bound = np.quantile(boot_data, [left_quant, right_quant])
    ci_length = (right_bound - left_bound)
    effect = np.mean(test) - np.mean(control)
    relative_distribution = sps.norm(loc=np.mean(boot_data), scale=np.std(boot_data))
    pvalue = 2 * min(relative_distribution.cdf(0), relative_distribution.sf(0))
    
    return ExperimentComparisonResults(pvalue, effect, ci_length, left_bound, right_bound)
# get cuped samples
def cuped_samples(control, test, control_before, test_before):
    theta = (np.cov(control, control_before)[0, 1] + np.cov(test, test_before)[0, 1]) /\
                (np.var(control_before) + np.var(test_before))

    control_cup = control - theta * control_before
    test_cup = test - theta * test_before
    return control_cup, test_cup
# checking functions
def checking_criterion(data, values_column, one_group_size = None, difference_pct = 0, alpha = 0.05, criterion = 'ttest', bootstrap_n_samples = 1000, stratified = False, categories_column = None, N = 20000, print_results = True):
    '''
    one_group_size: if our value is float, then we will take a fraction of our data, and if our value is int, then we will take this sample size
    difference_pct: if 0, then AA, else AB
    criterion: 'ttest' or 'bootstrap'. but if you choose bootstrap, then determine the number of bootstrap subsamples - bootstrap_n_samples
    '''
    
    # import
    from tqdm.notebook import tqdm as tqdm_notebook
    from collections import namedtuple
    from statsmodels.stats.proportion import proportion_confint
        
    # variables
    if one_group_size == None: one_group_size = 0.5    
    results = pd.DataFrame()
    
    # preparing for stratification
    if stratified == True:
        max_values_in_categories = data[categories_column].value_counts().rename('count').reset_index().rename(columns = {'index':'category'}).groupby('category')['count'].max().reset_index()
        max_values_in_categories['weight'] = max_values_in_categories['count'] / max_values_in_categories['count'].sum()
        categories = data[categories_column].unique()
        categories_sizes = {row['category']:row['count'] for _, row in max_values_in_categories.iterrows()}
        categories_dataframes = {row['category']:data.query(f'''{categories_column}=="{row['category']}"''') for _, row in max_values_in_categories.iterrows()}
        if type(one_group_size) == float:
            categories_samples_sizes = {row['category']:int(row['count'] * one_group_size) for _, row in max_values_in_categories.iterrows()}
        elif type(one_group_size) == int:
            categories_samples_sizes = {row['category']:int(row['weight'] * one_group_size) for _, row in max_values_in_categories.iterrows()}
        else:
            print('one_group_size variable must be an float or int value!')
            return None
        
    # create out counters 
    count_dif = 0
    count_level = 0
    
    # start N iterations
    for i in tqdm_notebook(range(N)):
        # get control and test samples
        if stratified == True:
            control = np.array([])
            test = np.array([])
            for category in categories: 
                category_size = categories_sizes[category]
                category_sample_size = categories_samples_sizes[category]

                indices = np.random.randint(0, category_size, category_sample_size)
                control = np.concatenate([control, categories_dataframes[category][values_column].values[indices]], axis = 0)
                indices = np.random.randint(0, category_size, category_sample_size)
                test = np.concatenate([test, categories_dataframes[category][values_column].values[indices] * (1 + (difference_pct / 100))], axis = 0)
        else:
            if type(one_group_size) == float:
                boot_len = int(len(data) * one_group_size)
            elif type(one_group_size) == int:
                boot_len = one_group_size
            indices = np.random.randint(0, len(data), boot_len)
            control = data[values_column].values[indices]
            indices = np.random.randint(0, len(data), boot_len)
            test = data[values_column].values[indices] * (1 + (difference_pct / 100))
        
        # using criterion and counting cases detected difference
        if criterion == 'ttest':
            _, _, _, left_bound, right_bound = relative_ttest(control, test, alpha)
        if criterion == 'bootstrap':
            _, _, _, left_bound, right_bound = relative_bootstrap(control, test, alpha, n_samples = bootstrap_n_samples)
        if left_bound > 0 or right_bound < 0:
            count_dif += 1
        if left_bound > difference_pct / 100 or right_bound < difference_pct / 100:
            count_level += 1
    
    # create results
    left_real_level, right_real_level = proportion_confint(count = count_level, nobs = N, alpha=0.05, method='wilson')
    results.loc[0, 'criterion'] = criterion
    if difference_pct == 0:
        results.loc[0, 'type'] = 'AA'
    else:
        results.loc[0, 'type'] = 'AB'
    if type(one_group_size) == float:
        results.loc[0, 'one_group_size'] = one_group_size * len(data)
    elif type(one_group_size) == int:
        results.loc[0, 'one_group_size'] = one_group_size
    results.loc[0, 'difference_pct'] = difference_pct
    results.loc[0, 'alpha'] = alpha
    results.loc[0, 'stratified'] = stratified
    results.loc[0, 'iterations'] = N
    results.loc[0, 'dif_detected/power'] = round(count_dif / N, 4)
    results.loc[0, 'real_level'] = round(count_level / N, 4)
    results.loc[0, 'left_real_level'] = left_real_level
    results.loc[0, 'right_real_level'] = right_real_level
    if print_results == True: 
        print('difference_pct in ci (level): {:.2%}('.format(round(count_level / N, 4)) + "[{:.2%}, {:.2%}]); ".format(round(left_real_level, 4),round(right_real_level, 4)) + 'dif_detected/power: {:.2%};'.format(round(count_dif / N, 4)))
    return results  
def checking_criterion_iterable(data, values_column, one_group_size = None, difference_pct = 0, alpha = 0.05, criterion = 'ttest', bootstrap_n_samples = 1000, stratified = False, categories_column = None, N = 20000, print_results = True):
    vars_dict = {
        'one_group_size':one_group_size,
        'difference_pct':difference_pct,
        'alpha':alpha,
        'criterion':criterion,
        'bootstrap_n_samples':bootstrap_n_samples,
        'stratified':stratified,
        'categories_column':categories_column,
        'N':N
    }
    iterable_vars_dict = {name:vars_dict[name] for name in vars_dict if type(vars_dict[name])==list}
    if np.mean([len(iterable_vars_dict[name]) for name in iterable_vars_dict]) != [len(iterable_vars_dict[name]) for name in iterable_vars_dict][0]:
        print('Error! vars lists must be the same length!')
        return None
    results = pd.DataFrame()
    for _, row in pd.DataFrame(vars_dict).iterrows():
        print('\n')
        print(row.to_dict())
        results = pd.concat([results, checking_criterion(data, values_column, **row.to_dict(), print_results = print_results)])
    return results