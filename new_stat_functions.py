#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd 
import scipy.stats as sps
from collections import namedtuple
import seaborn as sns 
    
class Sample:
    def __init__(self, array: np.ndarray, categories: pd.core.series.Series = None, array_before: np.ndarray = None):
        self.array = np.array(array)
        self.series = pd.Series(array)
        self.array_before = np.array(array_before)
        if len(self.array.shape) > 1:
            print('Выборкой должен быть одномерный массив!')
            return None
        
        if categories is None:
            self.categories = pd.Series(['no_category' for i in range(self.count())], name = 'category')
        else:
            self.categories = pd.Series(categories, name = 'category')
            
        if len(self.categories) != self.count():
            print('a и categories должны быть одной длины!')
            return None
        self.categories_n = len(set(self.categories))
        self.df = pd.DataFrame({'category':self.categories, 'value':self.array, 'value_before':self.array_before})
    def __str__(self):
        return f'{self.array}'
    def mean(self):
        return np.mean(self.array)
    def std(self):
        return np.std(self.array)
    def count(self):
        return self.array.shape[0]
    def count_category_sizes(self):
        return self.categories.value_counts().reset_index().rename(columns = {'index':'category','category':'count'})
    def describe(self):
        if self.categories is None:
            return pd.DataFrame({'value':y, 'value_before':x}).describe().reset_index().rename(columns = {'index':'statistic',0:'value'}).set_index('statistic')
        else:
            return self.df.groupby(['category']).describe()
    def hist(self, outliers_perc = [0,100]):
        down_limit, up_limit = np.percentile(self.array, outliers_perc)
        sns.distplot(self.array[np.logical_and(self.array<=up_limit, self.array>=down_limit)], label = 'values')
        plt.title('Distribution density; del outliers by percentiles {}'.format(outliers_perc))
        plt.legend()
        plt.show()
        
class BootstrapTwoSamples:
    def __init__(self, Control:Sample, Test:Sample):
        self.Control = Control
        self.Test = Test
    def max_category_sizes(self):
        return pd.concat([self.Control.count_category_sizes(),self.Test.count_category_sizes()]).groupby('category', as_index = False)['count'].max()
    def boot_samples(self, stat_func = np.mean, before_samples = None, n_samples = 1000):
        max_values_in_categories = self.max_category_sizes()
        for index, row in max_values_in_categories.iterrows():
            boot_len = row['count']
            category = row['category']
            data = CompareTwoSamples(self.Control, self.Test).df
            
            control_category_values = data.query('group=="control" and category==@category')['value']
            test_category_values = data.query('group=="test" and category==@category')['value']
            if before_samples == True:
                control_category_before_values = data.query('group=="control" and category==@category')['value_before']
                test_category_before_values = data.query('group=="test" and category==@category')['value_before']
            
            indices = np.random.randint(0, len(control_category_values), (n_samples, boot_len))
            if index == 0: control_samples = control_category_values.values[indices]
            if index != 0: control_samples = np.concatenate([control_samples, control_category_values.values[indices]], axis = 1)
            if before_samples == True:
                if index == 0: control_before_samples = control_category_before_values.values[indices]
                if index != 0: control_before_samples = np.concatenate([control_before_samples, control_category_before_values.values[indices]], axis = 1)

            indices = np.random.randint(0, len(test_category_values), (n_samples, boot_len))
            if index == 0: test_samples = test_category_values.values[indices]
            if index != 0: test_samples = np.concatenate([test_samples, test_category_values.values[indices]], axis = 1)
            if before_samples == True:
                if index == 0: test_before_samples = test_category_before_values.values[indices]
                if index != 0: test_before_samples = np.concatenate([test_before_samples, test_category_before_values.values[indices]], axis = 1)
        
        control_boot = np.array(list(map(lambda x: stat_func(x), control_samples)))
        test_boot = np.array(list(map(lambda x: stat_func(x), test_samples)))
        if before_samples == True:
            control_before_boot = np.array(list(map(lambda x: stat_func(x), control_before_samples)))
            test_before_boot = np.array(list(map(lambda x: stat_func(x), test_before_samples)))
        
        if before_samples == True:
            return control_boot, control_before_boot, test_boot, test_before_boot
        else:
            return control_boot, test_boot

class CompareTwoSamplesInterface:
    def hist(self):
        raise NotImplemented("you should override this method")
        
class CompareTwoSamples(CompareTwoSamplesInterface):
    ExperimentComparisonResults = namedtuple('ExperimentComparisonResults', ['alpha', 'pvalue', 'effect', 'ci_length', 'left_bound', 'right_bound'])
    def __init__(self, Control:Sample, Test:Sample):
        self.control = Control.array
        self.test = Test.array
        self.control_categories = Control.categories
        self.test_categories = Test.categories
        self.control_before = Control.array_before
        self.test_before = Test.array_before
        self.Control = Control
        self.Test = Test
        self.df = pd.concat([self.Control.df.assign(group = 'control'), self.Test.df.assign(group = 'test')])
    def describe(self):
        if self.control_categories is None or self.test_categories is None:
            return self.df.assign(category = 'all').groupby(['category','group']).describe()
        else:
            return pd.concat([self.df.groupby(['category','group']).describe(), self.df.assign(category = 'all').groupby(['category','group']).describe()]).sort_index()
    def t_test(self, alpha = 0.05, test_type = 'absolute'):
        mean_control = np.mean(self.control)
        mean_test = np.mean(self.test)
        var_mean_control  = np.var(self.control) / len(self.control)
        var_mean_test  = np.var(self.test) / len(self.test)
        
        difference_mean = mean_test - mean_control
        difference_mean_var = var_mean_control + var_mean_test
        
        if test_type == 'absolute':
            difference_distribution = sps.norm(loc=difference_mean, scale=np.sqrt(difference_mean_var))
            left_bound, right_bound = difference_distribution.ppf([alpha/2, 1 - alpha/2])
            ci_length = (right_bound - left_bound)
            pvalue = 2 * min(difference_distribution.cdf(0), difference_distribution.sf(0))
            effect = difference_mean
        if test_type == 'relative':
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
        return CompareTwoSamples.ExperimentComparisonResults(alpha, pvalue, effect, ci_length, left_bound, right_bound)
    def t_test_cuped(self, alpha = 0.05, test_type = 'absolute'):
        if self.control_before is None or self.test_before is None:
            raise ValueError('Не указаны before выборки')
        theta = (np.cov(np.array(self.control), np.array(self.control_before))[0, 1] + np.cov(np.array(self.test), np.array(self.test_before))[0, 1]) /\
                            (np.var(np.array(self.control_before)) + np.var(np.array(self.test_before)))
        
        control_cup = np.array(self.control) - theta * np.array(self.control_before)
        test_cup = np.array(self.test) - theta * np.array(self.test_before)
        
        if test_type == 'absolute':
            mean_control_cup = np.mean(control_cup)
            mean_test_cup = np.mean(test_cup)
            var_mean_control_cup  = np.var(control_cup) / len(control_cup)
            var_mean_test_cup  = np.var(test_cup) / len(test_cup)

            difference_mean_cup = mean_test_cup - mean_control_cup
            difference_mean_var_cup = var_mean_control_cup + var_mean_test_cup
            
            difference_distribution = sps.norm(loc=difference_mean_cup, scale=np.sqrt(difference_mean_var_cup))
            left_bound, right_bound = difference_distribution.ppf([alpha/2, 1 - alpha/2])
            ci_length = (right_bound - left_bound)
            pvalue = 2 * min(difference_distribution.cdf(0), difference_distribution.sf(0))
            effect = difference_mean_cup
            
            return CompareTwoSamples.ExperimentComparisonResults(alpha, pvalue, effect, ci_length, left_bound, right_bound)
        if test_type == 'relative':
            mean_den = np.mean(self.control)
            mean_num = np.mean(test_cup) - np.mean(control_cup)
            var_mean_den  = np.var(self.control) / len(self.control)
            var_mean_num  = np.var(test_cup) / len(test_cup) + np.var(control_cup) / len(control_cup)

            cov = -np.cov(np.array(control_cup), np.array(self.control))[0, 1] / len(self.control)

            relative_mu = mean_num / mean_den
            relative_var = var_mean_num / (mean_den ** 2)  + var_mean_den * ((mean_num ** 2) / (mean_den ** 4))\
                        - 2 * (mean_num / (mean_den ** 3)) * cov

            relative_distribution = sps.norm(loc=relative_mu, scale=np.sqrt(relative_var))
            left_bound, right_bound = relative_distribution.ppf([alpha/2, 1 - alpha/2])

            ci_length = (right_bound - left_bound)
            pvalue = 2 * min(relative_distribution.cdf(0), relative_distribution.sf(0))
            effect = relative_mu
            return CompareTwoSamples.ExperimentComparisonResults(alpha, pvalue, effect, ci_length, left_bound, right_bound)
    def bootstrap_test(self, alpha = 0.05, stat_func = np.mean, test_type = 'absolute', n_samples = 1000, chart = False):
        absolute_func = lambda C, T: T - C
        relative_func = lambda C, T: T / C - 1

        boot_func = absolute_func if test_type == 'absolute' else relative_func

        control_boot, test_boot = BootstrapTwoSamples(self.Control, self.Test).boot_samples(stat_func = stat_func, before_samples = None, n_samples = n_samples)
        boot_data = boot_func(control_boot, test_boot)

        left_bound, right_bound = np.quantile(boot_data, [alpha/2, 1 - alpha/2])
        ci_length = (right_bound - left_bound)
        effect = boot_func(stat_func(self.Control.array), stat_func(self.Test.array))
        pvalue = 2 * min(np.mean(boot_data > 0), np.mean(boot_data < 0))

        if chart == True:
            sns.distplot(control_boot, label = 'control')
            sns.distplot(test_boot, label = 'test')
            plt.title("Distributions of statistic")
            plt.xlabel('statistic')
            plt.ylabel('density')
            plt.legend()
            plt.show()

        return CompareTwoSamples.ExperimentComparisonResults(alpha, pvalue, effect, ci_length, left_bound, right_bound)
    def post_normed_bootstrap_test(self, alpha = 0.05, stat_func = np.mean, test_type='absolute', n_samples = 1000, chart = False):                           
        absolute_func = lambda C, T, C_b, T_b: T - (T_b / C_b) * C
        relative_func = lambda C, T, C_b, T_b: (T / C) / (T_b / C_b) - 1
                           
        boot_func = absolute_func if test_type == 'absolute' else relative_func
        
        control_boot, control_before_boot, test_boot, test_before_boot = BootstrapTwoSamples(self.Control, self.Test).boot_samples(stat_func = stat_func, before_samples = True, n_samples = n_samples)
        boot_data = boot_func(control_boot, test_boot, control_before_boot, test_before_boot)
        
        left_bound, right_bound = np.quantile(boot_data, [alpha/2, 1 - alpha/2])
        ci_length = (right_bound - left_bound)
        effect = boot_func(np.mean(control), np.mean(test), np.mean(control_before), np.mean(test_before))
        pvalue = 2 * min(np.mean(boot_data > 0), np.mean(boot_data < 0))
        
        if chart == True:
            sns.distplot(control_boot, label = 'control')
            sns.distplot(test_boot, label = 'test')
            sns.distplot(control_before_boot, label = 'control_before')
            sns.distplot(test_before_boot, label = 'test_before')
            plt.title("Distributions of statistic")
            plt.xlabel('statistic')
            plt.ylabel('density')
            plt.legend()
            plt.show()
            
        return CompareTwoSamples.ExperimentComparisonResults(alpha, pvalue, effect, ci_length, left_bound, right_bound)
    def hist(self, outliers_perc = [0,100]):
        down_limit, up_limit = np.percentile(np.concatenate([self.Control.array,self.Test.array]), outliers_perc)
        sns.distplot(self.Control.array[np.logical_and(self.Control.array<=up_limit, self.Control.array>=down_limit)], label = 'control')
        sns.distplot(self.Test.array[np.logical_and(self.Test.array<=up_limit, self.Test.array>=down_limit)], label = 'test')
        plt.title('Distribution density; del outliers by percentiles {}'.format(outliers_perc))
        plt.legend()
        plt.show()