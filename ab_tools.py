#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd 
import scipy.stats as sps
from collections import namedtuple
import seaborn as sns 
from tqdm.notebook import tqdm as tqdm_notebook
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
import math as mth
from scipy.stats import f_oneway
from itertools import combinations_with_replacement, permutations, combinations
import itertools
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import scipy.stats as stats

class GetSizeForSample:
    def needed_size(self, alpha=0.05, power=0.8, diff_type='pct', diff=10):
        ''' 
        :param diff_type: может быть равен 'pct' или 'absolute' 
        :param diff: если разница в процентах, то нужно число, напр. 5% или 10% 
        '''
        mean_1=self.mean()

        if diff_type=='absolute':
            mean_2=mean_1+diff
        elif diff_type=='pct':
            mean_2=mean_1*(1+diff/100)
        else:
            raise ValueError('diff_type is incorrect ("pct" or "absolute")')

        std=self.std()
        size=int(((2*std**2)*(stats.norm.ppf(1-alpha/2)+stats.norm.ppf(power))**2)/abs(mean_1-mean_2)**2)

        results = pd.DataFrame()
        results.loc[0, 'alpha'] = alpha
        results.loc[0, 'power'] = power
        results.loc[0, 'std'] = std
        results.loc[0, 'mean'] = mean_1
        results.loc[0, 'diff_type'] = diff_type
        results.loc[0, 'diff'] = diff
        results.loc[0, 'min_size'] = size

        return results
    def min_effect(self, alpha=0.05, power=0.8):

        mean_1=self.mean()
        std=self.std()
        size=len(df)
        diff=(((2*std**2)*(stats.norm.ppf(1-alpha/2)+stats.norm.ppf(power))**2)/size)**0.5
        mean_2=mean_1+diff
        effect=int((mean_2/mean_1-1)*100)

        results = pd.DataFrame()
        results.loc[0, 'alpha'] = alpha
        results.loc[0, 'power'] = power
        results.loc[0, 'std'] = std
        results.loc[0, 'mean'] = mean_1
        results.loc[0, 'mean_test'] = mean_2
        results.loc[0, 'min_effect'] = str(effect)+'%'

        return results    
class BootstrapOneSample:
    def boot_samples(self, before_samples = None, n_samples = 1000, stratify = False, size = None):
        if size is None:
            size = self.count()
        if stratify == True:
            categories_iterable_df = self.count_category_sizes()
        else:
            categories_iterable_df = pd.DataFrame({'category':['no_category'], 'count':[self.count()]})
        categories_iterable_df['weight'] = categories_iterable_df['count'] / categories_iterable_df['count'].sum()
        for index, row in categories_iterable_df.iterrows():
            category_size = row['count']
            category_weight = row['weight']
            category = row['category']
            if stratify == True:
                data = self.df()
            else:
                data = self.df().assign(category = 'no_category')
            
            category_values = data.query('category==@category')['value']
            if before_samples == True:
                category_before_values = data.query('category==@category')['value_before']
            
            if type(size) == float:
                boot_len = int(category_size * size)
            elif type(size) == int:
                boot_len = int(category_weight * size)
            
            indices = np.random.randint(0, len(category_values), (n_samples, boot_len))
            if index == 0: samples = category_values.values[indices]
            if index != 0: samples = np.concatenate([samples, category_values.values[indices]], axis = 1)
            if before_samples == True:
                if index == 0: samples_before = category_before_values.values[indices]
                if index != 0: samples_before = np.concatenate([samples_before, category_before_values.values[indices]], axis = 1)
                    
        if before_samples == True:
            return samples, samples_before
        else:
            return samples
    def boot_results(self, stat_func = np.mean, before_samples = None, n_samples = 1000, stratify = False):
        if before_samples == True:
            samples, samples_before = self.boot_samples(before_samples = before_samples, n_samples = n_samples, stratify = stratify)
        else:
            samples = self.boot_samples(before_samples = before_samples, n_samples = n_samples, stratify = stratify)
        
        boot = np.array(list(map(lambda x: stat_func(x), samples)))
        if before_samples == True:
            before_boot = np.array(list(map(lambda x: stat_func(x), samples_before)))
        
        if before_samples == True:
            return boot, before_boot
        else:
            return boot
    def bootstrap_ci(self, alpha = 0.05, stat_func = np.mean, n_samples = 1000, stratify = False):
        return np.percentile(self.boot_results(stat_func = stat_func, n_samples = n_samples, stratify = stratify), [alpha/2, 100-alpha/2])
class BootstrapTwoSamples:
    def max_category_sizes(self):
        return pd.concat([self.Control.count_category_sizes(),self.Test.count_category_sizes()]).groupby('category', as_index = False)['count'].max()
    def boot_samples(self, before_samples = None, n_samples = 1000, stratify = False):
        if stratify == True:
            categories_iterable_df = self.max_category_sizes()
        else:
            categories_iterable_df = pd.DataFrame({'category':['no_category'], 'count':[max(self.Control.count(), self.Test.count())]})
        for index, row in categories_iterable_df.iterrows():
            boot_len = row['count']
            category = row['category']
            if stratify == True:
                data = CompareTwoSamples(self.Control, self.Test).df()
            else:
                data = CompareTwoSamples(self.Control, self.Test).df().assign(category = 'no_category')
            
            control_category_values = data.query(f'group=="{self.control_name}" and category==@category')['value']
            test_category_values = data.query(f'group=="{self.test_name}" and category==@category')['value']
            if before_samples == True:
                control_category_before_values = data.query(f'group=="{self.control_name}" and category==@category')['value_before']
                test_category_before_values = data.query(f'group=="{self.test_name}" and category==@category')['value_before']
            
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
                    
        if before_samples == True:
            return control_samples, control_before_samples, test_samples, test_before_samples
        else:
            return control_samples, test_samples
    def boot_results(self, stat_func = np.mean, before_samples = None, n_samples = 1000, stratify = False):
        if before_samples == True:
            control_samples, control_before_samples, test_samples, test_before_samples = self.boot_samples(before_samples = before_samples, n_samples = n_samples, stratify = stratify)
        else:
            control_samples, test_samples = self.boot_samples(before_samples = before_samples, n_samples = n_samples, stratify = stratify)
        
        control_boot = np.array(list(map(lambda x: stat_func(x), control_samples)))
        test_boot = np.array(list(map(lambda x: stat_func(x), test_samples)))
        if before_samples == True:
            control_before_boot = np.array(list(map(lambda x: stat_func(x), control_before_samples)))
            test_before_boot = np.array(list(map(lambda x: stat_func(x), test_before_samples)))
        
        if before_samples == True:
            return control_boot, control_before_boot, test_boot, test_before_boot
        else:
            return control_boot, test_boot  
    def bootstrap_ci(self, alpha = 0.05, stat_func = np.mean, boot_func = lambda C,T:T-C, n_samples = 1000, stratify = False):
        control_boot, test_boot = self.boot_results(stat_func = stat_func, before_samples = False, n_samples = n_samples, stratify = stratify)
        boot_data = boot_func(control_boot, test_boot)
        return np.percentile(boot_data, [alpha/2, 100-alpha/2])
    def bootstrap_boxplots(self, alpha = 0.05, stat_func = np.mean, boot_func = lambda C,T:T-C, n_samples = 1000, stratify = False):
        control_boot, test_boot = self.boot_results(stat_func = stat_func, before_samples = False, n_samples = n_samples, stratify = stratify)       
        fig = plt.figure(figsize =(16, 9))
        ax = fig.add_subplot(111)
        bp = ax.boxplot([control_boot, test_boot], labels = [self.Control.name, self.Test.name], patch_artist = True, whis = [alpha/2, 100-alpha/2], widths = 0.5)
        colors = [u'#00538a', u'#ff9a42']
        for patch, color in zip(bp['boxes'], colors): 
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor('black') # or try 'black'
            patch.set_linewidth(1)
        plt.title('Distributions of statistic')
        plt.show()    
class MultipleSamples:
    def sample_combinations(self):
        return Utils.get_all_combinations(self.Samples, 2)
class DeleteOutliersTwoSamples:
    deleted_outliers = False
    deleted_outliers_perc = None
    deleted_outliers_data = pd.DataFrame()
    deleted_outliers_period_type = None
    def delete_outliers(self, outliers_perc = [0,100]):
        if not self.deleted_outliers:
            down_limit, up_limit = np.percentile(self.df()['value'], outliers_perc)
            without_outliers_data = self.df().query('value >= @down_limit and value <= @up_limit')
            self.deleted_outliers_data = self.df().query('value < @down_limit or value > @up_limit') # save deleted data
            control_name = self.control_name
            test_name = self.test_name

            control_array = without_outliers_data.query('group==@control_name')['value']
            control_array_before = without_outliers_data.query('group==@control_name')['value_before']
            control_categories = without_outliers_data.query('group==@control_name')['category']
            self.Control = Sample(control_array, control_categories, control_array_before, name = control_name)

            test_array = without_outliers_data.query('group==@test_name')['value']
            test_array_before = without_outliers_data.query('group==@test_name')['value_before']
            test_categories = without_outliers_data.query('group==@test_name')['category']
            self.Test = Sample(test_array, test_categories, test_array_before, name = test_name)

            self.deleted_outliers = True
            self.deleted_outliers_perc = outliers_perc
        elif outliers_perc != self.deleted_outliers_perc:
            down_limit, up_limit = np.percentile(pd.concat([self.df(), self.deleted_outliers_data])['value'], outliers_perc)
            without_outliers_data = pd.concat([self.df(), self.deleted_outliers_data]).query('value >= @down_limit and value <= @up_limit')
            self.deleted_outliers_data = pd.concat([self.df(), self.deleted_outliers_data]).query('value < @down_limit or value > @up_limit') # save deleted data
            control_name = self.control_name
            test_name = self.test_name

            control_array = without_outliers_data.query('group==@control_name')['value']
            control_array_before = without_outliers_data.query('group==@control_name')['value_before']
            control_categories = without_outliers_data.query('group==@control_name')['category']
            self.Control = Sample(control_array, control_categories, control_array_before, name = control_name)

            test_array = without_outliers_data.query('group==@test_name')['value']
            test_array_before = without_outliers_data.query('group==@test_name')['value_before']
            test_categories = without_outliers_data.query('group==@test_name')['category']
            self.Test = Sample(test_array, test_categories, test_array_before, name = test_name)

            self.deleted_outliers = True
            self.deleted_outliers_perc = outliers_perc
        self.deleted_outliers_period_type = 'after'
        return self
    def delete_outliers_before_period(self, outliers_perc = [0,100]):
        if not self.deleted_outliers:
            down_limit, up_limit = np.percentile(self.df()['value_before'], outliers_perc)
            without_outliers_data = self.df().query('value_before >= @down_limit and value_before <= @up_limit')
            self.deleted_outliers_data = self.df().query('value_before < @down_limit or value_before > @up_limit') # save deleted data
            control_name = self.control_name
            test_name = self.test_name

            control_array = without_outliers_data.query('group==@control_name')['value']
            control_array_before = without_outliers_data.query('group==@control_name')['value_before']
            control_categories = without_outliers_data.query('group==@control_name')['category']
            self.Control = Sample(control_array, control_categories, control_array_before, name = control_name)

            test_array = without_outliers_data.query('group==@test_name')['value']
            test_array_before = without_outliers_data.query('group==@test_name')['value_before']
            test_categories = without_outliers_data.query('group==@test_name')['category']
            self.Test = Sample(test_array, test_categories, test_array_before, name = test_name)

            self.deleted_outliers = True
            self.deleted_outliers_perc = outliers_perc
        elif outliers_perc != self.deleted_outliers_perc:
            down_limit, up_limit = np.percentile(pd.concat([self.df(), self.deleted_outliers_data])['value'], outliers_perc)
            without_outliers_data = pd.concat([self.df(), self.deleted_outliers_data]).query('value_before >= @down_limit and value_before <= @up_limit')
            self.deleted_outliers_data = pd.concat([self.df(), self.deleted_outliers_data]).query('value_before < @down_limit or value_before > @up_limit') # save deleted data
            control_name = self.control_name
            test_name = self.test_name

            control_array = without_outliers_data.query('group==@control_name')['value']
            control_array_before = without_outliers_data.query('group==@control_name')['value_before']
            control_categories = without_outliers_data.query('group==@control_name')['category']
            self.Control = Sample(control_array, control_categories, control_array_before, name = control_name)

            test_array = without_outliers_data.query('group==@test_name')['value']
            test_array_before = without_outliers_data.query('group==@test_name')['value_before']
            test_categories = without_outliers_data.query('group==@test_name')['category']
            self.Test = Sample(test_array, test_categories, test_array_before, name = test_name)

            self.deleted_outliers = True
            self.deleted_outliers_perc = outliers_perc
        self.deleted_outliers_period_type = 'before'
        return self
class DeleteOutliersMultipleSamples: # !!!надо доделать аналогично классу DeleteOutliersTwoSamples
    deleted_outliers = False
    def delete_outliers(self, outliers_perc = [0,100]):
        if self.deleted_outliers == False:
            down_limit, up_limit = np.percentile(self.df()['value'], outliers_perc)
            without_outliers_data = self.df().query('value >= @down_limit and value <= @up_limit')
            
            for sample in self.Samples:
                sample_name = self.sample.name

                self.sample.array = without_outliers_data.query('group==@sample_name')['value']
                self.sample.array_before = without_outliers_data.query('group==@sample_name')['value_before']
                self.sample.categories = without_outliers_data.query('group==@sample_name')['category']

            self.deleted_outliers = True
        return self
    def delete_outliers_before_period(self, outliers_perc = [0,100]):
        if self.deleted_outliers == False:
            down_limit, up_limit = np.percentile(self.df()['value_before'], outliers_perc)
            without_outliers_data = self.df().query('value_before >= @down_limit and value_before <= @up_limit')
            
            for sample in self.Samples:
                sample_name = self.sample.name

                self.sample.array = without_outliers_data.query('group==@sample_name')['value']
                self.sample.array_before = without_outliers_data.query('group==@sample_name')['value_before']
                self.sample.categories = without_outliers_data.query('group==@sample_name')['category']

            self.deleted_outliers = True
        return self
class Sample(BootstrapOneSample, GetSizeForSample):
    name = 'sample'
    array: np.ndarray = np.array([])
    array_before: np.ndarray = np.array([])
    categories: pd.core.series.Series = pd.Series([], name = 'category')
    def __init__(self, array: np.ndarray, categories: pd.core.series.Series = None, array_before: np.ndarray = None, name:str = 'sample'):
        self.array = np.array(array)
        self.categories = pd.Series(array)
        self.array_before = np.array(array_before)
        Sample.name = name
        self.name = name
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
    def __str__(self):
        return f'{self.array}'
    def df(self):
        return pd.DataFrame({'category':self.categories, 'value':self.array, 'value_before':self.array_before})
    def mean(self):
        return np.mean(self.array)
    def std(self):
        return np.std(self.array)
    def count(self):
        return self.array.shape[0]
    def count_categories(self):
        return len(set(self.categories))
    def count_category_sizes(self):
        return self.categories.value_counts().reset_index().rename(columns = {'index':'category','category':'count'})
    def describe(self):
        if self.categories.nunique() == 1:
            return self.df().groupby(['category']).describe()
        else:
            return pd.concat([self.df().groupby(['category']).describe(), self.df().assign(category = 'all').groupby(['category']).describe()])
    def hist(self, outliers_perc = [0,100]):
        down_limit, up_limit = np.percentile(self.array, outliers_perc)
        sns.distplot(self.array[np.logical_and(self.array<=up_limit, self.array>=down_limit)], label = self.name)
        plt.title('Distribution density; del outliers by percentiles {}'.format(outliers_perc))
        plt.legend()
        plt.show()
    def __repr__(self):
        return repr(self.df())
class CompareTwoSamples(BootstrapTwoSamples, DeleteOutliersTwoSamples):
    def __init__(self, Control:Sample, Test:Sample):
        self.control = Control.array
        self.test = Test.array
        self.control_categories = Control.categories
        self.test_categories = Test.categories
        self.control_before = Control.array_before
        self.test_before = Test.array_before
        self.Control = Control
        self.Test = Test
        self.control_name = 'control' if self.Control.name == 'sample' else self.Control.name
        self.test_name = 'test' if self.Test.name == 'sample' else self.Test.name
    def df(self):
        return pd.concat([self.Control.df().assign(group = self.control_name), self.Test.df().assign(group = self.test_name)])
    def describe(self):
        if self.control_categories is None or self.test_categories is None:
            return self.df().assign(category = 'all').groupby(['category','group']).describe()
        else:
            return pd.concat([self.df().groupby(['category','group']).describe(), self.df().assign(category = 'all').groupby(['category','group']).describe()]).sort_index()
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
        return ExperimentComparisonResults(alpha, pvalue, effect, ci_length, left_bound, right_bound)
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
            
            return ExperimentComparisonResults(alpha, pvalue, effect, ci_length, left_bound, right_bound)
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
            return ExperimentComparisonResults(alpha, pvalue, effect, ci_length, left_bound, right_bound)
    def bootstrap_test(self, alpha = 0.05, stat_func = np.mean, test_type = 'absolute', n_samples = 1000, stratify = False, chart = False):
        absolute_func = lambda C, T: T - C
        relative_func = lambda C, T: T / C - 1

        boot_func = absolute_func if test_type == 'absolute' else relative_func

        control_boot, test_boot = self.boot_results(stat_func = stat_func, before_samples = None, n_samples = n_samples, stratify = stratify)
        boot_data = boot_func(control_boot, test_boot)

        left_bound, right_bound = np.quantile(boot_data, [alpha/2, 1 - alpha/2])
        ci_length = (right_bound - left_bound)
        effect = boot_func(stat_func(self.Control.array), stat_func(self.Test.array))
        pvalue = 2 * min(np.mean(boot_data > 0), np.mean(boot_data < 0))

        if chart == True:
            sns.distplot(control_boot, label = self.control_name)
            sns.distplot(test_boot, label = self.test_name)
            plt.title("Distributions of statistic")
            plt.xlabel('statistic')
            plt.ylabel('density')
            plt.legend()
            plt.show()

        return ExperimentComparisonResults(alpha, pvalue, effect, ci_length, left_bound, right_bound)
    def post_normed_bootstrap_test(self, alpha = 0.05, stat_func = np.mean, test_type='absolute', n_samples = 1000, stratify = False, chart = False):
        absolute_func = lambda C, T, C_b, T_b: T - (T_b / C_b) * C
        relative_func = lambda C, T, C_b, T_b: (T / C) / (T_b / C_b) - 1
                           
        boot_func = absolute_func if test_type == 'absolute' else relative_func
        
        control_boot, control_before_boot, test_boot, test_before_boot = self.boot_results(stat_func = stat_func, before_samples = True, n_samples = n_samples, stratify = stratify)
        boot_data = boot_func(control_boot, test_boot, control_before_boot, test_before_boot)
        
        left_bound, right_bound = np.quantile(boot_data, [alpha/2, 1 - alpha/2])
        ci_length = (right_bound - left_bound)
        effect = boot_func(np.mean(self.control), np.mean(self.test), np.mean(self.control_before), np.mean(self.test_before))
        pvalue = 2 * min(np.mean(boot_data > 0), np.mean(boot_data < 0))
        
        if chart == True:
            sns.distplot(control_boot, label = self.control_name, color = u'#00538a')
            sns.distplot(test_boot, label = self.test_name, color =  u'#ff7f0e')
            sns.distplot(control_before_boot, label = str(self.control_name) + '_before', color = u'#2e93db')
            sns.distplot(test_before_boot, label = str(self.test_name) + '_before', color =  u'#ff9a42')
            plt.title("Distributions of statistic")
            plt.xlabel('statistic')
            plt.ylabel('density')
            plt.legend()
            plt.show()
            
            defalt_matplotlib_colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
            
        return ExperimentComparisonResults(alpha, pvalue, effect, ci_length, left_bound, right_bound)
    def hist(self, outliers_perc = [0,100]):
        down_limit, up_limit = np.percentile(np.concatenate([self.Control.array,self.Test.array]), outliers_perc)
        sns.distplot(self.Control.array[np.logical_and(self.Control.array<=up_limit, self.Control.array>=down_limit)], label = self.control_name)
        sns.distplot(self.Test.array[np.logical_and(self.Test.array<=up_limit, self.Test.array>=down_limit)], label = self.test_name)
        plt.title('Distribution density; del outliers by percentiles {}'.format(outliers_perc))
        plt.legend()
        plt.show()
    def __repr__(self):
        return repr(self.df())
class CompareMultipleSamples(MultipleSamples):
    def __init__(self, Samples:list):
        self.samples_count = len(Samples)
        self.Samples = Samples
        self.arrays = [sample.array for sample in Samples]
        self.names = [sample.name for sample in Samples]
    def df(self):
        return pd.concat([sample.df().assign(group = sample.name) for sample in self.Samples])
    def describe(self):
        if np.max([sample.count_categories() for sample in self.Samples]) > 1:
            return pd.concat([self.df().groupby(['category','group']).describe(), self.df().assign(category = 'all').groupby(['category','group']).describe()]).sort_index()
        else:
            return self.df().assign(category = 'all').groupby(['category','group']).describe()            
    def anova_test(self, alpha = 0.05):
        _, pvalue = f_oneway(*self.arrays)
        return ExperimentComparisonResults(alpha, pvalue, None, None, None, None)
    def tukey_hsd_test(self, alpha = 0.05):
        return pairwise_tukeyhsd(endog=self.df()['value'], groups=self.df()['group'], alpha=0.05).summary()
    def t_test(self, alpha = 0.05, test_type = 'absolute'):
        results = pd.DataFrame()
        for Control, Test in self.sample_combinations():
            compare_df = CompareTwoSamples(Control, Test).t_test(alpha = alpha, test_type = test_type).df().assign(group1 = Control.name, group2 = Test.name)
            results = pd.concat([results, compare_df])
        return results[['group1', 'group2', 'alpha', 'pvalue', 'effect', 'ci_length', 'left_bound', 'right_bound', 'reject']]
    def t_test_cuped(self, alpha = 0.05, test_type = 'absolute'):
        results = pd.DataFrame()
        for Control, Test in self.sample_combinations():
            compare_df = CompareTwoSamples(Control, Test).t_test_cuped(alpha = alpha, test_type = test_type).df().assign(group1 = Control.name, group2 = Test.name)
            results = pd.concat([results, compare_df])
        return results[['group1', 'group2', 'alpha', 'pvalue', 'effect', 'ci_length', 'left_bound', 'right_bound', 'reject']]
    def bootstrap_test(self, alpha = 0.05, stat_func = np.mean, test_type = 'absolute', n_samples = 1000, stratify = False, chart = False):
        results = pd.DataFrame()
        for Control, Test in self.sample_combinations():
            compare_df = CompareTwoSamples(Control, Test).bootstrap_test(alpha = alpha, stat_func = stat_func, test_type = test_type, n_samples = n_samples, stratify = stratify, chart = chart).df().assign(group1 = Control.name, group2 = Test.name)
            results = pd.concat([results, compare_df])
        return results[['group1', 'group2', 'alpha', 'pvalue', 'effect', 'ci_length', 'left_bound', 'right_bound', 'reject']]
    def post_normed_bootstrap_test(self, alpha = 0.05, stat_func = np.mean, test_type = 'absolute', n_samples = 1000, stratify = False, chart = False):
        results = pd.DataFrame()
        for Control, Test in self.sample_combinations():
            compare_df = CompareTwoSamples(Control, Test).post_normed_bootstrap_test(alpha = alpha, stat_func = stat_func, test_type = test_type, n_samples = n_samples, stratify = stratify, chart = chart).df().assign(group1 = Control.name, group2 = Test.name)
            results = pd.concat([results, compare_df])
        return results[['group1', 'group2', 'alpha', 'pvalue', 'effect', 'ci_length', 'left_bound', 'right_bound', 'reject']]
    def hist(self, outliers_perc = [0,100]):
        down_limit, up_limit = np.percentile(np.concatenate([sample.array for sample in self.Samples]), outliers_perc)
        for sample in self.Samples:
            sns.distplot(sample.array[np.logical_and(sample.array<=up_limit, sample.array>=down_limit)], label = sample.name)
        plt.title('Distribution density; del outliers by percentiles {}'.format(outliers_perc))
        plt.legend()
        plt.show()
    def __repr__(self):
        return repr(self.df())
class CheckCriterion(Sample):
    def __init__(self, Sample:Sample):
        self.Sample = Sample
        self.array = Sample.array
        self.array = Sample.array_before
        self.categories = Sample.categories
    def check(self, one_group_size = None, difference_pct = 0, alpha = 0.05, criterion = 'ttest', iterations = 5000):
        if one_group_size == None: one_group_size = 0.5    
        results = pd.DataFrame()
        # основные счётчики
        count_dif = 0
        count_level = 0
        
        if criterion in ['ttest_cuped', 'post_normed_bootstrap']:
            control_samples, control_before_samples = self.Sample.boot_samples(before_samples = True, n_samples = iterations, size = one_group_size)
            test_samples, test_before_samples = self.Sample.boot_samples(before_samples = True, n_samples = iterations, size = one_group_size)
            test_samples *= (1 + (difference_pct / 100))
        elif criterion in ['ttest', 'bootstrap']:
            control_samples = self.Sample.boot_samples(before_samples = False, n_samples = iterations, size = one_group_size)
            test_samples = self.Sample.boot_samples(before_samples = False, n_samples = iterations, size = one_group_size)
            test_samples *= (1 + (difference_pct / 100))
        
        if criterion in ['ttest', 'bootstrap']:
            for iteration in tqdm_notebook(range(iterations)):
                control, test = control_samples[iteration], test_samples[iteration]
                Control = Sample(array = control)
                Test = Sample(array = test)
                if criterion == 'ttest':
                    _, pvalue, _, _, left_bound, right_bound = CompareTwoSamples(Control, Test).t_test(alpha = alpha, test_type = 'relative').tuple()
                if criterion == 'bootstrap':
                    _, pvalue, _, _, left_bound, right_bound = CompareTwoSamples(Control, Test).bootstrap_test(alpha = alpha, stat_func = np.mean, test_type = 'relative', n_samples = 1000).tuple()
        
                if left_bound > 0 or right_bound < 0:
                    count_dif += 1
                if left_bound > difference_pct / 100 or right_bound < difference_pct / 100:
                    count_level += 1
        if criterion in ['ttest_cuped', 'post_normed_bootstrap']:
            for iteration in tqdm_notebook(range(iterations)):
                control, test, control_before, test_before = control_samples[iteration], test_samples[iteration], control_before_samples[iteration], test_before_samples[iteration]
                Control = Sample(array = control, array_before = control_before)
                Test = Sample(array = test, array_before = test_before)
                if criterion == 'ttest_cuped':
                    _, pvalue, _, _, left_bound, right_bound = CompareTwoSamples(Control, Test).t_test_cuped(alpha = alpha, test_type = 'relative').tuple()
                if criterion == 'post_normed_bootstrap':
                    _, pvalue, _, _, left_bound, right_bound = CompareTwoSamples(Control, Test).post_normed_bootstrap_test(alpha = alpha, stat_func = np.mean, test_type = 'relative', n_samples = 1000).tuple()
        
                if left_bound > 0 or right_bound < 0:
                    count_dif += 1
                if left_bound > difference_pct / 100 or right_bound < difference_pct / 100:
                    count_level += 1
        
        # create results
        left_real_level, right_real_level = proportion_confint(count = count_level, nobs = iterations, alpha=0.05, method='wilson')
        results.loc[0, 'criterion'] = criterion
        if difference_pct == 0:
            results.loc[0, 'type'] = 'AA'
        else:
            results.loc[0, 'type'] = 'AB'
        if type(one_group_size) == float:
            results.loc[0, 'one_group_size'] = int(one_group_size * self.Sample.count())
        elif type(one_group_size) == int:
            results.loc[0, 'one_group_size'] = one_group_size
        results.loc[0, 'difference_pct'] = difference_pct
        results.loc[0, 'alpha'] = alpha
        results.loc[0, 'iterations'] = iterations
        results.loc[0, 'dif_detected/power'] = round(count_dif / iterations, 4)
        results.loc[0, 'real_level'] = round(count_level / iterations, 4)
        results.loc[0, 'left_real_level'] = left_real_level
        results.loc[0, 'right_real_level'] = right_real_level
        return results  
    def check_iterable(self, one_group_size = None, difference_pct = 0, alpha = 0.05, criterion = 'ttest', iterations = 5000):
        vars_dict = {
            'one_group_size':one_group_size,
            'difference_pct':difference_pct,
            'alpha':alpha,
            'criterion':criterion,
            'iterations':iterations
        }
        iterable_vars_dict = {name:vars_dict[name] for name in vars_dict if type(vars_dict[name])==list}
        if iterable_vars_dict != {}:
            if np.mean([len(iterable_vars_dict[name]) for name in iterable_vars_dict]) != [len(iterable_vars_dict[name]) for name in iterable_vars_dict][0]:
                print('Error! vars lists must be the same length!')
                return None
            results = pd.DataFrame()
            for _, row in pd.DataFrame(vars_dict).iterrows():
                print(row.to_dict())  
                results = pd.concat([results, self.check(**row.to_dict())])
            return results
        else:
            return self.check(**vars_dict)
    def t_test(self, one_group_size = None, difference_pct = 0, alpha = 0.05, iterations = 5000):
        return self.check_iterable(one_group_size, difference_pct, alpha, criterion = 'ttest', iterations = iterations)
    def t_test_cuped(self, one_group_size = None, difference_pct = 0, alpha = 0.05, iterations = 5000):
        return self.check_iterable(one_group_size, difference_pct, alpha, criterion = 'ttest_cuped', iterations = iterations)
    def bootstrap_test(self, one_group_size = None, difference_pct = 0, alpha = 0.05, iterations = 5000):
        return self.check_iterable(one_group_size, difference_pct, alpha, criterion = 'bootstrap', iterations = iterations)
    def post_normed_bootstrap_test(self, one_group_size = None, difference_pct = 0, alpha = 0.05, iterations = 5000):
        return self.check_iterable(one_group_size, difference_pct, alpha, criterion = 'post_normed_bootstrap', iterations = iterations)
class ExperimentComparisonResults:
    def __init__(self, alpha, pvalue, effect, ci_length, left_bound, right_bound):
        self.alpha = alpha
        self.pvalue = pvalue
        self.effect = effect
        self.ci_length = ci_length
        self.left_bound = left_bound
        self.right_bound = right_bound
    def __str__(self):
        dict_results = {'alpha':alpha,'pvalue':pvalue,'effect':effect,'ci_length':ci_length,'left_bound':left_bound,'right_bound':right_bound}
        return f'{dict_results}'
    def df(self):
        results = pd.DataFrame({'alpha':[self.alpha],
            'pvalue':[self.pvalue],
            'effect':[self.effect],
            'ci_length':[self.ci_length],
            'left_bound':[self.left_bound],
            'right_bound':[self.right_bound]})
        results['reject'] = results.apply(lambda row: True if row['pvalue'] < row['alpha'] else False, axis = 1)
        return results
    def tuple(self):
        return self.alpha,self.pvalue,self.effect,self.ci_length,self.left_bound,self.right_bound
class Utils:
    def similar_sample(sample, size = None):
        '''
        Эта функция создает похожую выборку с таким же распределением
        '''
        if size is None:
            size = len(sample)
        hist = np.histogram(sample, bins = len(sample))
        return sps.rv_histogram(hist).rvs(size = len(sample))
    def get_all_combinations(array, lenght):
        return itertools.combinations(array, lenght)
    def get_samples(df, groups_column:str, values_column:str, categories_column:str = None, values_before_column:str = None):
        samples_list = list() 
        for group_name in df[groups_column].unique():
            sample_df = df.query(f'{groups_column}==@group_name')
            categories = None if categories_column == None else sample_df[categories_column]
            values_before = None if values_before_column == None else sample_df[values_before_column]
            samples_list.append(Sample(array = sample_df[values_column], categories = categories, array_before = values_before, name = group_name))
        return samples_list
    def get_fractions(df, groups_column:str, count_column:str, nobs_column:str):
        fractions_list = list() 
        for group_name in df[groups_column].unique():
            sample_df = df.query(f'{groups_column}==@group_name')
            fractions_list.append(Fraction(count = sample_df[count_column].iloc[0], nobs = sample_df[nobs_column].iloc[0], name = group_name))
        return fractions_list
    def date_from_text(text, pattern = '%Y-%m-%d'):
        '''
        default pattern = '%Y-%m-%d'
        '''
        return datetime.datetime.strptime(text, pattern).date()
class GetSizeForFraction:
    def needed_size(self, alpha=0.05, diff_type='pct', diff=5):
        ''' 
        :param diff_type: может быть равен 'pct' или 'absolute' 
        :param diff: если разница в процентах, то нужно число, напр. 5% или 10% 
        '''
        z = np.round(st.norm.interval(1 - alpha, loc=0, scale=1)[1], 2)
        p = self.count / self.nobs
        if diff_type=='pct': mde = diff / 100
        if diff_type=='absolute': mde = diff / p
        return z**2 * p * (1 - p) / mde**2
class Fraction(GetSizeForFraction):
    name = 'fraction'
    def __init__(self, count: int, nobs: int, name: str = 'fraction'):
        self.count = count
        self.nobs = nobs
        self.name = name
        self.fraction = count / nobs
        Fraction.name = name
    def df(self):
        df = pd.DataFrame()
        df.loc[0, 'group'], df.loc[0, 'count'], df.loc[0, 'nobs'], df.loc[0, 'fraction'] = self.name, self.count, self.nobs, self.fraction
        return df
    def ci(self, alpha = 0.05):
        return proportion_confint(count = self.count, nobs = self.nobs, alpha=alpha, method='wilson')
    def __repr__(self):
        return repr(self.df())
class CompareTwoFractions:
    def __init__(self, Test:Fraction, Control:Fraction):
        self.Control = Control
        self.Test = Test
        self.Control.name = 'Control' if self.Control.name == 'fraction' else self.Control.name
        self.Test.name = 'Test' if self.Test.name == 'fraction' else self.Test.name
    def z_test(self, alpha = 0.05):
        # пропорция успехов в первой группе:
        p1 = self.Control.count/self.Control.nobs
        # пропорция успехов во второй группе:
        p2 = self.Test.count/self.Test.nobs
        # пропорция успехов в комбинированном датасете:
        p_combined = (self.Control.count + self.Test.count) / (self.Control.nobs + self.Test.nobs)
        # разница пропорций в датасетах
        difference = effect = p1 - p2 

        # считаем статистику в ст.отклонениях стандартного нормального распределения
        z_value = difference / mth.sqrt(p_combined * (1 - p_combined) * (1/self.Control.nobs + 1/self.Test.nobs))
        # задаем стандартное нормальное распределение (среднее 0, ст.отклонение 1)
        distr = sps.norm(0, 1) 
        pvalue = (1 - distr.cdf(abs(z_value))) * 2
        # найдем доверительный интервал разницы
        se_1 = np.sqrt (p1 * (1 - p1) / self.Control.nobs)
        se_2 = np.sqrt (p2 * (1 - p2) / self.Test.nobs)
        se_diff = np.sqrt(se_1**2 + se_2**2)
        left_bound, right_bound = difference - sps.norm(0,1).ppf(1-alpha/2) * se_diff, difference + sps.norm(0,1).ppf(1-alpha/2) * se_diff
        ci_length = (right_bound - left_bound)
        
        return ExperimentComparisonResults(alpha, pvalue, effect, ci_length, left_bound, right_bound)
    def df(self):
        return pd.concat([self.Control.df(), self.Test.df()])
    def __repr__(self):
        return repr(self.df())
class MultipleFractions:
    def fraction_combinations(self):
        return Utils.get_all_combinations(self.Fractions, 2)
class CompareMultipleFractions(MultipleFractions):
    def __init__(self, Fractions:list):
        self.fractions_count = len(Fractions)
        self.Fractions = Fractions    
    def df(self):
        return pd.concat([sample.df() for sample in self.Fractions])
    def __repr__(self):
        return repr(self.df())
    def z_test(self, alpha = 0.05):
        results = pd.DataFrame()
        for Control, Test in self.fraction_combinations():
            compare_df = CompareTwoFractions(Control, Test).z_test(alpha = alpha).df().assign(group1 = Control.name, group2 = Test.name)
            results = pd.concat([results, compare_df])
        return results[['group1', 'group2', 'alpha', 'pvalue', 'effect', 'ci_length', 'left_bound', 'right_bound', 'reject']]
