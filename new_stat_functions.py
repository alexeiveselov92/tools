#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd 
import scipy.stats as sps
from collections import namedtuple
import seaborn as sns 
from tqdm.notebook import tqdm as tqdm_notebook
from statsmodels.stats.proportion import proportion_confint
    
class Utils:
    def similar_sample(sample, size = None):
        '''
        Эта функция создает похожую выборку с таким же распределением
        '''
        if size is None:
            size = len(sample)
        hist = np.histogram(sample, bins = len(sample))
        return sps.rv_histogram(hist).rvs(size = len(sample))

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
        
class BootstrapOneSample(Sample):
    def __init__(self, Sample:Sample):
        self.Sample = Sample
        self.array = Sample.array
        self.array_before = Sample.array_before
        self.categories = Sample.categories
    def boot_samples(self, before_samples = None, n_samples = 1000, stratify = False, size = None):
        if size is None:
            size = self.Sample.count()
        if stratify == True:
            categories_iterable_df = self.count_category_sizes()
        else:
            categories_iterable_df = pd.DataFrame({'category':['no_category'], 'count':[self.Sample.count()]})
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
        return pd.DataFrame({'alpha':[self.alpha],
            'pvalue':[self.pvalue],
            'effect':[self.effect],
            'ci_length':[self.ci_length],
            'left_bound':[self.left_bound],
            'right_bound':[self.right_bound]})
    def tuple(self):
        return self.alpha,self.pvalue,self.effect,self.ci_length,self.left_bound,self.right_bound
    
class CompareTwoSamplesInterface:
    def hist(self):
        raise NotImplemented("you should override this method")

class CompareTwoSamples(CompareTwoSamplesInterface):
    def __init__(self, Control:Sample, Test:Sample):
        self.control = Control.array
        self.test = Test.array
        self.control_categories = Control.categories
        self.test_categories = Test.categories
        self.control_before = Control.array_before
        self.test_before = Test.array_before
        self.Control = Control
        self.Test = Test
    def df(self):
        return pd.concat([self.Control.df().assign(group = 'control'), self.Test.df().assign(group = 'test')])
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

        control_boot, test_boot = BootstrapTwoSamples(self.Control, self.Test).boot_results(stat_func = stat_func, before_samples = None, n_samples = n_samples, stratify = stratify)
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

        return ExperimentComparisonResults(alpha, pvalue, effect, ci_length, left_bound, right_bound)
    def post_normed_bootstrap_test(self, alpha = 0.05, stat_func = np.mean, test_type='absolute', n_samples = 1000, stratify = False, chart = False):
        absolute_func = lambda C, T, C_b, T_b: T - (T_b / C_b) * C
        relative_func = lambda C, T, C_b, T_b: (T / C) / (T_b / C_b) - 1
                           
        boot_func = absolute_func if test_type == 'absolute' else relative_func
        
        control_boot, control_before_boot, test_boot, test_before_boot = BootstrapTwoSamples(self.Control, self.Test).boot_results(stat_func = stat_func, before_samples = True, n_samples = n_samples, stratify = stratify)
        boot_data = boot_func(control_boot, test_boot, control_before_boot, test_before_boot)
        
        left_bound, right_bound = np.quantile(boot_data, [alpha/2, 1 - alpha/2])
        ci_length = (right_bound - left_bound)
        effect = boot_func(np.mean(control), np.mean(test), np.mean(control_before), np.mean(test_before))
        pvalue = 2 * min(np.mean(boot_data > 0), np.mean(boot_data < 0))
        
        if chart == True:
            sns.distplot(control_boot, label = 'control', color = u'#00538a')
            sns.distplot(test_boot, label = 'test', color =  u'#ff7f0e')
            sns.distplot(control_before_boot, label = 'control_before', color = u'#2e93db')
            sns.distplot(test_before_boot, label = 'test_before', color =  u'#ff9a42')
            plt.title("Distributions of statistic")
            plt.xlabel('statistic')
            plt.ylabel('density')
            plt.legend()
            plt.show()
            
            defalt_matplotlib_colors = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf']
            
        return ExperimentComparisonResults(alpha, pvalue, effect, ci_length, left_bound, right_bound)
    def hist(self, outliers_perc = [0,100]):
        down_limit, up_limit = np.percentile(np.concatenate([self.Control.array,self.Test.array]), outliers_perc)
        sns.distplot(self.Control.array[np.logical_and(self.Control.array<=up_limit, self.Control.array>=down_limit)], label = 'control')
        sns.distplot(self.Test.array[np.logical_and(self.Test.array<=up_limit, self.Test.array>=down_limit)], label = 'test')
        plt.title('Distribution density; del outliers by percentiles {}'.format(outliers_perc))
        plt.legend()
        plt.show()

class CheckCriterion(Sample):
    def __init__(self, Sample:Sample):
        self.Sample = Sample
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
            control_samples, control_before_samples = BootstrapOneSample(self.Sample).boot_samples(before_samples = True, n_samples = iterations, size = one_group_size)
            test_samples, test_before_samples = BootstrapOneSample(self.Sample).boot_samples(before_samples = True, n_samples = iterations, size = one_group_size)
            test_samples *= (1 + (difference_pct / 100))
        elif criterion in ['ttest', 'bootstrap']:
            control_samples = BootstrapOneSample(self.Sample).boot_samples(before_samples = False, n_samples = iterations, size = one_group_size)
            test_samples = BootstrapOneSample(self.Sample).boot_samples(before_samples = False, n_samples = iterations, size = one_group_size)
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
            results.loc[0, 'one_group_size'] = one_group_size * len(data)
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