#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import datetime
import sys
# lifetimes
import lifetimes
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import \
    calibration_and_holdout_data, \
    summary_data_from_transaction_data, \
    calculate_alive_path

class CallBGFModelError:
    def check_the_availability_bgf_model(self):
        if self.bgf_model is None:
            raise BaseException('Вы не обучили модель для прогнозирования. Используйте метод fit_bgf_model()')
class CallGGModelError:
    def check_the_availability_ggf_model(self):
        if self.ggf_model is None:
            raise BaseException('Вы не обучили модель для прогнозирования. Используйте метод fit_ggf_model()')
    def check_corr_monetary_and_freq(self):
        corr_matrix = self.rftv_data[[self.monetary_col, self.frequency_col]].corr()
        corr = corr_matrix.iloc[1,0]
        print('Gamma-Gamma model requires a Pearson correlation close to 0 between purchase frequency and monetary value')
        print("Pearson correlation of monetary value and frequency: %.3f" % corr)
class RFTVData:
    def __init__(self, data, customer_id_col, recency_col = 'recency', frequency_col = 'frequency', T_col = 'T', monetary_col = None):
        data_cols = [recency_col, frequency_col, T_col, monetary_col] if monetary_col else [customer_id_col, recency_col, frequency_col, T_col]
        current_cols = ['recency', 'frequency', 'T', 'monetary_value'] if monetary_col else ['recency', 'frequency', 'T']
        self.rftv_data = data.set_index(customer_id_col)[data_cols].query(f'{monetary_col}>0')
        self.rftv_data.columns = current_cols
        self.monetary_col = monetary_col
        self.customer_id_col = customer_id_col
class TransactionsData:
    def __init__(self, data, datetime_col, customer_id_col, monetary_col = None):
        data[datetime_col] = pd.to_datetime(data[datetime_col]).dt.date
        data[customer_id_col] = data[customer_id_col].astype('str')
        self.monetary_col = monetary_col
        self.customer_id_col = customer_id_col
        self.period_end = data[datetime_col].max()
        
        self.rftv_data = summary_data_from_transaction_data(
            transactions = data.query(f'{monetary_col}>0'), 
            customer_id_col = customer_id_col, 
            datetime_col = datetime_col, 
            observation_period_end = data[datetime_col].max(), 
            freq = "D",
            monetary_value_col = monetary_col
        )
class LifetimesTools(CallBGFModelError, CallGGModelError):
    bgf_model = None
    ggf_model = None
    recency_col = 'recency'
    frequency_col = 'frequency'
    T_col = 'T'
    monetary_value_col = 'monetary_value'
    def __init__(self, data_class):
        self.rftv_data = data_class.rftv_data
        self.monetary_col = data_class.monetary_col
        self.customer_id_col = data_class.customer_id_col
    # bgf_model
    def fit_bgf_model(self, weights = None, verbose = True, tol = 1e-06):
        fit_success = False
        penalizer_coef = 0.000001
        while not fit_success:
            try:
                self.bgf_model = BetaGeoFitter(penalizer_coef=penalizer_coef)
                self.bgf_model.fit(
                    recency = self.rftv_data[self.recency_col], 
                    frequency = self.rftv_data[self.frequency_col], 
                    T = self.rftv_data[self.T_col], 
                    weights = weights, 
                    verbose = verbose, 
                    tol = tol
                )
                fit_success = True
            except:
                penalizer_coef = penalizer_coef * 10
                print(f'changed penalizer_coef to {penalizer_coef}')
        return self.bgf_model.summary
    def bgf_model_predict_purchases(self, pred_days = [10], inplace = False):
        '''
        возвращает pd.DataFrame с прогнозным числом покупок по всем юзерам через pred_days дней 
        inplace - список, надо ли вставить прогноз числа дней покупок в rftv_data
        '''
        self.check_the_availability_bgf_model()
        if type(pred_days) != list:
            pred_days = [pred_days]
        pred_purchases_result_df = pd.DataFrame()
        for days in pred_days:
            pred_purchases_series = self.bgf_model.predict(days, self.rftv_data[self.frequency_col], self.rftv_data[self.recency_col], self.rftv_data[self.T_col])
            pred_purchases_df = pd.DataFrame({self.customer_id_col:self.rftv_data.reset_index()[self.customer_id_col], 'pred_purch_{}'.format(days):pred_purchases_series.to_list()})
            if pred_purchases_result_df.shape[0] == 0:
                pred_purchases_result_df = pred_purchases_df
            else:
                pred_purchases_result_df = pred_purchases_result_df.merge(pred_purchases_df, on = self.customer_id_col, suffixes=['_old',''])
        if inplace:
            self.rftv_data = self.rftv_data.merge(pred_purchases_result_df, on = self.customer_id_col, suffixes=['_old',''])
        # drop repeat columns
        self.rftv_data = self.rftv_data[[col for col in self.rftv_data.columns if '_old' not in col]]
        return pred_purchases_result_df
    def bgf_model_predict_alive(self, inplace = False):
        self.check_the_availability_bgf_model()
        # probability that a customer is alive for each customer in dataframe
        prob_alive = self.bgf_model.conditional_probability_alive(
            frequency = self.rftv_data[self.frequency_col], 
            recency = self.rftv_data[self.recency_col], 
            T = self.rftv_data[self.T_col]
        )
        pred_results = pd.DataFrame({self.customer_id_col:self.rftv_data.reset_index()[self.customer_id_col], 'prob_alive':prob_alive})
        if inplace and type(inplace) == list:
            if inplace:
                self.rftv_data = self.rftv_data.merge(pred_results, on = self.customer_id_col, suffixes = ['_old', ''])
        if inplace == True:
            self.rftv_data = self.rftv_data.merge(pred_results, on = self.customer_id_col, suffixes=['_old',''])
            
        # drop repeat columns
        self.rftv_data = self.rftv_data[[col for col in self.rftv_data.columns if '_old' not in col]]
        return pred_results
    # ggf_model
    def fit_ggf_model(self, penalizer_coef = 0.00001, weights = None, verbose = True, tol = 1e-06, q_constraint = True):
        self.ggf_model = GammaGammaFitter(penalizer_coef = penalizer_coef)
        self.ggf_model.fit(
            frequency = self.rftv_data[self.rftv_data[self.monetary_value_col]>0][self.frequency_col],
            monetary_value = self.rftv_data[self.rftv_data[self.monetary_value_col]>0][self.monetary_value_col],  
            weights = weights,   
            verbose = verbose,  
            tol = tol,  
            q_constraint = q_constraint
        )
        return self.ggf_model.summary
    def predict_ggf_model_exp_avg_rev(self, inplace = False):
        # estimate the average transaction value of each customer, based on frequency and monetary value
        exp_avg_rev = self.ggf_model.conditional_expected_average_profit(
            self.rftv_data[self.rftv_data[self.monetary_value_col]>0][self.frequency_col],
            self.rftv_data[self.rftv_data[self.monetary_value_col]>0][self.monetary_value_col]
        )
        exp_avg_rev_result_df = pd.DataFrame()
        exp_avg_rev_result_df[self.customer_id_col] = self.rftv_data[self.rftv_data[self.monetary_value_col]>0].index
        exp_avg_rev_result_df = exp_avg_rev_result_df.set_index(self.customer_id_col)
        exp_avg_rev_result_df["exp_avg_rev"] = exp_avg_rev
        exp_avg_rev_result_df['avg_rev'] = self.rftv_data[self.monetary_value_col]
        exp_avg_rev_result_df['error_rev'] = exp_avg_rev_result_df["exp_avg_rev"] - exp_avg_rev_result_df["avg_rev"]
        if inplace:
            self.rftv_data = self.rftv_data.merge(exp_avg_rev_result_df, left_index = True, right_index = True, how = 'left')
            self.rftv_data["exp_avg_rev"] = exp_avg_rev
            self.rftv_data["avg_rev"] = self.rftv_data[self.monetary_value_col]
            self.rftv_data["error_rev"] = self.rftv_data["exp_avg_rev"] - self.rftv_data["avg_rev"]
        return exp_avg_rev_result_df