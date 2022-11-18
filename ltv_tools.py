#!/usr/bin/env python
# coding: utf-8

from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

class LTVApproximate:
    ltv_func = lambda x, a, b, c: a*np.log(x+100*np.tanh(b))+c
    def __init__(self, data, x_col, y_col):
        '''
        Данный класс предназначен для расчёта коэффицентов a,b,c кривой накопительного ARPU / ARPPU,
        используемых далее в создании прогноза LTV.
        :param x_col: столбец со временем жизни юзера (1,2,3...N, где N - день жизни юзера от регистрации)
        :param y_col: столбец с накопительным значением метрики ARPU / ARPPU, соответствующим возрасту юзера из x_col 
        '''
        self.data = data
        self.x_col = x_col
        self.y_col = y_col
        self.predict_params_data = None
    def get_predict_params(self, segment_cols = None, query_cond = None, plot = False):
        '''
        Данный метод возращает датафрейм с коэффициентами прогнозирования LTV и попутно показывает графики аппроксимации нашей метрики. 
        :param segment_cols: список столбцов, по которым следует разбить наши входные данные на сегменты, чтобы рассчитать коэффцициенты для каждого из данных сегментов
        :param query_cond: - доп. условие, которое мы хотим применять к данным перед расчётом коэффициентов. Например, если у нас в датафрейме есть данные о числе юзеров, мы 
        можем отфильтровать строки, где их слишком мало у нас, прописав 'users >= 50'.
        :param plot: отрисовать ли график аппроксимации нашей метрики 
        '''
        predict_params_data = pd.DataFrame()
        data = self.data if segment_cols is not None else self.data.assign(segment = 'no_segments')
        segment_cols = 'segment' if segment_cols is None else segment_cols
        segment_cols = [segment_cols] if segment_cols is not None and type(segment_cols) == str else segment_cols
            
        for groupby_tuple in data.groupby(segment_cols):
            sample = groupby_tuple[1].sort_values(by = self.x_col).query(query_cond) if query_cond is not None else groupby_tuple[1].sort_values(by = self.x_col)
            segment_vals = groupby_tuple[0]
            segment_dict = {col_name: val for col_name, val in zip(segment_cols, segment_vals)}
            
            try:
                popt, pcov = curve_fit(LTVApproximate.ltv_func, sample[self.x_col], sample[self.y_col], maxfev = 1000000, bounds = [0,np.inf])
            except:
                continue
            a,b,c = popt
            
            predict_params_data = predict_params_data.append({**segment_dict, **{
                'a': a,
                'b': b,
                'c': c,
                'size': sample.shape[0],
                'last_x': sample[self.x_col].max(),
                'x_values': sample[self.x_col].values,
                'y_values': sample[self.y_col].values
            }}, ignore_index = True)
            predict_params_data = predict_params_data[segment_cols + ['a','b','c','size','last_x', 'x_values', 'y_values']]
            if plot:
                sns.lineplot(data = sample, x = self.x_col, y = self.y_col, label = 'y')
                plt.plot(sample[self.x_col], LTVApproximate.ltv_func(sample[self.x_col], *popt), 'r--', label = 'y_pred')
                plt.title('segment:\n' + '\n'.join([f'{key}: {segment_dict[key]}' for key in segment_dict.keys()]))
                plt.legend()
                plt.show()
                
        if segment_cols is None:
            predict_params_data = predict_params_data.drop(columns = ['segment'])
        self.predict_params_data = predict_params_data
        return predict_params_data