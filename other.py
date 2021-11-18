#!/usr/bin/env python
# coding: utf-8

import datetime
import requests
import json
import io
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats as st
from scipy.optimize import curve_fit

# datetime or date from text
def datetime_from_text(text, pattern = '%Y-%m-%d %H:%M:%S'):
    '''
    default pattern = '%Y-%m-%d %H:%M:%S'
    '''
    return datetime.datetime.strptime(text, pattern)
def date_from_text(text, pattern = '%Y-%m-%d'):
    '''
    default pattern = '%Y-%m-%d'
    '''
    return datetime.datetime.strptime(text, pattern).date()
# post messages to slack
def post_message_to_slack(
        text, 
        slack_token, 
        channel_or_user_id, 
        slack_icon_url, 
        slack_user_name='alarm_bot', 
        as_user=False, 
        blocks=None
    ):
    '''
    ### blocks example
    blocks = [{  
      "type": "section",
      "text": {  
        "type": "mrkdwn",
        "text": "*The script* has run\n successfully on the dev."
      }
    }]

    More about blocks - https://app.slack.com/block-kit-builder/
    '''
    body_dict = {
        'token': slack_token,
        'channel': channel_or_user_id,
        'text': text,
        'username': slack_user_name,
        'blocks': json.dumps(blocks) if blocks else None,
        'icon_url':slack_icon_url
    }
    if as_user == True:
        body_dict['as_user'] = 'true'
    return requests.post('https://slack.com/api/chat.postMessage', body_dict).json()
# convert matplotlib plot to bytes object
def plt_chart_to_bytes():
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img_in_bytes = buf.read()
    buf.close()
    return img_in_bytes
# post files to slack
def post_file_to_slack(
        slack_token, 
        channel_or_user_id, 
        text, 
        file_name, 
        file_bytes, 
        file_type=None, 
        title=None
    ): 
    '''
    send matplotlib plot as example:
    
    plt.figure(figsize=(20,7))
    plt.plot([1,2,3],[2,3,4])
    img = plt_chart_to_bytes()
    plt.show()
    post_file_to_slack(slack_token, channel_or_user_id, 'something text', something.png', img)
    '''
    return requests.post(
      'https://slack.com/api/files.upload', 
      {
        'token': slack_token,
        'filename': file_name,
        'channels': channel_or_user_id,
        'filetype': file_type,
        'initial_comment':  text,
        'title': title
      },
      files = { 'file': file_bytes }).json()
# function for ltv forecasting
def curve_f(x, a, b, c):
    return a*np.log(x + 100*np.tanh(b)) + c
# get fcst ltv coefficients
def get_ltv_coefficients(df, x_column, target_column):
    '''
    x_column: string - column name of the independent variable, days as example
    target_column: string - column name of the dependent variable, ltv or arppu cummulative
    '''
    curve_coefs,_ = curve_fit(curve_f, df[days_column], df[target_column], maxfev = 500000) 
    return curve_coefs
# get fcst plot
def get_ltv_fcst_plot(df, x_column, target_column, x_range_tuple = None):
    '''
    x_column: string - column name of the independent variable, days as example
    target_column: string - column name of the dependent variable, ltv or arppu cummulative
    x_range_tuple: tuple, range for x values to display on the chart as a forecast - (0,365) as example
    '''
    df = df.sort_values(by = x_column, ascending = True)
    coefs = get_ltv_coefficients(df, x_column, target_column)
    df.apply(lambda x: curve_f(365,*coefs), axis = 1)
    plt.plot(df[x_column], df[target_column], label = 'fact', color = '#2b6ca3')
    if x_range_tuple != None:
        x_fcst_values = [i for i in range(*x_range_tuple)]
    else:
        x_fcst_values = df[x_column].to_list()
    plt.plot(x_fcst_values, pd.Series(map(lambda x: curve_f(x, *coefs), x_fcst_values)), label = 'fcst', color = 'green')
    plt.legend()
    plt.show()