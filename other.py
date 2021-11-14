#!/usr/bin/env python
# coding: utf-8

import datetime
import requests
import json

# datetime from text
def datetime_from_text(text, pattern = '%Y-%m-%d %H:%M:%S'):
    '''
    default pattern = '%Y-%m-%d %H:%M:%S'
    '''
    return datetime.datetime.strptime(text, pattern)
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