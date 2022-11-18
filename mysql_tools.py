#!/usr/bin/env python
# coding: utf-8

# pip install PyMySQL
import pymysql
import pandas as pd
class MySQLTools:
    def __init__(self, host, port, user, password, database):
        self.conn = pymysql.connect(
            host = host, 
            port = port,
            user = user, 
            password = password, 
            database = database
         )
    def select(self, q):
        df = pd.read_sql_query(f'''
        {q}
        ''', self.conn)
        return df