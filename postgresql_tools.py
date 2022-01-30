#!/usr/bin/env python
# coding: utf-8

import pandahouse as ph
import pandas as pd
import numpy as np
import time
import datetime

import psycopg2
from sqlalchemy import create_engine
class postgresql_tools:
    def __init__(self, db_name, db_user, db_password, db_host, db_port):
        try:
            self.connection = psycopg2.connect(
                database=db_name,
                user=db_user,
                password=db_password,
                host=db_host,
                port=db_port,
            )
            print("Connection to PostgreSQL DB successful")
        except Exception as e:
            print(f"The error '{e}' occurred")
        self.connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(db_user,db_password,db_host,db_port,db_name)
    def execute(self, query):
        connection = self.connection
        connection.autocommit = True
        cursor = connection.cursor()
        try:
            cursor.execute(query)
            cursor.close()
            print("Query executed successfully")
        except Exception as e:
            print(f"The error '{e}' occurred")
            cursor.close()          
    def select(self, query):
        connection_string = self.connection_string
        engine = create_engine(connection_string)
        df = pd.io.sql.read_sql(query, con = engine)
        return df
    def insert_df_to_db(self, df, table_name, print_results = True):
        connection_string = self.connection_string
        engine = create_engine(connection_string)
        df.to_sql(table_name, engine, if_exists = 'append', index = False)
        if print_results == True: print(f'Select data have been successfully writed to {table_name}!')