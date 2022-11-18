#!/usr/bin/env python
# coding: utf-8

import pandahouse as ph
import pandas as pd
from clickhouse_driver import Client
import numpy as np
import time
import croniter
import datetime
import re

class ClickHouseTools:
    def __init__(self, user, password, host, database, port = 8123, native_port = 9000, untouchable_tables = []):
        '''
        untouchable_tables: list of names of tables, which can be read but not modified or dropped
        '''
        self.__user = user
        self.__password = password
        self.__host = host
        self.__port = port
        self.__database = database
        self.__native_port = native_port
        self.untouchable_tables = untouchable_tables
        
        self.__pandahouse_connection_dict = {'host': f'http://{user}:{password}@{host}:{port}', 'database': database}
        self.__native_connection_dict = {'host':host, 'user':user, 'password':password, 'port':native_port}  
    def select(self, q, info = False, max_load_info_time = 60):
        '''
        this function will unload the data of your select query to a dataframe
        '''
        from uuid import uuid4
        import time
        native_connection_dict = self.__native_connection_dict
        client = Client(**native_connection_dict)
        
        query_id = str(uuid4())
        results_df = client.query_dataframe(q, query_id = query_id)
        if info:
            print(f'query_id: {query_id}')
            start_time = time.time()
            while time.time() - start_time < max_load_info_time:
                query_info = client.query_dataframe(f'''
                    SELECT 
                        *
                    FROM system.query_log
                    WHERE query_id == '{query_id}'
                        AND type = 'QueryFinish'
                ''')
                
                if len(query_info) != 0:
                    query_duration_ms, read_rows, read_bytes, memory_usage = query_info[[col for col in query_info.columns if col in ['query_duration_ms','read_rows', 'read_bytes', 'memory_usage']]].loc[0].values
                    print('query_duration_ms: {}\nread_rows: {}\nread_bytes: {}\nmemory_usage: {}'.format(query_duration_ms, read_rows, read_bytes, memory_usage))
                    break
            if len(query_info) == 0: 
                print('the data did not have time to load')
        return results_df
    def execute(self, q):
        '''
        this function will execute your query, but if your query is a select expression, you will get the result of the expression as a list of tuples
        '''
        native_connection_dict = self.__native_connection_dict
        client = Client(**native_connection_dict)
        return client.execute(q)
    def create_table_by_select(self, q, table_name, engine, order_by, partition_by = None, database = None, print_results = False):
        '''
        q: select query for creating table
        engine: 'ReplacingMergeTree()', 'MergeTree()', 'SummingMergeTree()', 'AggregatingMergeTree()' as example
        partition_by: block PARTITION BY in DDL
        order_by: block ORDER BY in DDL
        '''
        partition_by_cond = '' if not partition_by else f'''PARTITION BY {partition_by}'''
        if not database: database = self.__database
        self.execute(f'''
        CREATE TABLE IF NOT EXISTS {database}.{table_name}
        ENGINE = {engine}
        {partition_by_cond}
        ORDER BY ({order_by})
        SETTINGS index_granularity = 8192 AS 
        {q}
        ''')
        if print_results == True: print('Table {} has been successfully created!'.format(table_name))
    def create_table(self, columns_dict, table_name, engine, order_by, partition_by = None, database = None, print_results = False):
        '''
        q: select query for creating table
        engine: 'ReplacingMergeTree()', 'MergeTree()', 'SummingMergeTree()', 'AggregatingMergeTree()' as example
        partition_by: block PARTITION BY in DDL
        order_by: block ORDER BY in DDL
        '''
        column_str_list = []
        columns_str = ''
        for key in columns_dict.keys():
            value = columns_dict[key]
            column_str = '''`{}` {}'''.format(key, value)
            column_str_list.append(column_str)
        columns_str = ',\n'.join(column_str_list)
        
        partition_by_cond = '' if not partition_by else f'''PARTITION BY {partition_by}'''
        if not database: database = self.__database
        self.execute(f'''
        CREATE TABLE IF NOT EXISTS {database}.{table_name}
        ({columns_str})
        ENGINE = {engine}
        {partition_by_cond}
        ORDER BY ({order_by})
        SETTINGS index_granularity = 8192 
        ''')
        if print_results == True: print('Table {} has been successfully created!'.format(table_name))
    def insert_select_to_db(self, q, table_name, database = None, print_results = False):
        '''
        this function will insert data from your select expression to table in db
        '''
        if not database: database = self.__database
        if table_name not in self.untouchable_tables:
            self.execute(f'''
            INSERT INTO {database}.{table_name}
            {q}
            ''')
            if print_results == True: print(f'Select data have been successfully writed to {database}{table_name}!')
        else:
            print(f'Not done! Table {table_name} in untochable_tables list!')
    def insert_df_to_db(self, df, table_name, database = None, print_results = False):
        '''
        this function will insert data from pandas dataframe to table in db
        '''
        if not database: database = self.__database
        if table_name not in self.untouchable_tables:
            connection = self.__pandahouse_connection_dict.copy()
            connection['database'] = database
            affected_rows = ph.to_clickhouse(df, table=table_name, connection=connection, index = False)
            if print_results == True: print(f'Dataframe data have been successfully writed to {table_name}!')
            return affected_rows
        else:
            print(f'Not done! Table {table_name} in untochable_tables list!')
    def drop_table(self, table_name, database = None, print_results = False): 
        if not database: database = self.__database
        if table_name in self.untouchable_tables:
            raise Exception(f'Not done! Table {database}.{table_name} in untochable_tables list!')         
        else:
            self.execute(f'''DROP TABLE IF EXISTS {database}.{table_name}''')
            if print_results == True: print(f'''Table {database}.{table_name} has been successfully dropped!''')
    def get_statistics_of_query(self, q, timeout = 15, done_limit = 0.25):
        '''
        return: tuple - result, done_progress
        result: True if query have been successfully ended, else False
        done_progress: part of total_rows processed
        timeout: maximum request execution time
        done_limit: the percentage of processed rows that must be processed during timeout for a successful result
        '''
        client = Client(**self.__native_connection_dict)
        progress = client.execute_with_progress(q)
        started_at = datetime.datetime.now()
        for num_rows, total_rows in progress:
            if total_rows:
                done = float(num_rows) / total_rows
            else:
                done = total_rows
            elapsed = (datetime.datetime.now() - started_at).total_seconds()
            # Cancel query if it takes more than {timeout} seconds
            # to process 50% of rows.
            if elapsed > timeout and done < done_limit:
                client.cancel()
                return False, done  
        else: 
            return True, done
    def get_table_info(self, table_name):
        df = self.select(f'''
        SELECT 
            name,
            engine,
            partition_key,
            create_table_query,
            sorting_key,
            primary_key,
            sampling_key
        FROM system.tables
        WHERE name = '{table_name}'
        ''')
        return df
    def get_all_tables(self):
        df = self.select(f'''
        SELECT 
            name,
            engine,
            partition_key,
            create_table_query,
            sorting_key,
            primary_key,
            sampling_key
        FROM system.tables
        ''')
        return df
    def get_database_metrics(self):
        df = self.select('''
        SELECT * FROM system.metrics
        ''')
        return df
    def get_queries_log(self, last_period_str = '5 minutes', limit = None):
        utc_now = datetime.datetime.utcnow() - datetime.timedelta(microseconds = datetime.datetime.utcnow().microsecond)
        start_time = utc_now - pd.Timedelta(last_period_str)
        if limit != None:
            limit_str = f'''LIMIT {limit}'''
        else:
            limit_str = ''
        df = self.select(f'''
        SELECT 
            toString(type) AS type,
            event_date,
            event_time,
            query_start_time,
            query_duration_ms,
            read_rows,
            read_bytes,
            written_rows,
            written_bytes,
            result_rows,
            result_bytes,
            memory_usage,
            query,
            query_id,
            exception,
            user
        FROM system.query_log
        WHERE event_time >= toDateTime('{start_time}')
        {limit_str}
        ''')
        return df
    def how_many_bytes_in_table(self, table_name):
        df = self.select(f'''
        SELECT 
            sum(data_compressed_bytes) AS compressed_bytes_total
        FROM system.columns
        WHERE table = '{table_name}'
        ''')
        return df['compressed_bytes_total'].min()
    def how_many_mb_in_table(self, table_name):
        df = self.select(f'''
        SELECT 
            sum(data_compressed_bytes) AS compressed_bytes_total
        FROM system.columns
        WHERE table = '{table_name}'
        ''')
        return np.round(df['compressed_bytes_total'].min() * 9.537 * (10**-7))
    def table_isin_db(self, table_name, database = None):
        if not database: database = self.__database
        df = self.select(f'''SELECT count() AS columns FROM system.columns WHERE table = '{table_name}' AND database = '{database}' ''')
        if df['columns'].min() == 0:
            return False
        else:
            return True