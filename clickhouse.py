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

class clickhouse_tools:
    '''
    Your query must contains in block "where" condition by time column with variables $from and $to. "WHERE event_time BETWEEN $from AND $to" as example.
    '''
    elt_jobs_table_name = 'elt_jobs'
    elt_jobs_progress_table_name = 'elt_progress'
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
    ###### common functions
    def select(self,q):
        '''
        this function will unload the data of your select query to a dataframe
        '''
        connection = self.__pandahouse_connection_dict
        request = ph.read_clickhouse(query=q, connection=connection)
        return request   
    def execute(self, q):
        '''
        this function will execute your query, but if your query is a select expression, you will get the result of the expression as a list of tuples
        '''
        native_connection_dict = self.__native_connection_dict
        client = Client(**native_connection_dict)
        return client.execute(q)
    def create_table(self, columns_dict, table_name, engine, order_by, partition_by = None, print_results = False):
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
        
        if partition_by == None:
            partition_by_cond = ''
        else:
            partition_by_cond = f'''PARTITION BY {partition_by}'''
        self.execute(f'''
        CREATE TABLE IF NOT EXISTS {self.__database}.{table_name}
        ({columns_str})
        ENGINE = {engine}
        {partition_by_cond}
        ORDER BY ({order_by})
        SETTINGS index_granularity = 8192 
        ''')
    def create_table_by_select(self, q, table_name, engine, partition_by, order_by, print_results = False):
        '''
        q: select query for creating table
        engine: 'ReplacingMergeTree()', 'MergeTree()', 'SummingMergeTree()', 'AggregatingMergeTree()' as example
        partition_by: block PARTITION BY in DDL
        order_by: block ORDER BY in DDL
        '''
        if partition_by == None:
            partition_by_cond = ''
        else:
            partition_by_cond = f'''PARTITION BY {partition_by}'''
        self.execute(f'''
        CREATE TABLE IF NOT EXISTS {self.__database}.{table_name}
        ENGINE = {engine}
        {partition_by_cond}
        ORDER BY ({order_by})
        SETTINGS index_granularity = 8192 AS 
        {q}
        ''')
        if print_results == True: print('Table {} have been successfully created!'.format(table_name))
    def insert_select_to_db(self, q, table_name, print_results = False):
        '''
        this function will insert data from your select expression to table in db
        '''
        if table_name not in self.untouchable_tables:
            self.execute(f'''
            INSERT INTO {self.__database}.{table_name}
            {q}
            ''')
            if print_results == True: rint(f'Select data have been successfully writed to {table_name}!')
        else:
            print(f'Not done! Table {table_name} in untochable_tables list!')
    def insert_df_to_db(self, df, table_name, print_results = False):
        '''
        this function will insert data from pandas dataframe to table in db
        '''
        if table_name not in self.untouchable_tables:
            connection = self.__pandahouse_connection_dict
            affected_rows = ph.to_clickhouse(df, table=table_name, connection=connection, index = False)
            if print_results == True: print(f'Dataframe data have been successfully writed to {table_name}!')
            return affected_rows
        else:
            print(f'Not done! Table {table_name} in untochable_tables list!')
    def drop_table(self, table_name, print_results = False): 
        if table_name not in self.untouchable_tables:
            self.execute(f'''DROP TABLE IF EXISTS {self.__database}.{table_name}''')
            if print_results == True: print(f'''\t - table {table_name} have been successfully dropped!''')
        else:
            print(f'\t - not done! table {table_name} in untochable_tables list!')         
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
    def table_isin_db(self, table_name):
        df = self.select(f'''SELECT count() AS columns FROM system.columns WHERE table = '{table_name}' ''')
        if df['columns'].min() == 0:
            return False
        else:
            return True
    ###### elt functions
    # recreate main tables
    def elt_recreate_jobs_table(self, password, print_results = False):
        '''
        password: connection password
        the password is needed so that it is impossible to accidentally clear the jobs data table
        '''
        if password == self.__password:
            self.drop_table(clickhouse_tools.elt_jobs_table_name)
            self.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.__database}.{clickhouse_tools.elt_jobs_table_name}
            (
            `created_at` DateTime,
            `table_name` String,
            `engine` String,
            `partition_by` String,
            `order_by` String,
            `since_date` Date,
            `min_writing_interval_cron` String,
            `delay_string` String,
            `query` String,
            `relations` String
            )
            ENGINE = ReplacingMergeTree()
            ORDER BY (created_at,table_name)
            SETTINGS index_granularity = 8192
            ''')
            if print_results == True: print(f'{clickhouse_tools.elt_jobs_table_name} table have been recreated!')  
        else:
            print('Wrong password! You needed jdbc connection password!')
    def elt_recreate_jobs_progress_table(self, password, print_results = False):
        '''
        password: connection password
        the password is needed so that it is impossible to accidentally clear the jobs progress table
        '''
        if password == self.__password:
            self.drop_table(clickhouse_tools.elt_jobs_progress_table_name)
            self.execute(f'''
            CREATE TABLE IF NOT EXISTS {self.__database}.{clickhouse_tools.elt_jobs_progress_table_name}
            (
            `start_time` DateTime,
            `finish_time` DateTime,
            `table_name` String,
            `insert_duration` Float64
            )
            ENGINE = ReplacingMergeTree()
            ORDER BY (start_time,table_name)
            SETTINGS index_granularity = 8192
            ''')
            if print_results == True: print(f'{clickhouse_tools.elt_jobs_progress_table_name} table have been recreated!')
        else:
            return
    def elt_create_main_tables_if_not_exists(self, print_results = False):
        if self.table_isin_db(clickhouse_tools.elt_jobs_table_name) == False:
            self.elt_recreate_jobs_table(password = self.__password, print_results = print_results)
        else:
            if print_results == True: print('Table {} already exists!'.format(clickhouse_tools.elt_jobs_table_name))
        if self.table_isin_db(clickhouse_tools.elt_jobs_progress_table_name) == False:
            self.elt_recreate_jobs_progress_table(password = self.__password, print_results = print_results)
        else:
            if print_results == True: print('Table {} already exists!'.format(clickhouse_tools.elt_jobs_progress_table_name))
        return 
    # add-delete job
    def elt_add_job(self, q, table_name, since_date_str, engine, order_by, partition_by = None, min_writing_interval_cron_str = '0 0 * * *', delay_string = None):
        '''
        delay_string: value for pd.Timedelta object
        '''
        if '$from' not in q or '$to' not in q:
            print('Your query must contains $from and $to variables in WHERE block by time column for job working.')
            return 
        if delay_string == None:
            delay_string = ''
        if table_name in self.untouchable_tables:
            print(f'Not done! Table {table_name} in untochable_tables list!')
            return 
        else:
            if self.elt_get_job_by_table_name(table_name).shape[0] == 0:
                utcnow = datetime.datetime.utcnow() - datetime.timedelta(microseconds = datetime.datetime.utcnow().microsecond)
                job_row_df = pd.DataFrame({
                    'created_at':[utcnow],
                    'table_name':[table_name],
                    'engine':[engine],
                    'partition_by':[partition_by],
                    'order_by':[order_by],
                    'since_date':[since_date_str],
                    'min_writing_interval_cron':[min_writing_interval_cron_str],
                    'delay_string':[delay_string],
                    'query':[q],
                    'relations':[', '.join(set(re.findall(self.__database + '.([A-z0-9]+)',q)))]
                })
                self.insert_df_to_db(job_row_df, clickhouse_tools.elt_jobs_table_name)
                print(f'Done! Job with table {table_name} have been added!')
            else:
                print(f'Not done! Job with table {table_name} already exists!')
    def elt_delete_job(self, table_name, print_results = False):
        if self.elt_get_all_jobs().query('table_name==@table_name').shape[0] != 0:
            df = self.elt_get_all_jobs().query('table_name!=@table_name')
            df['since_date'] = df['since_date'].dt.strftime('%Y-%m-%d')
            self.elt_recreate_jobs_table(password = self.__password)
            self.insert_df_to_db(df,clickhouse_tools.elt_jobs_table_name)
            if print_results == True: print(f'Job {table_name} have been deleted!')
        else:
            if print_results == True: print(f'Job {table_name} does not exist!')
    def elt_delete_job_progress_with_table(self, table_name, print_results = False):
        self.drop_table(table_name, print_results = print_results)
        df = self.elt_get_all_jobs_progress().query('table_name!=@table_name')
        self.elt_recreate_jobs_progress_table(password = self.__password)
        self.insert_df_to_db(df,clickhouse_tools.elt_jobs_progress_table_name)
        if print_results == True: print(f'Progress of job with {table_name} have been deleted!')
    def elt_delete_job_with_table(self, table_name, print_results = False):
        self.elt_delete_job(table_name, print_results = print_results)
        self.drop_table(table_name, print_results = print_results)
        df = self.elt_get_all_jobs_progress().query('table_name!=@table_name')
        self.elt_recreate_jobs_progress_table(password = self.__password)
        self.insert_df_to_db(df,clickhouse_tools.elt_jobs_progress_table_name)
        if print_results == True: print(f'Job with table {table_name} have been deleted!')
    # get job/jobs or job process tables
    def elt_get_all_jobs(self):
        df = self.select(f'''
        SELECT
            *
        FROM {self.__database}.{clickhouse_tools.elt_jobs_table_name}
        ORDER BY 
            created_at
        ''')
        return df
    def elt_get_all_jobs_progress(self):
        df = self.select(f'''
        SELECT
            *
        FROM {self.__database}.{clickhouse_tools.elt_jobs_progress_table_name}
        ORDER BY 
            table_name, start_time, finish_time
        ''')
        return df
    def elt_get_job_by_table_name(self, table_name):
        df = self.select(f'''
        SELECT
            *
        FROM {self.__database}.{clickhouse_tools.elt_jobs_table_name}
        WHERE table_name = '{table_name}'
        ''')
        return df
    def elt_get_job_progress_by_table_name(self, table_name):
        df = self.select(f'''
        SELECT
            *
        FROM {self.__database}.{clickhouse_tools.elt_jobs_progress_table_name}
        WHERE table_name = '{table_name}'
        ''')
        return df
    def elt_get_job_query_by_table_name(self, table_name, print_query = True):
        df = self.select(f'''
        SELECT
            *
        FROM {self.__database}.{clickhouse_tools.elt_jobs_table_name}
        WHERE table_name = '{table_name}'
        ''')
        if print_query == True: print(df.iloc[0]['query'])
        return df.iloc[0]['query']
    # get datetime batches
    def __datetime_from_text(self, text, pattern = '%Y-%m-%d %H:%M:%S'):
        '''
        default pattern = '%Y-%m-%d %H:%M:%S'
        '''
        return datetime.datetime.strptime(text, pattern)
    def __get_datetimes_by_cron(self, start_time, finish_time, cron_str = '0 * * * *'):
        '''
        cron_str: default - every hour at 0 minutes
        cron description: {minute} {hour} {day(month)} {month} {day(week)}
        return: datetimes list

        *** start_time included, finish_time excluded
        '''
        cron = croniter.croniter(cron_str, start_time)
        cron.get_prev(datetime.datetime)
        stop = False
        datetime_list = []
        while stop != True:
            next_datetime = cron.get_next(datetime.datetime)
            if next_datetime < finish_time and next_datetime >= start_time: datetime_list.append(next_datetime)
            if next_datetime >= finish_time: stop = True
        return datetime_list
    def __get_finish_time_current_interval_by_cron(self, start_time, cron_str = '0 * * * *'):
        '''
        cron_str: default - every hour at 0 minutes
        cron description: {minute} {hour} {day(month)} {month} {day(week)}
        return: finish datetime current interval
        '''
        cron = croniter.croniter(cron_str, start_time)
        next_datetime = cron.get_next(datetime.datetime)
        finish_time_current_interval = next_datetime - datetime.timedelta(seconds = 1)
        return finish_time_current_interval
    def __get_datetimes_periods_by_cron(self, start_time, finish_time, cron_str = '0 * * * *'):
        start_datetimes_list = self.__get_datetimes_by_cron(start_time,finish_time, cron_str = cron_str)
        datetimes_df = pd.DataFrame()
        datetimes_df['start_time'] = pd.Series(start_datetimes_list)
        datetimes_df['finish_time'] = datetimes_df['start_time'].apply(lambda x: self.__get_finish_time_current_interval_by_cron(x, cron_str = cron_str))
        return datetimes_df[datetimes_df['finish_time']<finish_time]
    def __get_datetimes_by_split(self, datetimes_df, start_column = 'start_time', finish_column = 'finish_time'):
        '''
        This function get tuple of start and finish datetimes by reducing initial df.
        return: start_time, finish_time
        start_time: first start_time in df
        finish_time: less than max finish_time in df
        '''
        if datetimes_df.shape[0] == 1:
            return datetimes_df.iloc[0][start_column], datetimes_df.iloc[0][finish_column]
        else:
            if datetimes_df.shape[0] >= 500:
                split_coef = 4
            else:
                split_coef = 2   
            split_iloc = datetimes_df.shape[0] // split_coef
            last_iloc = datetimes_df.shape[0] - 1
            start_time = datetimes_df.iloc[0][start_column]
            finish_time = datetimes_df.iloc[split_iloc-1][finish_column]
            return start_time, finish_time
    def __get_statistics_of_query(self, q, start_time, finish_time, timeout = 15, done_limit = 0.25):
        '''
        return: result, done_progress
        result: True if query have been successfully ended, else False
        done_progress: part of total_rows processed
        '''
        start_time = f'''toDateTime('{start_time}')'''
        finish_time = f'''toDateTime('{finish_time}')'''
        client = Client(**self.__native_connection_dict)
        settings = {'max_block_size': 100000}
        progress = client.execute_with_progress(q.replace('$from', start_time).replace('$to',finish_time), settings = settings)
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
    def __get_optimal_datetime_batches(self, q, start_time, finish_time, datetimes_df, datetime_batches_for_populate_df, print_results = False):
        '''
        start_time and finish_time: must be in datetimes_df
        datetime_batches_for_populate_df: empty dataframe for results
        '''
        ### imports
        
        ### checking variables
        if start_time not in datetimes_df['start_time'].to_list() or finish_time not in datetimes_df['finish_time'].to_list():
            print('Error! start_time and finish_time must be in datetimes_df!')
            return 
        ### additional variables
        datetime_batches_for_populate_row_df = pd.DataFrame({'start_time':[start_time], 'finish_time':[finish_time]})
        current_batches = datetimes_df.query('start_time>=@start_time and finish_time<=@finish_time').shape[0]
        datetimes_current_df = datetimes_df.query('start_time>=@start_time and finish_time<=@finish_time')
        datetimes_continue_df = datetimes_df.query('start_time>@finish_time')
        ### get statistics of query
        try:
            success, _ = self.__get_statistics_of_query(q, start_time, finish_time)
            if print_results == True: print('\t\t - {} - {}: success={}, min_intervals={}'.format(start_time, finish_time, success, current_batches))
        except Exception as e:
            # print(e)
            time.sleep(15)
            try:
                success, _ = self.__get_statistics_of_query(q, start_time, finish_time)
                if print_results == True: print('\t\t - {} - {}: success={}, min_intervals={}'.format(start_time, finish_time, success, current_batches))
            except Exception as e:
                # print(e)
                time.sleep(15)
                success, _ = self.__get_statistics_of_query(q, start_time, finish_time)
                if print_results == True: print('\t\t - {} - {}: success={}, min_intervals={}'.format(start_time, finish_time, success, current_batches))
        ### recursion cases
        if success == True:
            datetime_batches_for_populate_row_df['success'] = True
            datetime_batches_for_populate_row_df['batches'] = current_batches
            datetime_batches_for_populate_df = pd.concat([datetime_batches_for_populate_df, datetime_batches_for_populate_row_df])
            not_loaded_batches = datetime_batches_for_populate_df.query('batches==1 and success==False').shape[0]
            success_steps_pct = datetime_batches_for_populate_df['success'].mean()

            if finish_time == datetimes_df['finish_time'].max():
                if print_results == True: print('\t\t - search optimal batches is completed! {:.2%} success steps! {} not loaded batches!'.format(success_steps_pct,not_loaded_batches))
                return datetime_batches_for_populate_df#.query('success==True or batches==1')
            else:
                start_time = datetimes_continue_df['start_time'].min()
                successes_in_last_3_cases = datetime_batches_for_populate_df.tail(3).query('success==True').shape[0]
                last_success_batches = int(datetime_batches_for_populate_df.query('success==True')[-1:]['batches'].max())
                if successes_in_last_3_cases == 3:  
                    growth_coef = 1.2
                else: 
                    growth_coef = 1
                next_attempt_batches = int(last_success_batches * growth_coef)
                finish_time = datetimes_continue_df[:next_attempt_batches]['finish_time'].max()
                return self.__get_optimal_datetime_batches(q, start_time, finish_time, datetimes_continue_df, datetime_batches_for_populate_df, print_results = print_results)
        else:
            datetime_batches_for_populate_row_df['success'] = False
            datetime_batches_for_populate_row_df['batches'] = current_batches
            datetime_batches_for_populate_df = pd.concat([datetime_batches_for_populate_df, datetime_batches_for_populate_row_df])
            not_loaded_batches = datetime_batches_for_populate_df.query('batches==1 and success==False').shape[0]
            success_steps_pct = datetime_batches_for_populate_df['success'].mean()

            if current_batches == 1:
                if finish_time == datetimes_df['finish_time'].max():
                    if print_results == True: print('\tSearch optimal batches is completed! {:.2%} success steps! {} not loaded batches!'.format(success_steps_pct,not_loaded_batches))
                    return datetime_batches_for_populate_df#.query('success==True or batches==1')
                else:
                    start_time = datetimes_continue_df['start_time'].min()
                    finish_time = datetimes_continue_df['finish_time'].max()
                    return self.__get_optimal_datetime_batches(q, start_time, finish_time, datetimes_continue_df, datetime_batches_for_populate_df, print_results = print_results)
            else:
                successes_in_prev_3_cases = datetime_batches_for_populate_df[-4:-1].query('success==True').shape[0]
                failure_in_prev_case = datetime_batches_for_populate_df[-2:-1].query('success==False').shape[0]
                successes_in_all_prev_cases = datetime_batches_for_populate_df.query('success==True').shape[0]
                if successes_in_prev_3_cases == 3:
                    last_success_batches = int(datetime_batches_for_populate_df.query('success==True')[-1:]['batches'].max())
                    last_attempt_batches = int(datetime_batches_for_populate_df[-1:]['batches'].max())
                    start_time = datetimes_current_df['start_time'].min()
                    next_attempt_batches = last_success_batches
                    finish_time = datetimes_df[:next_attempt_batches]['finish_time'].max()
                    return self.__get_optimal_datetime_batches(q, start_time, finish_time, datetimes_df, datetime_batches_for_populate_df, print_results = print_results)
                elif successes_in_all_prev_cases != 0:
                    last_attempt_batches = int(datetime_batches_for_populate_df[-1:]['batches'].max())
                    start_time = datetimes_current_df['start_time'].min()
                    next_attempt_batches = int(last_attempt_batches // 1.5)
                    finish_time = datetimes_df[:next_attempt_batches]['finish_time'].max()
                    return self.__get_optimal_datetime_batches(q, start_time, finish_time, datetimes_df, datetime_batches_for_populate_df, print_results = print_results)
                else:
                    start_time, finish_time = self.__get_datetimes_by_split(datetimes_current_df)
                return self.__get_optimal_datetime_batches(q, start_time, finish_time, datetimes_df, datetime_batches_for_populate_df, print_results = print_results)
    # populate table
    def __elt_get_batches_for_populate_table(self, table_name, print_results = False):
        job_row_df = self.elt_get_job_by_table_name(table_name)
        q = job_row_df['query'].min()
        since_date = job_row_df['since_date'].min()
        min_writing_interval_cron_str = job_row_df['min_writing_interval_cron'].min()
        delay_string = job_row_df['delay_string'].min()
        
        start_time = since_date
        utc_now = datetime.datetime.utcnow() - datetime.timedelta(microseconds = datetime.datetime.utcnow().microsecond)
        if delay_string != '':
            finish_time = utc_now - pd.Timedelta(delay_string)
        else:
            finish_time = utc_now
        datetimes_populate_df = self.__get_datetimes_periods_by_cron(start_time, finish_time, cron_str = min_writing_interval_cron_str)

        datetime_batches_for_populate_df = pd.DataFrame()
        start_time = datetimes_populate_df['start_time'].min()
        finish_time = datetimes_populate_df['finish_time'].max()
        datetime_batches_for_populate_df = self.__get_optimal_datetime_batches(q, start_time, finish_time, datetimes_populate_df, datetime_batches_for_populate_df, print_results = print_results)
        datetime_batches_for_populate_df['table_name'] = table_name
        return datetime_batches_for_populate_df 
    def elt_populate_table(self, table_name, print_results = False):
        job_row_df = self.elt_get_job_by_table_name(table_name)
        q = job_row_df['query'].min()
        engine = job_row_df['engine'].min()
        if job_row_df['partition_by'].min() == '':
            partition_by = None
        else:
            partition_by = job_row_df['partition_by'].min()
        order_by = job_row_df['order_by'].min()
        
        if print_results == True: print('\t - search optimal batches for populate table:')
        batches_df = self.__elt_get_batches_for_populate_table(table_name, print_results = print_results)
        if batches_df.shape[0] == 0:
            if print_results == True: print('\t - create and populate table:')
            if print_results == True: print(f'\t\t - job with table {table_name} has no data yet!')
            return
        not_loaded_batches = batches_df.query('batches==1 and success==False').shape[0]
        batches_for_populate_df = batches_df.query('success==True').reset_index()[['start_time','finish_time','table_name']]
        if not_loaded_batches == 0:
            for index, row in batches_for_populate_df.iterrows():
                start_time = f'''toDateTime('{row['start_time']}')'''
                finish_time = f'''toDateTime('{row['finish_time']}')'''
                batch_q = f'''
                {q.replace('$from', start_time).replace('$to',finish_time)}
                '''                       
                if index == 0:
                    if self.table_isin_db(table_name) == False:
                        if print_results == True: print('\t - create and populate table:')
                        execution_start_time = datetime.datetime.utcnow()
                        self.create_table_by_select(batch_q, table_name, engine, partition_by, order_by)
                        execution_timedelta = datetime.datetime.utcnow() - execution_start_time
                        execution_seconds = execution_timedelta.seconds
                        batches_for_populate_df.loc[index, 'insert_duration'] = execution_seconds
                        if print_results == True: print(f'\t\t - table {table_name} have been created!')
                        if print_results == True: print(f'''\t\t - batch {row['start_time']} - {row['finish_time']} have been successfully inserted in {table_name}!''')
                        continue
                    else:
                        if print_results == True: print('\t - create and populate table:')
                        if self.how_many_bytes_in_table(table_name) == 0:
                            execution_start_time = datetime.datetime.utcnow()
                            self.insert_select_to_db(q = batch_q, table_name = table_name)
                            execution_timedelta = datetime.datetime.utcnow() - execution_start_time
                            execution_seconds = execution_timedelta.seconds
                            batches_for_populate_df.loc[index, 'insert_duration'] = execution_seconds
                            if print_results== True: print(f'''\t\t - batch {row['start_time']} - {row['finish_time']} have been successfully inserted in {table_name}!''')
                            continue
                        else:
                            print(f'\t\t - not done! Table {table_name} already exists and has data!')
                            return
                else:
                    execution_start_time = datetime.datetime.utcnow()
                    self.insert_select_to_db(q = batch_q, table_name = table_name)
                    execution_timedelta = datetime.datetime.utcnow() - execution_start_time
                    execution_seconds = execution_timedelta.seconds
                    batches_for_populate_df.loc[index, 'insert_duration'] = execution_seconds
                    if print_results== True: print(f'''\t\t - batch {row['start_time']} - {row['finish_time']} have been successfully inserted in {table_name}!''')
            self.insert_df_to_db(batches_for_populate_df, clickhouse_tools.elt_jobs_progress_table_name)
            print(f'''\t\t - table {table_name} have been successfully populated!''')
            return True
        else:
            print(f'\t\t - change your min_writing_interval_cron for table {table_name}! min_writing_interval_cron is too big for loading!')
            return False
    # update table
    def elt_update_table(self, table_name, print_results = False):
        job_row_df = self.elt_get_job_by_table_name(table_name)
        q = job_row_df['query'].min()
        min_writing_interval_cron_str = job_row_df['min_writing_interval_cron'].min()
        delay_string = job_row_df['delay_string'].min()
        
        isin_progress_table = False
        progress_df = self.elt_get_job_progress_by_table_name(table_name)
        if progress_df.shape[0] != 0: isin_progress_table = True
        if print_results == True: print('\t - update table:')
        if isin_progress_table == True:
            last_time_updated = progress_df['finish_time'].max()
            utc_now = datetime.datetime.utcnow() - datetime.timedelta(microseconds = datetime.datetime.utcnow().microsecond)
            
            start_time = last_time_updated
            if delay_string != '':
                finish_time = utc_now - pd.Timedelta(delay_string)
            else:
                finish_time = utc_now
            update_datetimes_df = self.__get_datetimes_periods_by_cron(start_time, finish_time, cron_str = min_writing_interval_cron_str)
            if update_datetimes_df.shape[0] != 0:
                batches_df = pd.DataFrame()
                start_time = update_datetimes_df['start_time'].min()
                finish_time = update_datetimes_df['finish_time'].max()
                batches_df = self.__get_optimal_datetime_batches(q, start_time, finish_time, update_datetimes_df, batches_df)
                
                not_loaded_batches = batches_df.query('batches==1 and success==False').shape[0]
                batches_for_populate_df = batches_df.query('success==True').reset_index()[['start_time','finish_time']]
                batches_for_populate_df['table_name'] = table_name
                if not_loaded_batches == 0:
                    for index, row in batches_for_populate_df.iterrows():
                        start_time = f'''toDateTime('{row['start_time']}')'''
                        finish_time = f'''toDateTime('{row['finish_time']}')'''
                        batch_q = f'''
                        {q.replace('$from', start_time).replace('$to',finish_time)}
                        '''
                        execution_start_time = datetime.datetime.utcnow()
                        self.insert_select_to_db(q = batch_q, table_name = table_name)
                        execution_timedelta = datetime.datetime.utcnow() - execution_start_time
                        execution_seconds = execution_timedelta.seconds
                        batches_for_populate_df.loc[index, 'insert_duration'] = execution_seconds
                        if print_results== True: print(f'''\t\t - batch {row['start_time']} - {row['finish_time']} have been successfully inserted in {table_name}!''')
                    self.insert_df_to_db(batches_for_populate_df, clickhouse_tools.elt_jobs_progress_table_name)
                    print(f'''\t\t - table {table_name} have been successfully updated!''')
                    return True
                else:
                    print(f'\t\t - error! Change your min_writing_interval_cron for table {table_name}! min_writing_interval_cron is too big for loading!')
                    return False 
            else:
                print(f'\t\t - table {table_name} already have been updated!')
                return
        else:
            print(f'\t\t - error! Table {table_name} is not populated yet!')
            return 
    # run job/jobs
    def elt_run_job(self, table_name, print_results = False):
        print(f' - table {table_name}:')
        table_in_bd = self.table_isin_db(table_name)
        bytes_in_table = self.how_many_bytes_in_table(table_name)
        
        table_in_jobs = False
        if self.elt_get_job_by_table_name(table_name).shape[0] != 0: table_in_jobs = True
        
        table_in_jobs_progress = False
        if self.elt_get_job_progress_by_table_name(table_name).shape[0] != 0: table_in_jobs_progress = True
        
        # run
        if table_in_jobs == True:
            if table_in_jobs_progress == False:
                if table_in_bd == True and bytes_in_table != 0:
                    print(f'\t - not done! Table {table_name} already exists and has data, but there is no data about inserts!')
                    return 
                else:
                    self.elt_populate_table(table_name, print_results = print_results)
            else:
                if table_in_bd == True:
                    self.elt_update_table(table_name, print_results = print_results)
                else:
                    print(f'\t - there is no such table {table_name}, but there is data about inserts. It is recommended to use the method "self.elt_delete_job_with_table" to delete job data. After that, you can recreate the job.')
        else:
            print(f'\t - table {table_name} not in jobs!')
    def elt_run_all_jobs(self, print_results = False):
        all_jobs_df = self.elt_get_all_jobs()
        for table_name in set(all_jobs_df['table_name']):
            self.elt_run_job(table_name, print_results = print_results)
        print(' - all {} jobs have been successfully completed!'.format(all_jobs_df.shape[0]))
    def elt_run_overwriting_job(self, table_name, print_results = False):
        progress_df = self.elt_get_job_progress_by_table_name(table_name)
        job_row_df = self.elt_get_job_by_table_name(table_name)
        delay_string = job_row_df['delay_string'].min()
        min_writing_interval_cron_str = job_row_df['min_writing_interval_cron'].min()
        
        if print_results == True: print(f' - job with overwrite table {table_name}:')
        if self.table_isin_db(table_name) == False:
            df = self.elt_get_all_jobs_progress().query('table_name!=@table_name')
            self.elt_recreate_jobs_progress_table(password = self.__password)
            self.insert_df_to_db(df,clickhouse_tools.elt_jobs_progress_table_name)
            self.elt_run_job(table_name, print_results = print_results)
        else:
            if progress_df.shape[0] != 0:
                last_time_updated = progress_df['finish_time'].max()
                utc_now = datetime.datetime.utcnow() - datetime.timedelta(microseconds = datetime.datetime.utcnow().microsecond)

                start_time = last_time_updated
                if delay_string != '':
                    finish_time = utc_now - pd.Timedelta(delay_string)
                else:
                    finish_time = utc_now
                update_datetimes_df = self.__get_datetimes_periods_by_cron(start_time, finish_time, cron_str = min_writing_interval_cron_str)
                if update_datetimes_df.shape[0] != 0:
                    self.drop_table(table_name, print_results = print_results)
                    df = self.elt_get_all_jobs_progress().query('table_name!=@table_name')
                    self.elt_recreate_jobs_progress_table(password = self.__password)
                    self.insert_df_to_db(df,clickhouse_tools.elt_jobs_progress_table_name)
                    self.elt_run_job(table_name, print_results = print_results)
                else:
                    print(f'\t\t - job with table {table_name} already have been re-created!')
            else:
                self.drop_table(table_name, print_results = print_results)
                self.elt_run_job(table_name, print_results = print_results)
                print(f'\t\t - job with table {table_name} have been successfully re-created!')