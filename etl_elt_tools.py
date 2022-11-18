#!/usr/bin/env python
# coding: utf-8

import datetime
import pandas as pd
import numpy as np
import inspect
from jinja2 import Template
import jinja2schema

# additional classes and functions
class DateFromStringParser:
    def get_date_from_string(self, string_value, pattern = '%Y-%m-%d'):
        return datetime.datetime.strptime(string_value, pattern).date()
    def get_date_from_string_or_date(self, value, pattern = '%Y-%m-%d'):
        if type(value) == datetime.date:
            return value
        else:
            return self.get_date_from_string(string_value = value)
class StringFromDateParser:
    def get_string_from_date(self, date_value, pattern = '%Y-%m-%d'):
        return date_value.strftime(pattern)
    def get_string_from_string_or_date(self, value, pattern = '%Y-%m-%d'):
        if type(value) == datetime.date:
            return self.get_string_from_date(value, pattern) 
        else:
            value = DateFromStringParser().get_date_from_string(value)
            return self.get_string_from_date(value, pattern) 
class GetFunctionArgs:
    def get_default_args(func):
        signature = inspect.signature(func)
        return {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
    def get_non_default_args(func):
        signature = inspect.signature(func)
        return {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is inspect.Parameter.empty
        }
    def get_args(func):
        signature = inspect.signature(func)
        return {
            k: v.default
            for k, v in signature.parameters.items()
        }
# base classes
class BaseConnection:
    connection_type: 'only_read'
    required_methods = dict(only_read = ['select'], admin = ['select', 'execute', 'create_table', 'create_table_by_select', 'drop_table', 'insert_df_to_db', 'insert_select_to_db', 'table_isin_db'])
    connection_types = required_methods.keys()
    def __init__(self):
        if self.connection_type not in self.connection_types: raise ValueError(f'connection_type property of the class object must be one of the values {self.connection_types}!')
        for method in self.required_methods[self.connection_type]:
            getattr(self, method)      
class BaseTable:
    schema: str = None
    table_name: str = None
    table_engine: str = None
    table_order_by: str = None
    table_partition_by: str = None
    table_columns_dict: str = None
    connection = None
    def __init__(self):
        if self.connection is None: raise ValueError('The connection property of the class object must not be None!')
        if self.connection.connection_type != 'admin': raise ValueError('''The connection property of the class object must be BaseConnection with connection_type == 'admin'!''')
        self.table_exists = self.connection.table_isin_db(self.table_name, database = self.schema)
        self.table_columns = self.table_columns_dict.keys()
    def table_isin_db(self):
        self.table_exists = self.connection.table_isin_db(self.table_name)
        return self.table_exists
    def recreate_table(self):
        if self.connection is None: raise ValueError('The connection property of the class object must not be None!')
        self.connection.drop_table(self.table_name, database = self.schema)
        self.connection.create_table(
            columns_dict = self.table_columns_dict,
            table_name = self.table_name,
            engine = self.table_engine,
            order_by = self.table_order_by,
            partition_by = self.table_partition_by,
            database = self.schema
        )
        print(f'Table {self.table_name} has been recreated!')
class BaseView:
    schema: str = None
    view_name: str = None
    q: str = None
    connection = None
    def __init__(self):
        if self.connection is None: raise ValueError('The connection property of the class object must not be None!')
        if self.connection.connection_type != 'admin': raise ValueError('''The connection property of the class object must be BaseConnection with connection_type == 'admin'!''')
        self.view_exists = self.connection.table_isin_db(self.view_name, database = self.schema)
    def view_isin_db(self):
        self.view_exists = self.connection.table_isin_db(self.view_name)
        return self.view_exists
    def recreate_view(self):
        if self.connection is None: raise ValueError('The connection property of the class object must not be None!')
        self.connection.execute(f'''
            DROP VIEW IF EXISTS {self.schema}.{self.view_name}
        ''')
        self.connection.execute(f'''
            CREATE VIEW IF NOT EXISTS {self.schema}.{self.view_name} AS {self.q}
        ''')
        print(f'View {self.view_name} has been recreated!')
class BaseSource:
    source_name: str = None
    required_methods = ['get']
    def __init__(self):
        for method in self.required_methods:
            getattr(self, method)  
class BaseQuery:
    query_name: str = None
    q: str = None
    query_args_schema: list = list()
    def __init__(self):
        self.query_args_schema = jinja2schema.infer(self.q)
    def render(self, query_args: dict = dict()):
        return Template(self.q).render(**query_args)
class BaseETL:
    source: BaseSource
    destination_table: BaseTable
    incremental_strategy = None
    incremental_strategy_condition = None
        
    source_args = dict()
    source_non_default_args = dict()
    destination_table_exists = None
    destination_table_name = None
    today = datetime.datetime.utcnow().date()
    yesterday = today - datetime.timedelta(days = 1)
    def __init__(self):
        self.connection = self.destination_table.connection
        self.destination_table_exists = self.destination_table.table_exists
        self.destination_table_name = self.destination_table.table_name
        self.destination_table_schema = self.destination_table.schema
        self.source_args = GetFunctionArgs.get_args(self.source.get)
        self.source_non_default_args = GetFunctionArgs.get_non_default_args(self.source.get)
    def get_date_range(self, start_date, finish_date = datetime.datetime.utcnow().date() - datetime.timedelta(days = 1), freq = '1D'):
        return pd.Series(pd.date_range(start = start_date, end = finish_date, freq = freq)).dt.date
    def run(self, source_args: dict = dict()):
        '''
        :param source_args: dict - to see the expected arguments use property source_args of ETL class
        '''
        if self.source_non_default_args != dict() and source_args == dict(): raise ValueError(f'There are empty non-default arguments of the function in source_get_args! {self.source_non_default_args.keys()}') # check args for get method
        if not self.destination_table.table_exists: self.destination_table.recreate_table() # create table if not exists
        print(f'incremental_strategy: {self.incremental_strategy}')
        print(f'incremental_strategy_condition: {self.incremental_strategy_condition}')
        if not self.incremental_strategy or self.incremental_strategy_condition:
            data = self.source.get(**self.source_args) if not source_args else self.source.get(**source_args) # unload data
            try:
                self.connection.insert_df_to_db(data, table_name = self.destination_table.table_name, database = self.destination_table.schema) # insert data
            except BaseException as e:
                raise BaseException(e)           
class BaseELT:
    query = BaseQuery
    destination_table: BaseTable
    connection = BaseConnection
    incremental_strategy = None
    incremental_strategy_condition = None
    
    query_args = dict()
    destination_table_exists = None
    destination_table_name = None
    today = datetime.datetime.utcnow().date()
    yesterday = today - datetime.timedelta(days = 1)
    def __init__(self):
        self.destination_table_exists = self.destination_table.table_exists
        self.destination_table_name = self.destination_table.table_name
        self.destination_table_schema = self.destination_table.schema
        self.query_args = jinja2schema.infer(self.query.q)
    def get_date_range(self, start_date, finish_date = datetime.datetime.utcnow().date() - datetime.timedelta(days = 1), freq = '1D'):
        return pd.Series(pd.date_range(start = start_date, end = finish_date, freq = freq)).dt.date
    def run(self, query_args: dict = dict()):
        '''
        :param source_args: dict - to see the expected arguments use property source_args of ELT class
        '''
        if not self.destination_table.table_exists: self.destination_table.recreate_table() # create table if not exists
        print(f'incremental_strategy: {self.incremental_strategy}')
        print(f'incremental_strategy_condition: {self.incremental_strategy_condition}')
        if not self.incremental_strategy or self.incremental_strategy_condition:
            q = Template(self.query.q).render(**query_args)
            self.connection.insert_select_to_db(q, table_name = self.destination_table_name, database = self.destination_table_schema)