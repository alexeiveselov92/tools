#!/usr/bin/env python
# coding: utf-8

from jinja2 import Template # jinja2schema==0.1.4
import jinja2schema
class QueryFunc:
    def select_function(self, rendered_query):
        return kiss.select(rendered_query)
class QueryRendered:
    def render(self, query_params:dict = dict()):
        return Template(self.query).render(**query_params)
class Queries:
    data = pd.DataFrame()
class Query(QueryFunc, QueryRendered, Queries):
    q: str = ''
    def __init__(self, query: str, query_name: str = None):
        self.query = query
        self.query_name = query_name if query_name is not None else 'no_name'
        self.query_params = list(jinja2schema.infer(query).keys())
        self.df = pd.DataFrame({'query':[query], 'query_name':[self.query_name], 'query_params':[self.query_params]})
        Queries.data = pd.concat([Queries.data, self.df]).drop_duplicates(subset = ['query', 'query_name'])
    def select(self, query_params: dict = dict()):
        if len(set(query_params) ^ set(self.query_params)) > 0:
            raise ValueError(f'Missing variables: {set(query_params) ^ set(self.query_params)}')
        return self.select_function(self.render(query_params))