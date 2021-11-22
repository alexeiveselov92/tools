#!/usr/bin/env python
# coding: utf-8

import gspread
import httplib2
import pandas as pd
from googleapiclient import discovery
from oauth2client.service_account import ServiceAccountCredentials
import matplotlib.colors as colors
from gspread.utils import rowcol_to_a1, a1_to_rowcol
class google_sheets_tools:
    '''
    path_to_credential - path to json key you received when creating a service account for google sheets
    '''
    header_format_default_dict = {
        'background_color':'#2b6ca3',
        'text_color':'white',
        'font_size':8,
        'font_bold':True,
        'horizontal_alignment':'CENTER',
        'vertical_alignment':'MIDDLE',
        'borders_style':'DOTTED',
        'borders_color':'black'
    }
    body_format_default_dict = {
        'background_color':'white',
        'text_color':'black',
        'font_size':8,
        'font_bold':False,
        'borders_style':'DOTTED',
        'borders_color':'black'
    }
    def __init__(self, path_to_credential):
        scope = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
        credentials = ServiceAccountCredentials.from_json_keyfile_name(path_to_credential, scope)
        # google api v4
        httpAuth = credentials.authorize(httplib2.Http())
        self.__service = discovery.build('sheets', 'v4', http = httpAuth)
        # gspread
        self.__gs = gspread.authorize(credentials)
    # common operations
    def create_spreadsheet(self, new_spreadsheet_name, list_of_editors_emails):
        spreadsheet = self.__gs.create(new_spreadsheet_name)
        print('- Table "{}" have been created successfully!'.format(new_spreadsheet_name))
        link = f'https://docs.google.com/spreadsheets/d/{spreadsheet.id}/' 
        print(f"- Table's link: {link}")
        for email in list_of_editors_emails:
            try:
                spreadsheet.share(email, perm_type='user', role='writer')
                print('- Table "{}" have been shared successfully to {}!'.format(new_spreadsheet_name, email))
            except Exception as e: print(f'- Email "{email}" error - {e}')    
    def __open_by_url(self, table_url):
        spreadsheet = self.__gs.open_by_url(table_url)
        return spreadsheet
    def share_spreadsheet(self, table_url, list_of_emails, role = 'writer'):
        '''
        roles - 'writer', 'owner', 'reader'
        '''
        spreadsheet = self.__open_by_url(table_url)
        for email in list_of_emails:
            try:
                spreadsheet.share(email, perm_type='user', role=role)
                print('- Table "{}" have been shared successfully to {} with role {}!'.format(new_table_name, email, role))
            except Exception as e: print(f'- Email "{email}" error - {e}')
    def get_permissions_by_url(self, table_url):
        spreadsheet = self.__gs.open_by_url(table_url)
        return spreadsheet.list_permissions()
    def delete_sheets(self, table_url, list_of_sheet_names_to_delete):
        spreadsheet = self.__gs.open_by_url(table_url)
        for sheet_name in list_of_sheet_names_to_delete:
            try:
                spreadsheet.del_worksheet(spreadsheet.worksheet(sheet_name))
                print(f'- Sheet "{sheet_name}" deleted successfully in https://docs.google.com/spreadsheets/d/{spreadsheet.id}/ !')
            except Exception as e: print(f'- Sheet "{sheet_name}" was not deleted in https://docs.google.com/spreadsheets/d/{spreadsheet.id}/ - {e}')
    def create_sheets(self, table_url, list_of_sheet_names_to_create):
        spreadsheet = self.__gs.open_by_url(table_url)
        for sheet_name in list_of_sheet_names_to_create:
            try:
                spreadsheet.add_worksheet(title=sheet_name, rows="50", cols="4")
                print(f'- Sheet "{sheet_name}" created successfully in https://docs.google.com/spreadsheets/d/{spreadsheet.id}/ !')
            except Exception as e: print(f'- Sheet "{sheet_name}" was not created in https://docs.google.com/spreadsheets/d/{spreadsheet.id}/ - {e}')
    def get_worksheet(self, table_url, sheet_name):
        spreadsheet = self.__gs.open_by_url(table_url)
        worksheet = spreadsheet.worksheet(sheet_name)
        return worksheet
    # pandas <--> sheet
    def df_to_sheet(self, df, table_url, sheet_name, start_cell='A1', insert_column_names = True, formatted_default = True):
        # insert df
        str_df = df.copy()
        for column in str_df.columns:
            str_df[column] = str_df[column].astype('str')
        spreadsheet = self.__gs.open_by_url(table_url)
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
        except Exception as e: 
            print(f'- Sheet_name is missing https://docs.google.com/spreadsheets/d/{spreadsheet.id}/ - {e}\nSheet "{sheet_name}" created successfully!')
            spreadsheet.add_worksheet(title=sheet_name, rows="30", cols="4")
            worksheet = spreadsheet.worksheet(sheet_name) 
        if insert_column_names == True:
            values = [str_df.columns.values.tolist()]
            values.extend(str_df.values.tolist())
        else:
            values = str_df.values.tolist()
        spreadsheet.values_update(sheet_name+'!'+start_cell, params={'valueInputOption': 'USER_ENTERED'}, body={'values': values})
        print(f'- Dataframe writed successfully in "{sheet_name}" in https://docs.google.com/spreadsheets/d/{spreadsheet.id}/ !')
        # formatted
        if formatted_default == True:
            # format range
            start_row, start_col = a1_to_rowcol(start_cell)
            if insert_column_names == True: 
                insert_rows = df.shape[0] + 1
            else:
                insert_rows = df.shape[0]
            insert_cols = df.shape[1]   
            finish_row = start_row + insert_rows - 1
            finish_col = insert_cols
            # format
            if insert_column_names == True:
                # header
                range_name_start = rowcol_to_a1(start_row, start_col)
                range_name_finish = rowcol_to_a1(start_row, insert_cols)
                self.sheet_format_range(table_url, sheet_name, range_name = f'{range_name_start}:{range_name_finish}',**self.header_format_default_dict, print_results = False)
                # body
                range_name_start = rowcol_to_a1(start_row + 1, start_col)
                range_name_finish = rowcol_to_a1(finish_row, finish_col)
                self.sheet_format_range(table_url, sheet_name, range_name = f'{range_name_start}:{range_name_finish}',**self.body_format_default_dict, print_results = True)
            else:
                # only body
                range_name_start = rowcol_to_a1(start_row, start_col)
                range_name_finish = rowcol_to_a1(finish_row, finish_col)
                self.sheet_format_range(table_url, sheet_name, range_name = f'{range_name_start}:{range_name_finish}',**self.body_format_default_dict, print_results = True)
    def df_to_sheet_append(self, df, table_url, sheet_name, insert_column_names = False, skip_rows = 0, formatted_default = True):
        str_df = df.copy()
        for column in str_df.columns:
            str_df[column] = str_df[column].astype('str')
        spreadsheet = self.__gs.open_by_url(table_url)
        try:
            worksheet = spreadsheet.worksheet(sheet_name)
        except Exception as e: 
            print(f'- Sheet_name is missing https://docs.google.com/spreadsheets/d/{spreadsheet.id}/ - {e}') 
            
        last_filled_row = len(worksheet.get_all_values())
            
        if insert_column_names == True:
            values = [str_df.columns.values.tolist()]
            values.extend(str_df.values.tolist())
        else:
            values = str_df.values.tolist()
        
        start_row = last_filled_row + 1 + skip_rows
        start_column = 1
        start_cell = rowcol_to_a1(start_row, start_column)
        sheet_info_row_df = self.get_sheet_info(table_url, sheet_name)
        sheet_rows = sheet_info_row_df.iloc[0]['sheet_rows']
        if start_row > sheet_rows:
            worksheet.add_rows(int(start_row - sheet_rows))
        self.df_to_sheet(df, table_url, sheet_name, start_cell = start_cell, insert_column_names = insert_column_names, formatted_default = formatted_default)
    def sheet_to_df(self, table_url, sheet_name):
        spreadsheet = self.__gs.open_by_url(table_url)
        worksheet = spreadsheet.worksheet(sheet_name)
        list_of_lists = worksheet.get_all_values()
        headers = list_of_lists.pop(0)
        df = pd.DataFrame(worksheet.get_all_records())
        return df
    # get info
    def get_spreadsheet_metadata(self, table_url):
        spreadsheet = self.__gs.open_by_url(table_url)
        return spreadsheet.fetch_sheet_metadata()
    def get_sheets_info(self, table_url):
        spreadsheet_metadata_dict = self.get_spreadsheet_metadata(table_url)
        sheet_names = [i['properties']['title'] for i in spreadsheet_metadata_dict['sheets']]
        sheet_ids = [i['properties']['sheetId'] for i in spreadsheet_metadata_dict['sheets']]
        sheet_indexes = [i['properties']['index'] for i in spreadsheet_metadata_dict['sheets']]
        sheet_rows = [i['properties']['gridProperties']['rowCount'] for i in spreadsheet_metadata_dict['sheets']]
        sheet_frozen_rows = [i['properties']['gridProperties'].get('frozenRowCount') for i in spreadsheet_metadata_dict['sheets']]
        sheet_columns = [i['properties']['gridProperties']['columnCount'] for i in spreadsheet_metadata_dict['sheets']]
        sheet_frozen_columns = [i['properties']['gridProperties'].get('frozenColumnCount') for i in spreadsheet_metadata_dict['sheets']]
        df = pd.DataFrame({
            'sheet_name':sheet_names,
            'sheet_id':sheet_ids,
            'sheet_index':sheet_indexes,
            'sheet_rows':sheet_rows,
            'sheet_columns':sheet_columns,
            'sheet_frozen_rows':sheet_frozen_rows,
            'sheet_frozen_columns':sheet_frozen_columns
        })
        return df
    def get_sheet_info(self, table_url, sheet_name):
        return self.get_sheets_info(table_url).query('sheet_name==@sheet_name')
    # sheet operations
    def sheet_rename(self, table_url, sheet_name, new_sheet_name):
        spreadsheet = self.__gs.open_by_url(table_url)
        worksheet = spreadsheet.worksheet(sheet_name)
        worksheet.update_title(new_sheet_name)
        print('- Sheet "{}" in table "{}" have been renames successfully to "{}"!'.format(sheet_name, spreadsheet.title, new_sheet_name))
    def sheet_freeze_area(self, table_url, sheet_name, rows = 1, cols = None):
        spreadsheet = self.__gs.open_by_url(table_url)
        worksheet = spreadsheet.worksheet(sheet_name)
        worksheet.freeze(rows = rows, cols = cols)
        print('- Area in sheet "{}" in table "{}" have been successfully freezed!'.format(sheet_name, spreadsheet.title))
    def sheet_duplicate(self, table_url, source_sheet_name, new_sheet_name):
        spreadsheet = self.__gs.open_by_url(table_url)
        sheet_id = spreadsheet.worksheet(source_sheet_name).id
        spreadsheet.duplicate_sheet(sheet_id, new_sheet_name = new_sheet_name)
        print('- Sheet "{}" in table "{}" have been duplicated successfully with name "{}"!'.format(source_sheet_name, spreadsheet.title, new_sheet_name))
    def sheet_copy_to_spreadsheet(self, source_table_url, sheet_name, new_table_url, new_sheet_name = None):
        source_spreadsheet = self.__gs.open_by_url(source_table_url)
        source_worksheet = source_spreadsheet.worksheet(sheet_name)
        new_spreadsheet = self.__gs.open_by_url(new_table_url)
        source_worksheet.copy_to(new_spreadsheet.id)
        new_worksheet = new_spreadsheet.worksheet('Copy of ' + str(sheet_name))
        if new_sheet_name != None:
            new_worksheet.update_title(new_sheet_name)
            print('- Sheet "{}" in table "{}" have been successfully copied to table "{}" to new sheet "{}"!'.format(sheet_name, source_spreadsheet.title, new_spreadsheet.title, new_sheet_name))
        else:
            new_worksheet.update_title(sheet_name)
            print('- Sheet "{}" in table "{}" have been successfully copied to table "{}"!'.format(sheet_name, source_spreadsheet.title, new_spreadsheet.title))
    def sheet_format_range(self, table_url, sheet_name, range_name = 'A2:B2', 
                     background_color = None,
                     text_color = None,
                     font_size = None,
                     font_bold = None,
                     horizontal_alignment = None,
                     vertical_alignment = None,
                     borders_style = None,
                     borders_color = None,
                     print_results = True
                    ):
        '''
        font_size: integer
        font_bold: bool - True, False
        horizontal_alignment: string - 'LEFT', 'CENTER', 'RIGHT'
        vertical_alignment: string - 'TOP', 'MIDDLE', 'BOTTOM'
        borders_style: string - 'DOTTED', 'DASHED', 'SOLID', 'SOLID_MEDIUM', 'SOLID_THICK', 'NONE', 'DOUBLE'
        
        example format params:
        background_color = '#2b6ca3',
        text_color = 'white',
        font_size = 8,
        font_bold = True,
        horizontal_alignment = 'CENTER',
        vertical_alignment = 'MIDDLE',
        borders_style = 'DOTTED',
        borders_color = 'black'
        '''
        spreadsheet = self.__gs.open_by_url(table_url)
        worksheet = spreadsheet.worksheet(sheet_name)
        
        update_dict = {}
        background_color_dict = {}
        text_format_dict = {}
        borders_format_dict = {}
        
        if background_color != None: background_color_dict = {'red':colors.to_rgb(background_color)[0],'green':colors.to_rgb(background_color)[1],'blue':colors.to_rgb(background_color)[2]}
        if len(background_color_dict) != 0: update_dict['backgroundColor'] = background_color_dict
            
        if text_color != None: text_format_dict['foregroundColor'] = {'red':colors.to_rgb(text_color)[0],'green':colors.to_rgb(text_color)[1],'blue':colors.to_rgb(text_color)[2]}
        if font_size != None: text_format_dict['fontSize'] = font_size
        if font_bold != None: text_format_dict['bold'] = font_bold
        if len(text_format_dict) != 0: update_dict['textFormat'] = text_format_dict
            
        if horizontal_alignment != None: update_dict['horizontalAlignment'] = horizontal_alignment
        if vertical_alignment != None: update_dict['verticalAlignment'] = vertical_alignment
        
        # if we have parametres to change borders
        borders_format_dict = {'top':{},'bottom':{},'left':{},'right':{}}
        if borders_style != None:
            for border_type in ['top','bottom','left','right']:
                borders_format_dict[border_type]['style'] = borders_style
        if borders_color != None:
            for border_type in ['top','bottom','left','right']:
                borders_format_dict[border_type]['color'] = {'red':colors.to_rgba(borders_color)[0],'green':colors.to_rgba(borders_color)[1],'blue':colors.to_rgba(borders_color)[2],'alpha':colors.to_rgba(borders_color)[3]}   
        if borders_style != None or borders_color != None: update_dict['borders'] = borders_format_dict
        
        if len(update_dict) != 0:
            worksheet.format(range_name, update_dict)
            if print_results == True: print('- Sheet "{}" in table "{}" have been successfully formatted!'.format(sheet_name, spreadsheet.title))   
        else:
            if print_results == True: print('''- Sheet "{}" in table "{}" haven't been formatted! Formatting options were not selected!'''.format(sheet_name, spreadsheet.title))
    def sheet_format_full_range(self, table_url, sheet_name,
                          background_color = None,
                          text_color = None,
                          font_size = None,
                          font_bold = None,
                          horizontal_alignment = None,
                          vertical_alignment = None,
                          borders_style = None,
                          borders_color = None
                         ):
        sheet_info_row_df = self.get_sheet_info(table_url, sheet_name)
        sheet_rows = sheet_info_row_df.iloc[0]['sheet_rows']
        sheet_columns = sheet_info_row_df.iloc[0]['sheet_columns']
        range_name_start = 'A1'
        range_name_finish = rowcol_to_a1(sheet_rows, sheet_columns)
        self.sheet_format_range(table_url, sheet_name, range_name = f'{range_name_start}:{range_name_finish}',
                                background_color = background_color,
                                text_color = text_color,
                                font_size = font_size,
                                font_bold = font_bold,
                                horizontal_alignment = horizontal_alignment,
                                vertical_alignment = vertical_alignment,
                                borders_style = borders_style,
                                borders_color = borders_color)
    def sheet_format_full_range_default(self, table_url, sheet_name, header_cols = 1):
        sheet_info_row_df = self.get_sheet_info(table_url, sheet_name)
        sheet_rows = sheet_info_row_df.iloc[0]['sheet_rows']
        sheet_columns = sheet_info_row_df.iloc[0]['sheet_columns']
        # header
        range_name_start = 'A1'
        range_name_finish = rowcol_to_a1(header_cols, sheet_columns)
        self.sheet_format_range(table_url, sheet_name, range_name = f'{range_name_start}:{range_name_finish}',**self.header_format_default_dict, print_results = False)
        # body
        range_name_start = rowcol_to_a1(header_cols + 1, 1)
        range_name_finish = rowcol_to_a1(sheet_rows, sheet_columns)
        self.sheet_format_range(table_url, sheet_name, range_name = f'{range_name_start}:{range_name_finish}',**self.body_format_default_dict, print_results = True)