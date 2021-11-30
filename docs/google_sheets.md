# Google sheets module
> [to code link](https://github.com/alexeiveselov92/tools/blob/main/google_sheets.py)


## Quick start
```
from google_sheets import google_sheets_tools
gsh = google_sheets_tools(path_to_credential)

# create new spreadsheet
gsh.create_spreadsheet('new_spreadsheet', [my_mail@mail.com])

# create sheet in spreadsheet
gsh.create_sheets(table_url, ['new_sheet'])

# rename sheet
gsh.sheet_rename(table_url, sheet_name, new_sheet_name)

# freeze first row in sheet
gsh.sheet_freeze_area(table_url, sheet_name, rows = 1, cols = None)

# duplicate sheet
gsh.sheet_duplicate(table_url, source_sheet_name, new_sheet_name)

# copy sheet to other spreadsheet
gsh.sheet_copy_to_spreadsheet(source_table_url, sheet_name, new_table_url, new_sheet_name = None)

# df to sheet with default format
gsh.df_to_sheet(df, table_url, sheet_name, start_cell='A1', insert_column_names = True, formatted_default = True)
gsh.df_to_sheet_append(df, table_url, sheet_name, insert_column_names = False, skip_rows = 0, formatted_default = True)

# sheet to df
gsh.sheet_to_df(table_url, sheet_name)
```

**All class methods you can see in [code](https://github.com/alexeiveselov92/tools/blob/main/google_sheets.py)**