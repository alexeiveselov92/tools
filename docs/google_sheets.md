# Google sheets module
> [to code link](https://github.com/alexeiveselov92/tools/blob/main/google_sheets.py)


## Quick start
```
from google_sheets import google_sheets_tools
gsh = google_sheets_tools(path_to_credential)
# create new spreadsheet
gsh.create_spreadsheet('new_spreadsheet', [my_mail@mail.com])
# create sheet in spreadsheet
gsh.create_sheets(your_table_url, ['new_sheet'])
```