#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import pandas as pd
from pathlib import Path

file1 = input("Enter the file1 name:")
file2 = input("Enter the file2 name:")
df_OLD = pd.read_excel(file1).fillna('')
df_NEW = pd.read_excel(file2).fillna('')
dfDiff = df_OLD.copy()
for row in range(dfDiff.shape[0]):
    for col in range(dfDiff.shape[1]):
        value_OLD = df_OLD.iloc[row,col]
        try:
            value_NEW = df_NEW.iloc[row,col]
            if value_OLD==value_NEW:
                dfDiff.iloc[row,col] = df_NEW.iloc[row,col]
            else:
                dfDiff.iloc[row,col] = ('{}→{}').format(value_OLD,value_NEW)
        except:
            dfDiff.iloc[row,col] = ('{}→{}').format(value_OLD, 'NaN')

writer = pd.ExcelWriter("Output.xlsx", engine='xlsxwriter')
dfDiff.to_excel(writer, sheet_name='DIFF', index=False)

workbook  = writer.book
worksheet = writer.sheets['DIFF']
worksheet.set_column('A1:ZZ1000',25)
worksheet.hide_gridlines(2)

# define formats
date_fmt = workbook.add_format({'align': 'center', 'num_format': 'yyyy-mm-dd'})
center_fmt = workbook.add_format({'align': 'center'})
number_fmt = workbook.add_format({'align': 'center', 'num_format': '#,##0.00'})
cur_fmt = workbook.add_format({'align': 'center', 'num_format': '$#,##0.00'})
perc_fmt = workbook.add_format({'align': 'center', 'num_format': '0%'})
grey_fmt = workbook.add_format({'font_color': '#E0E0E0'})
highlight_fmt = workbook.add_format({'font_color': '#FF0000', 'bg_color':'#B1B3B3'})
new_fmt = workbook.add_format({'font_color': '#32CD32','bold':True})
highlight_fmt = workbook.add_format({'bold': True,'font_color': '#FF0000', 'bg_color':'#FFFF00'})
new_fmt = workbook.add_format({'font_color': '#808080','bold':False})
# set format over range
    
                # highlight changed cells
worksheet.conditional_format('A1:ZZ1000', { 'type': 'text',
                                                'criteria': 'not containing',
                                                'value':'→',
                                                'format':new_fmt })
                # highlight unchanged cells
worksheet.conditional_format('A1:ZZ1000', { 'type': 'text',
                                                'criteria': 'containing',
                                                'value':'→',
                                                'format':highlight_fmt})
# save
writer.save()
print("Output differences created.!")


# In[ ]:




