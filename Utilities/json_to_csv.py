#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json 
import csv 
import pandas as pd

filename = input("Enter the JSON file name:")
with open(filename) as json_file: 
    data = json.load(json_file) 

# if there is a root element in json file enter that else press enter as root
try:
    root = input("enter the root of JSON file: ")
    employee_data = data[root] 

    data_file = open("Out_Put.csv", 'w',newline='',encoding="utf8") 

    csv_writer = csv.writer(data_file) 

    count = 0

    for emp in employee_data: 
        if count == 0: 

            header = emp.keys() 
            csv_writer.writerow(header) 
            count += 1
        csv_writer.writerow(emp.values()) 
  
    data_file.close() 
    
except KeyError as e:
    
    df = pd.json_normalize (data)
    df.to_csv (r"Out_Put.csv", index = None,header=True)


# In[ ]:





# In[ ]:




