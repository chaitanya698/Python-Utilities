#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import xml.etree.cElementTree as etree
import xml.etree.ElementTree as ET
import pandas as pd

#Dynamically taking the file name from user
input_xml_file = input("Enter the xml file name:")
with open(input_xml_file) as xml_file:
    
    tree = etree.iterparse(xml_file)
  
    headers=[]
    attributes=[]
    data=[]
    
    for item in tree:
        if item[1].text != None:
            headers.append(item[1].tag)
            attributes.append(item[1].attrib)
            data.append(item[1].text)

print("Total tags extraced:", len(headers)) 
print("Total tag data extraced:", len(data)) 

writer = pd.ExcelWriter("output.xlsx",engine='xlsxwriter')
df=pd.DataFrame(data,headers).T
df1=pd.DataFrame(attributes).T
dff= pd.concat([df,df1],axis=1)
dff.to_excel(writer, sheet_name='Sheet1',startrow=0,index=False)
df1.to_excel(writer, sheet_name='Sheet2',startrow=0,index=False)
workbook  = writer.book
header_format = workbook.add_format({
    'bold': True,
    'fg_color': '#ffcccc',
    'border': 1})

writer.save()

print("XML is Extracted and Excel is created.!")


# In[ ]:





# In[ ]:




