#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tempfile import mkstemp
from shutil import move
from os import fdopen, remove
import os as os
import sys
from datetime import datetime
import re

def text_replace():
    try:
        path = input("Enter the file-name or the file path: ")
        try:
            patterns = int(input("Enter How many patterns to be replaced: "))
        except:
            generate_log("Invalid Number.!,Enter a valid number.","error-code:0")
            print("Enter a valid number.!")
            sys.exit("Invalid Number.!")

        for pat in range(patterns):
            pattern = input("Enter the URL pattren to serach: ")
            print(pattern)
            subst = input("Enter the URL pattren to replace: ")
            print(subst)
            
            # This will search the json,text,tsv,csv files
            
            filelist = [f for f in os.listdir(path) if (f.endswith('.json') or f.endswith('.txt') or f.endswith('.tsv') or f.endswith('.csv'))]
            print("Number of file in the directory are: ",len(filelist))
            try:
                for f in filelist:  
                    file_path = os.path.join(path, f)
                    print(file_path)
                    fh, abs_path = mkstemp()
                    with fdopen(fh,'w',encoding="UTF-8",errors='ignore', newline='') as new_file:
                        with open(file_path,encoding="UTF-8",errors='ignore', newline='') as old_file:
                            for line in old_file:
                                new_file.write(re.sub(re.escape(pattern), subst, line, flags=re.IGNORECASE))
                    remove(file_path)
                    move(abs_path, file_path)   
                generate_log("Operation successfull.!"+",Searched Files: ",len(filelist))
            except Exception as e:
                    print(e)
                    if("codec can't decode byte" in str(e)):
                        continue

    except Exception as e:
        print(e)
        generate_log(str(e)+",Searched Files: ",0)

# Custom Logging to update the status of text replacement
        
def generate_log(text,files):
    
    f = open("LogInfo.ini", "a")
    f.write("{0} -- {1} {2}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), text,files))
    f.close()

if __name__ == '__main__':
    
    logfile = open("./LogInfo.ini", "a")
    text_replace()


# In[ ]:





# In[ ]:




