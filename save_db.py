import pandas as pd
import time
from datetime import datetime
import os

# os.makedirs("DB",exist_ok=True)



def put_data(id):
    now = datetime.now()
    date="{d}_{m}_{y}.csv".format(d =now.day, m=now.month, y=now.year)
    root = "export_DB"
    file_name = os.path.join(root,date)
    if not os.path.exists(file_name):
        column = ['ID', 'DATE', 'START_TIME', 'END_TIME']
        df = pd.DataFrame(columns=column)
        df.to_csv(file_name, index=False)
    else:
        df = pd.read_csv(file_name, engine='python')
        # if check_event():
        if id not in df['ID'].values:
            start_time = '{hour}:{minute}:{second}'.format(hour = now.hour, minute = now.minute,second = now.second)
            end_time = 'null'
            new_row = pd.Series({'ID':id,'DATE': date.replace('.csv',''),'START_TIME': start_time,'END_TIME': end_time})
            df = pd.concat([df, new_row.to_frame().T], ignore_index=True)
            df.to_csv(file_name,index=False)
        else :
            end_time ='{hour}:{minute}:{second}'.format(hour = now.hour, minute = now.minute,second = now.second)
            update_index = df.index[df['ID'] == id]
            df.at[update_index,"END_TIME"] = end_time
            df.to_csv(file_name, index=False)