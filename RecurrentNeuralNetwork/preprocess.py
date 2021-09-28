import pandas as pd

import numpy as np

table = pd.read_excel("/home/grandpadzb/下载/C/附件1：123家有信贷记录企业的相关数据.xlsx", sheet_name=1)

rows = table.shape[0]
cols = table.shape[1]

train_x = []
tmp = []
j = 0
for i in range(0,rows):
    if(table.iloc[i][0] == f'E{j+1}'):
        if(table.iloc[i][-1] == '有效发票'):
            tmp.append(table.iloc[i][6])
            pass
    if(i == rows-1):
        train_x.append(np.array(tmp))
        tmp = []
        j = j+1
        break
    if(table.iloc[i+1][0] != f'E{j+1}'):
        train_x.append(np.array(tmp))
        tmp = []
        j = j+1
    print(f'{i/rows*100}%')

max_len = 31435

for i in range(len(train_x)):
    train_x[i] = np.concatenate(
        [train_x[i], np.zeros(max_len-train_x[i].size,)],
        0
    )