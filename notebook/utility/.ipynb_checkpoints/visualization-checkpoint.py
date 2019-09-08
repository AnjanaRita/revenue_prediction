import numpy as np
import pandas as pd
import seaborn as sns
from ast import literal_eval
import matplotlib.pyplot as plt
from collections import Counter


def visualization_list_data(data, columns, replace_value = [], max_feature = 20, extracte_data = 'name'):
    print(f'Visualization of the value of feature {columns}')
    data = dealing_null_value(data, columns, replace_value)
    gerner_count = Counter([j.get(extracte_data) for i in data[columns] for j in i])
    if len(gerner_count) > 20:
        print(f'total availabe different values of {columns} are {len(gerner_count)}. We will print top 20 values')
        gerner_count = gerner_count.most_common(max_feature)
    gerner_count = dict(gerner_count)
    sns.set(rc={'figure.figsize':(15,9)})
    sns.barplot(list(gerner_count.values()), list(gerner_count.keys()))
    plt.xlabel(f'count of {columns}')
    plt.ylabel(f'{columns}')
    plt.show()
    
def visualization_value_count(data, columns, replace_value = [], figure_size=None):
    print(f"per movie number of {columns} value")
    data =dealing_null_value(data, columns, replace_value)
    temp = data[columns].apply(lambda x: len(x) if isinstance(x,list) else 0).value_counts().reset_index()
    if figure_size == None:
        sns.set(rc={'figure.figsize':(11,8)})
    else:
        sns.set(rc={'figure.figsize':figure_size})
    sns.barplot(x='index',y=columns,data=temp)
    plt.xlabel(f'Number of {columns}')
    plt.ylabel('count of the movie')
    plt.show()
    
def dealing_null_value(data, columns, replace_value = []):
    data[columns] = data[columns].fillna(0)
    try:
        data[columns] = [literal_eval(i) if i != 0 else replace_value for i in data[columns]]
    except:
        return data
    return data