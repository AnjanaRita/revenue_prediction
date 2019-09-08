import pandas as pd

def csv_reader(path):
    dataset = pd.read_csv(path,error_bad_lines=False)
    return dataset

def missing_data_stats(data, columns=None):
    if columns == None:
        columns = [i for i in data.columns.values if i not in ['id', 'revenue']]
    data = data[columns]
    missing_data = pd.DataFrame(data.isnull().sum()).reset_index()
    missing_data = missing_data.rename(columns={'index':'feature',0:'nan_count'})
    missing_data['% for nan'] = missing_data['nan_count']/ len(data)
    return missing_data

def advance_missing_data_stat(missing_data):
    temp = missing_data[missing_data['% for nan'] < 0.5]
    less_than_one = temp[temp['% for nan'] > 0.0]
    no_missing = missing_data[missing_data['nan_count'] == 0.0]
    more_then_seventy  = missing_data[missing_data['% for nan'] > 0.6]
    
    data = [('no missing value',no_missing.shape[0], no_missing['feature'].values),
            ('missing value less than 5%', less_than_one.shape[0], less_than_one.feature.values),
           ('missing value more than 70%', more_then_seventy.shape[0], more_then_seventy.feature.values)]
    
    return pd.DataFrame(data, columns=['#','feature count', 'feature_name']).set_index("#")

def remove_columns(data, drop_columns = None):
    if drop_columns != None:
        data = data.drop(columns=drop_columns)
    return data