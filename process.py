#此代码为第三步 删除没有电荷补偿元素的体系 删除小于7元的体系 删除S config小于1.5R的体系
import pandas as pd
import numpy as np

file_path = 'your data path'

df = pd.read_csv(file_path)
R = 8.314  # J/mol·K

df['ElementCount'] = df[['Element1', 'Element2', 'Element3', 'Element4', 'Element5', 'Element6', 'Element7']].apply(lambda x: (x != 0).sum(), axis=1)
df = df[df['ElementCount'] == 7] 

target_elements = ['Ti', 'Mn', 'Fe', 'Ni', 'Co', 'Cr']
df = df[df[['Element1', 'Element2', 'Element3', 'Element4', 'Element5', 'Element6', 'Element7']].apply(lambda row: any(elem in target_elements for elem in row), axis=1)]

fractions = ['Fraction1', 'Fraction2', 'Fraction3', 'Fraction4', 'Fraction5', 'Fraction6', 'Fraction7']

df['S_Config'] = -R * df[fractions].apply(lambda row: np.nansum([x * np.log(x) for x in row if x > 0]), axis=1)

threshold = 1.5 * R
df = df[df['S_Config'] > threshold]

output_file = 'your data path'
df.to_csv(output_file, index=False)





