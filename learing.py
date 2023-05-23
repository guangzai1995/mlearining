import pandas as pd

df1=pd.DataFrame({'id':[1,2,3],'name':['Andy1','Jacky1','Bruce1']})
df2=pd.DataFrame({'id':[1,2],'name':['Andy2','Jacky2']})
print(df1)
print(df2)
s = df2.set_index('id')['name']
df1['name'] = df1['id'].map(s).fillna(df1['name']).astype(str)
print(df1)
