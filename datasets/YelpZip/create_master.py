import pandas as pd

file_name = "df_YelpZip_comp.csv"

# idx, prod_id, rating, date, review, label

meta = pd.read_csv("metadata", sep='\t', header=None)
meta.columns = ['user_id', 'prod_id', 'rating', 'label', 'date']

review = pd.read_csv("reviewContent", sep='\t', header=None)
review.columns = ['user_id', 'prod_id', 'date', 'review']

df = pd.merge(meta, review, on=['user_id', 'prod_id', 'date'], how='left')

df = df.reset_index()
df = df.drop(columns=['user_id'])
df = df.rename(columns={'index':'idx'})

cols = ['idx', 'prod_id', 'rating', 'date', 'review', 'label']
df = df[cols]

df.to_csv(file_name, sep='\t', index=False, header=False)

hi = 9
