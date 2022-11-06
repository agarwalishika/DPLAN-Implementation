import pandas as pd

file_name = "df_YelpZip_comp.csv"

# idx, prod_id, rating, date, review, label

meta = pd.read_csv("metadata", sep='\t', header=None)
meta.columns = ['user_id', 'prod_id', 'rating', 'label', 'date']

review = pd.read_csv("reviewContent", sep='\t', header=None)
review.columns = ['user_id', 'prod_id', 'date', 'review']

meta['review'] = ""

for ind, r in meta.iterrows():
    u = r['user_id']
    p = r['prod_id']
    rev = review.query(f'user_id == {u} and prod_id == {p}')['review'].reset_index(drop=True)
    try:
        meta.at[ind, 'review'] = rev[0]
    except:
        meta.at[ind, 'review'] = ""

    if meta.at[ind, 'label'] == -1:
        meta.at[ind, 'label'] = 1
    else:
        meta.at[ind, 'label'] = 0

meta = meta.reset_index()
meta = meta.drop(columns=['user_id'])
meta = meta.rename(columns={'index':'idx'})

cols = ['idx', 'prod_id', 'rating', 'date', 'review', 'label']
meta = meta[cols]

meta.to_csv(file_name, sep='\t', index=False, header=False)

hi = 9
