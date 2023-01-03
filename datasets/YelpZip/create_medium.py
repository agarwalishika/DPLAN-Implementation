import pandas as pd
import random

num_rows_total = 608598
times = 60
num_rows_keep = num_rows_total-1
row_section = int(num_rows_keep/times)

rows_to_keep = random.sample(range(1, num_rows_total), num_rows_keep)

for i in range(times):
    rows = rows_to_keep[i*row_section:(i+1)*row_section]

    medium_meta = pd.read_csv('raw/metadata', sep='\t', skiprows = lambda x: x not in rows)
    medium_review = pd.read_csv('raw/reviewContent', sep='\t', skiprows = lambda x: x not in rows)


    medium_meta.to_csv(f'metagdn/metadata{i}', sep = '\t', index=False)
    medium_review.to_csv(f'metagdn/reviewContent{i}', sep = '\t', index=False)

#medium_meta.iloc[r:].to_csv('pret_gdn_medium_raw/metadata', sep = '\t', index=False)
#medium_review.iloc[r:].to_csv('pret_gdn_medium_raw/reviewContent', sep = '\t', index=False)
