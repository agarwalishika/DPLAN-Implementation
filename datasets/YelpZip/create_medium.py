import pandas as pd
import random

num_rows_total = 608598
num_rows_keep = 2000

rows_to_keep = random.sample(range(1, num_rows_total), num_rows_keep)

medium_meta = pd.read_csv('raw/metadata', sep='\t', skiprows = lambda x: x not in rows_to_keep)
medium_review = pd.read_csv('raw/reviewContent', sep='\t', skiprows = lambda x: x not in rows_to_keep)

r = 1000

medium_meta.iloc[0:r].to_csv('medium_raw/metadata', sep = '\t', index=False)
medium_meta.iloc[r:].to_csv('gdn_medium_raw/metadata', sep = '\t', index=False)
medium_review.iloc[0:r].to_csv('medium_raw/reviewContent', sep = '\t', index=False)
medium_review.iloc[r:].to_csv('gdn_medium_raw/reviewContent', sep = '\t', index=False)
