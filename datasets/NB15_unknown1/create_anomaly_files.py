import pandas as pd

anomaly_types = ["Fuzzers", "Generic", "Reconnaissance"]
anom_data = pd.read_csv("anomalies.csv")
norm_data = pd.read_csv("normal.csv")

contamination_rate = 0.02
num_known = 60
data_size = 10000

num_anom = int(data_size * contamination_rate)
num_norm = int(data_size * (1 - contamination_rate))
for at in anomaly_types:
    anom_subset = anom_data[anom_data['attack_cat'] == at]
    anom_subset = anom_subset[0:num_anom]
    anom_subset.loc[:,'Label'] = 1
    anom_subset.loc[:,'attack_cat'] = 0


    norm_subset = norm_data[0:num_norm]

    frames = [anom_subset, norm_subset]
    combined_data = pd.concat(frames)
    combined_data = combined_data.drop(['attack_cat'], axis=1)
    
    fname = "{}_{}_{}.csv".format(at, contamination_rate, num_known)
    combined_data.to_csv(fname, index=False)
    
