import pandas as pd

file_name = "test_for_all.csv"
data = pd.read_csv(file_name, skipinitialspace=True)

normal_data = data[data['Label'] == 0]
normal_data.to_csv("normal.csv", index=False)

anom_data = data[data['Label'] != 0]
anom_data.to_csv("anomalies.csv", index=False)
