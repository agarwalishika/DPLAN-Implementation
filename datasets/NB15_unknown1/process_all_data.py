import pandas as pd
import numpy as np
import os.path

def process_data_from_scratch():
    file_name = "raw_test_for_all.csv"
    #file_name = "small.csv"
    data = pd.read_csv(file_name)


    data.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    #one hot encode stuff
    encodings = ['srcip', 'dstip', 'proto', 'state', 'service']
    for encode in encodings:
        one_hot = pd.get_dummies(data[encode])
        data = data.drop(encode,axis=1)

        data = data.join(one_hot, lsuffix=encode)



    #set all non-numeric values in numeric variables to 0
    num_var = ['sport','dsport','dur','sbytes','dbytes','sttl','dttl','sloss','dloss','sload','dload','spkts','dpkts',
        'swin','dwin','stcpb','dtcpb','smeansz','dmeansz','trans_depth','res_bdy_len','sjit','djit','stime','ltime',
        'sintpkt','dintpkt','tcprtt','synack','ackdat','is_sm_ips_ports','ct_state_ttl','ct_flw_http_mthd',
        'is_ftp_login','ct_ftp_cmd','ct_srv_src','ct_srv_dst','ct_dst_ltm','ct_src_ltm','ct_src_dport_ltm',
        'ct_dst_sport_ltm','ct_dst_src_ltm']

    float32_max = np.finfo(np.float32).max
    for nv in num_var:
        data[nv] = (pd.to_numeric(data[nv], errors='coerce').fillna(0))
        float_data = data[nv].astype(np.float64)
        data[nv] = np.where(float_data > float32_max, float32_max - 1, data[nv])


    data.to_csv("test_for_all.csv", index=False)

def process_data_from_starting_point():
    file_name = "test_for_all.csv"
    data = pd.read_csv(file_name)

    data = data.drop(['attack_cat'], axis=1)
    #randomize and get first 1000
    data = data.sample(frac=1)
    data = data.head(10000)

    #move the labels to the end of the table
    data = data[[c for c in data if c not in ['Label']] + ['Label']]
    data.to_csv(file_name, index=False)


file_name = "test_for_all.csv"
if os.path.isfile(file_name):
    process_data_from_starting_point()
else:
    process_data_from_scratch()
