import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend.tensorflow_backend as K
tf.device("cuda")

from DPLAN_stream import DPLAN
from ADEnv import ADEnv
from utils import writeResults
from sklearn.metrics import roc_auc_score, average_precision_score
from GDN import score_sample
from DPLAN_stream import DQN_iforest
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

### Basic Settings
# data path settings
data_path="datasets"
data_folders = ["YelpZip"]
data_subsets={"YelpZip" : ["small_embeds/final_embeddings"]}
# data_folders=["NB15_unknown1"]
# data_subsets={"NB15_unknown1":["Fuzzers","Generic","Reconnaissance"]}
# data_subsets={"NB15_unknown1":["Analysis","DoS","Exploits","Fuzzers","Reconnaissance"]}
testdata_subset="small_embeds/test_final_embeddings.csv" # test data is the same for subsets of the same class
# experiment settings
runs=15
model_path="./small_embeds"
result_path="./results/small_embeds"
result_file="results.csv"
Train=True
Test=True

### Anomaly Detection Environment Settings
size_sampling_Du=860
prob_au=0.5
label_normal=1
label_anomaly=-1


### DPLAN Settings
settings={}
settings["hidden_layer"]=40 # l
settings["memory_size"]=100000 # M
settings["warmup_steps"]=100 # 10000
settings["episodes"]=10
settings["steps_per_episode"]=200 #2000
settings["epsilon_max"]=1
settings["epsilon_min"]=0.1
settings["epsilon_course"]=10000
settings["minibatch_size"]=64
settings["discount_factor"]=0.99 # gamma
settings["learning_rate"]=0.00025
settings["minsquared_gradient"]=0.01
settings["gradient_momentum"]=0.95
settings["penulti_update"]=2000 # N
settings["target_update"]=10000 # K

devices = 'cuda'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

config = tf.compat.v1.ConfigProto(
    device_count={'GPU': 1},
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1,
    allow_soft_placement=True
)

config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.6

session = tf.compat.v1.Session(config=config)

K.set_session(session)

model = None
# different datasets
if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(result_path):
    os.mkdir(result_path)
for data_f in data_folders:
    # different unknown datasets for each dataset
    subset=data_subsets[data_f][0]
    testdata_path=os.path.join(data_path,data_f,testdata_subset)
    test_table=pd.read_csv(testdata_path)
    test_table.drop(columns=test_table.columns[0], axis=1, inplace=True)
    test_dataset=test_table.values

    prev_weights_file = 0
    prev_model = 0
    graph = 0

    print("Dataset: {}".format(subset))
    for i in range(runs):
        np.random.seed(42)
        tf.random.set_seed(42)
        # location of unknwon datasets
        data_name="{}{}".format(subset, i) #"{}_{}_{}".format(subset,contamination_rate,num_knowns)
        unknown_dataname=data_name+".csv"
        undata_path=os.path.join(data_path,data_f,unknown_dataname)
        # get unknown dataset
        table=pd.read_csv(undata_path)
        table.drop(columns=table.columns[0], axis=1, inplace=True)
        undataset=table.values

        print()
        rocs=[]
        prs=[]
        train_times=[]
        test_times=[]
        # run experiment
        print("#######################################################################")
        print("Round: {}".format(i))
        weights_file=os.path.join(model_path,"{}_{}_{}_weights.h4f".format(subset,i,data_name))
        # initialize environment and agent
        #tf.compat.v1.reset_default_graph()

        train_points = []

        if i > 5:
            num_samples, _ = undataset.shape
            feat = 201
            X = undataset[:,:feat]
            Y = undataset[:,feat]

            #with graph.as_default():
            #    with session.as_default():
            #        with session.graph.as_default():
            model.load_weights(prev_weights_file)         
            iforest_scores = DQN_iforest(X, model.qnet)

            X = np.reshape(X, (num_samples, feat))
            preds = model.predict_label(X)

            for j in range(len(undataset)):           
                # Check iForest
                ifs = iforest_scores[j]

                # Check correctness
                pred = preds[j]
                f = open('prediction_se', 'a')
                f.write(f'pred: {pred}, label: {Y[j]}\n')
                f.close() 
                
                if (ifs >= 0.4 and ifs <= 0.6) or (pred != Y[j]):
                    train_points.append(undataset[j])
                
            train_points = np.array(train_points)
        else:
            train_points = undataset

        env=ADEnv(dataset=train_points,
                    sampling_Du=size_sampling_Du,
                    prob_au=prob_au,
                    label_normal=label_normal,
                    label_anomaly=label_anomaly,
                    name=data_name)
        if i > 0:
            model=DPLAN(env=env,
                    settings=settings, weights_file=prev_weights_file)
        else:
            model = DPLAN(env=env, settings=settings)

        prev_weights_file = weights_file
        graph = tf.compat.v1.get_default_graph()
        # train the agent
        train_time=0
        if Train:
            # train DPLAN
            train_start=time.time()
            model.fit(weights_file=weights_file)
            train_end=time.time()
            train_time=train_end-train_start
            print("Train time: {}/s".format(train_time))

        # test the agent
        test_time=0
        if Test:
            test_X, test_y=test_dataset[:,:-1], test_dataset[:,-1]
            model.load_weights(weights_file)
            # test DPLAN
            test_start=time.time()
            pred_y=model.predict_label(test_X)

            for j in range(len(pred_y)):
                f = open('predictions_se', 'a')
                f.write(f'pred: {pred_y[j]}, label: {test_y[j]}\n')
                f.close() 

            test_end=time.time()
            test_time=test_end-test_start
            print("Test time: {}/s".format(test_time))

            roc = roc_auc_score(test_y, pred_y)
            pr = average_precision_score(test_y, pred_y)
            print("{} Run {}: AUC-ROC: {:.4f}, AUC-PR: {:.4f}, train_time: {:.2f}, test_time: {:.2f}".format(subset,
                                                                                                                i,
                                                                                                                roc,
                                                                                                                pr,
                                                                                                                train_time,
                                                                                                                test_time))

            rocs.append(roc)
            prs.append(pr)
            train_times.append(train_time)
            test_times.append(test_time)

            #Draw ROC AUC curve
            fpr, tpr, _ = metrics.roc_curve(test_y,  pred_y)
            plt.plot(fpr,tpr)
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.show()
            plt.savefig(f'ROC1_{i}.png')
            plt.figure().clear()

            #Confusion matrix
            m = confusion_matrix(test_y, pred_y)
            f = open('confusion_se.txt', 'a')
            f.write("Run {} - ".format(i))
            f.write(np.array2string(m))
            f.write('\n')
            f.close()

            prec = precision_score(test_y, pred_y)
            recall = recall_score(test_y, pred_y)
            f1 = f1_score(test_y, pred_y)
            f = open('metrics_se.txt', 'a')
            f.write("Run {} - ".format(i))
            f.write("precision: {}\t recall: {}\t f1: {}".format(prec, recall, f1))
            f.write('\n')
            f.close()


        if Test:
            # write results
            writeResults(i, rocs, prs, train_times, test_times, os.path.join(result_path,result_file))
