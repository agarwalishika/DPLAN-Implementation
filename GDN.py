import torch
import os
import numpy as np
import  scipy.stats
# from    torch.utils.data import DataLoader
from    torch.optim import lr_scheduler
import  random, sys, pickle
import  argparse
from meta import *
from getConfig import modelArch
from data import DataProcessor, task_generator, test_task_generator, test_task_generator_backup
from models import SGC
from sklearn.metrics import auc, roc_curve
from utils import  aucPerformance

run_no = 0

def get_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_name', help='pubmed/yelp', default='yelp')
    argparser.add_argument('--num_epochs', type=int, help='epoch number', default=100)
    argparser.add_argument('--num_epochs_GDN', type=int, help='epoch number for GDN', default=100)
    argparser.add_argument('--gdn_lr', type=float, help='learning rate for GDN', default=0.01)
    argparser.add_argument('--bs', type=int, help='batch size', default=16)
    argparser.add_argument('--num_graph', type=int, help='meta batch size, namely task num', default=2)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.003)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.5)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=3)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=3)
    argparser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    argparser.add_argument('--num_run', type=int, help='run the experiments multiple times', default=10)

    args = argparser.parse_args()
    return args

#Meta-GDN
'''
def score_sample(x):
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    num_labeled_ano = 10 # each graph (auxiliary or target) has 10 sampled anomaly nodes

    results_meta_gdn = []
    for t in range(args.num_run):
        dataset = DataProcessor(num_graph=args.num_graph, degree=2, data_name=args.data_name)
        dataset.data_loader()

        # training meta-gdn
        [feature_list, l_list, ul_list], [target_feature, target_l_idx, target_ul_idx] = dataset.sample_anomaly(num_labeled_ano)

        config = modelArch(feature_list[0].shape[1], 1)

        maml = Meta(args, config).to(device)
        best_val_auc = 0
        for e in range(1, args.num_epochs + 1):

            # training
            maml.train()
            x_train, y_train, x_qry, y_qry = task_generator(feature_list, l_list, ul_list, bs=args.bs, device=device)
            loss = maml(x_train, y_train, x_qry, y_qry)
            torch.save(maml.state_dict(), 'temp.pkl')
            # validation
            model_meta_eval = Meta(args, config).to(device)
            model_meta_eval.load_state_dict(torch.load('temp.pkl'))
            model_meta_eval.eval()
            x_train, y_train, x_val, y_val = test_task_generator(target_feature, target_l_idx,
                                                                   target_ul_idx, args.bs,
                                                                   dataset.target_label,
                                                                   dataset.target_idx_val, device)
            auc_roc, auc_pr, ap = model_meta_eval.evaluate(x_train, y_train, x_val, y_val)

            if auc_roc > best_val_auc: # store the best model
                best_val_auc = auc_roc
                torch.save(maml.state_dict(), 'best_meta_GDN.pkl')

        # testing
        maml = Meta(args, config).to(device)
        maml.load_state_dict(torch.load('best_meta_GDN.pkl'))
        maml.eval()
        x_train, y_train, x_test, y_test = test_task_generator(target_feature, target_l_idx,
                                                               target_ul_idx, args.bs,
                                                               dataset.target_label,
                                                               dataset.target_idx_test, device)
        
        x = torch.Tensor(x)
        x = x.to(device)
        y_pred = maml.predict(x_train, y_train, x)
        return y_pred

'''
#GDN
def score_sample(x):
    print('GDNrunthrough')
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    if os.path.isfile('best_GDN.pkl'):
        model = SGC(8000, 1).to(device)
        model.load_state_dict(torch.load('best_GDN.pkl'))
        model.eval()
        x = torch.Tensor(x)
        x = x.to(device) 
        y_pred = model(x).detach().cpu().numpy()
        return y_pred
    
    
    num_labeled_ano = 10 # each graph (auxiliary or target) has 10 sampled anomaly nodes

    #results_meta_gdn = []
    results_gdn = []
    for t in range(args.num_run):
        dataset = DataProcessor(num_graph=args.num_graph, degree=2, data_name=args.data_name)
        dataset.data_loader()

        [feature_list, l_list, ul_list], [target_feature, target_l_idx, target_ul_idx] = dataset.sample_anomaly(num_labeled_ano)

        config = modelArch(feature_list[0].shape[1], 1)

        #maml = Meta(args, config).to(device)
        best_val_auc = 0

        # GDN training
        model = SGC(target_feature.shape[1], 1).to(device)
        optim = torch.optim.Adam(model.parameters(), lr=args.gdn_lr, weight_decay=0)
        best_val_auc = 0
        for e in range(1, args.num_epochs_GDN + 1):

            x_train, y_train, x_test, y_test = test_task_generator_backup(target_feature, target_l_idx,
                                                                   target_ul_idx, num_labeled_ano * 2,
                                                                   dataset.target_label,
                                                                   dataset.target_idx_test, device)
            x_train, y_train = x_train.to(device), y_train.to(device)
            model.train()
            optim.zero_grad()
            y_pred = model(x_train)
            loss = dev_loss(y_train, y_pred)
            loss.backward()
            optim.step()

            # validation
            _, _, x_val, y_val = test_task_generator_backup(target_feature, target_l_idx,
                                                                   target_ul_idx, num_labeled_ano * 2,
                                                                   dataset.target_label,
                                                                   dataset.target_idx_val, device)
            model.eval()
            y_pred = model(x_val).detach().cpu().numpy()
            y_val = y_val.detach().cpu().numpy()
            auc_roc, auc_pr, ap = aucPerformance(y_val, y_pred)

            if auc_roc > best_val_auc: # store the best model
                best_val_auc = auc_roc
                torch.save(model.state_dict(), 'best_GDN.pkl')

        print(".", end = "")

    # testing
    model = SGC(target_feature.shape[1], 1).to(device)
    model.load_state_dict(torch.load('best_GDN.pkl'))
    model.eval()
    '''_, _, x_test, y_test = test_task_generator_backup(target_feature, target_l_idx,
                                                           target_ul_idx, num_labeled_ano * 2,
                                                           dataset.target_label,
                                                           dataset.target_idx_test, device)'''
    x = torch.Tensor(x)
    x = x.to(device) 
    y_pred = model(x).detach().cpu().numpy()
    return y_pred   

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--data_name', help='pubmed/yelp', default='yelp')
    argparser.add_argument('--num_epochs', type=int, help='epoch number', default=100)
    argparser.add_argument('--num_epochs_GDN', type=int, help='epoch number for GDN', default=100)
    argparser.add_argument('--gdn_lr', type=float, help='learning rate for GDN', default=0.01)
    argparser.add_argument('--bs', type=int, help='batch size', default=16)
    argparser.add_argument('--num_graph', type=int, help='meta batch size, namely task num', default=5)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=0.003)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.5)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=3)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=3)
    argparser.add_argument('--seed', type=int, default=1234, help='Random seed.')
    argparser.add_argument('--num_run', type=int, help='run the experiments multiple times', default=100)

    args = argparser.parse_args()

    main()
