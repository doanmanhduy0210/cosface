
import torch 
import confi_evaluate 
import numpy as np
import math
import argparse
from sklearn import metrics
from losses.ArcFaceLossMargin import ArcFaceLossMargin
from losses.CosFaceLossMargin import CosFaceLossMargin
from losses.CombinedLossMargin import CombinedLossMargin
from losses.CenterLoss import CenterLoss
from losses.FocalLoss import FocalLoss
from dataset.get_data import get_data
from models.resnet import *
from models.irse import *
from helpers import *
from datetime import datetime, timedelta
import time
from accuracy_model import evaluate_forward_pass_accuracy
from evaluate import * 


def test( model, device, test_loader, total_loss, loss_criterion):
    print('test processing.... \n')
    correct = 0
    model.eval()
    t = time.time()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            feats = model(data)

            if confi_evaluate.criterion_type == 'arcface':
                logits = loss_criterion(feats, target)
                outputs = logits
            elif confi_evaluate.criterion_type == 'cosface':
                logits, _ = loss_criterion(feats, target)
                outputs = logits
            elif confi_evaluate.criterion_type == 'combined':
                logits = loss_criterion(feats, target)
                outputs = logits
            elif confi_evaluate.criterion_type == 'centerloss':
                _, outputs = loss_criterion(feats, target)

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == target.data).sum()
            print('.',end="")
            
    accuracy = 100. * correct / len(test_loader.dataset)
    nonaccuracy = 100.0*(len(test_loader.dataset) - correct)/(len(test_loader.dataset))  
    print('\n Test set:, Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),accuracy))
    time_for_test = int(time.time() - t)
    print( 'time for test: ' + str(time_for_test))
    print('Total time for test: {}'.format(timedelta(seconds=time_for_test)))
    log = 'test accuracy: ({}-{}/{}) [(acc) {}% :(non acc) {}%] \t |'.format (correct,len(test_loader.dataset)- correct ,len(test_loader.dataset),accuracy, nonaccuracy)
    print(log)
    

def evaluate(validation_data_dic, model, device, distance_metric):
    print('evaluate processing.....')
    embedding_size = confi_evaluate.features_dim

    for val_type in confi_evaluate.validations:
        dataset = validation_data_dic[val_type+'_dataset']
        loader = validation_data_dic[val_type+'_loader']

        model.eval()
        t = time.time()
        print('\n\nRunnning forward pass on {} images'.format(val_type))

        accuracy= evaluate_forward_pass_accuracy(model, 
                                                loader, 
                                                dataset, 
                                                embedding_size, 
                                                device,
                                                distance_metric=distance_metric, 
                                                subtract_mean=confi_evaluate.evaluate_subtract_mean)
        print("Evaluate accuracy: " + str(accuracy))
        log = "(acc) %2.5f+-%2.5f "%(np.mean(accuracy),np.std(accuracy))  
        print(log)

            

def main():
# load model 
    use_cuda = torch.cuda.is_available() 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print("Use CUDA: " + str(use_cuda))
    model = get_model(confi_evaluate.model_type, confi_evaluate.input_size)
    if use_cuda: 
        model.load_state_dict(torch.load(confi_evaluate.model_path))
    else: 
        model.load_state_dict(torch.load(confi_evaluate.model_path, map_location='cpu')) 
    model.to(device) 
    model.eval()

    validation_paths_dic = {
                    "LFW" : confi_evaluate.lfw_dir,
                    "CALFW" : confi_evaluate.calfw_dir,
                    "CPLFW" : confi_evaluate.cplfw_dir,
                    "CFP_FF" : confi_evaluate.cfp_ff_dir,
                    "CFP_FP" : confi_evaluate.cfp_fp_dir
                    }
    print("Validation_paths_dic: " + str(validation_paths_dic))
    validation_data_dic = {}
    for val_type in confi_evaluate.validations:
        print( 'Init dataset and loader for validation type: {}'.format(val_type))
        dataset, loader = get_evaluate_dataset_and_loader(root_dir=validation_paths_dic[val_type], 
                                                                type=val_type, 
                                                                num_workers=confi_evaluate.num_workers, 
                                                                input_size=confi_evaluate.input_size, 
                                                                batch_size=confi_evaluate.batch_size)
        validation_data_dic[val_type+'_dataset'] = dataset
        validation_data_dic[val_type+'_loader'] = loader

    test_loader, _ = get_data(confi_evaluate, device)

    if confi_evaluate.total_loss_type == 'softmax':
        total_loss = nn.CrossEntropyLoss().to(device)
    elif confi_evaluate.total_loss_type == 'focal':
        total_loss = FocalLoss().to(device)
    else:
        raise AssertionError('Unsuported total_loss_type {}. We only support:  [\'softmax\', \'focal\']'.format(confi_evaluate.total_loss_type))

    ####### Criterion setup
    print('Criterion type: %s' % confi_evaluate.criterion_type)
    if confi_evaluate.criterion_type == 'arcface':
        distance_metric = confi_evaluate.distance_metric
        loss_criterion = ArcFaceLossMargin(num_classes=test_loader.dataset.num_classes, feat_dim=confi_evaluate.features_dim, device=device, s=confi_evaluate.margin_s, m=confi_evaluate.margin_m).to(device)
    elif confi_evaluate.criterion_type == 'cosface':
        distance_metric = confi_evaluate.distance_metric
        loss_criterion = CosFaceLossMargin(num_classes=test_loader.dataset.num_classes, feat_dim=confi_evaluate.features_dim, device=device, s=confi_evaluate.margin_s, m=confi_evaluate.margin_m).to(device)
    elif confi_evaluate.criterion_type == 'combined':
        distance_metric = confi_evaluate.distance_metric
        loss_criterion = CombinedLossMargin(num_classes=test_loader.dataset.num_classes, feat_dim=confi_evaluate.features_dim, device=device, s=confi_evaluate.margin_s, m1=confi_evaluate.margin_m1, m2=confi_evaluate.margin_m2).to(device)
    elif confi_evaluate.criterion_type == 'centerloss':
        distance_metric = confi_evaluate.distance_metric
        loss_criterion = CenterLoss(device=device, num_classes=test_loader.dataset.num_classes, feat_dim=confi_evaluate.features_dim, use_gpu=use_cuda)
    else:
        raise AssertionError('Unsuported criterion_type {}. We only support:  [\'arcface\', \'cosface\', \'combined\', \'centerloss\']'.format(confi_evaluate.criterion_type))
    
    test( model, device, test_loader, total_loss, loss_criterion)
    evaluate( validation_data_dic, model, device, confi_evaluate.distance_metric)
    
                           
if __name__=='__main__': 
    main()                                                       
