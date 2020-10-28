from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
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
from evaluate import *
from datetime import datetime, timedelta
import time
from logger import Logger
from pdb import set_trace as bp
import confi 

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False





def train( model, device, train_loader, total_loss, loss_criterion, optimizer, log_file_path, model_dir, logger, epoch):
    model.train()
    t = time.time()
    log_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        tt = time.time()

        data, target = data.to(device), target.to(device)

        # Forward prop.
        features = model(data)

        if confi.criterion_type == 'arcface':
            logits = loss_criterion(features, target)
            loss = total_loss(logits, target)
        elif confi.criterion_type == 'cosface':
            logits, mlogits = loss_criterion(features, target)
            loss = total_loss(mlogits, target)
        elif confi.criterion_type == 'combined':
            logits = loss_criterion(features, target)
            loss = total_loss(logits, target)
        elif confi.criterion_type == 'centerloss':
            weight_cent = 1.
            loss_cent, outputs = loss_criterion(features, target)
            loss_cent *= weight_cent
            los_softm = total_loss(outputs, target)
            loss = los_softm + loss_cent

        optimizer.zero_grad()

        if APEX_AVAILABLE:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if confi.criterion_type == 'centerloss':
            # by doing so, weight_cent would not impact on the learning of centers
            for param in loss_criterion.parameters():
                param.grad.data *= (1. / weight_cent)

        # Update weights
        optimizer.step()

        time_for_batch = int(time.time() - tt)
        time_for_current_epoch = int(time.time() - t)
        percent = 100. * batch_idx / len(train_loader)

        log = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} \tbatch_time: {}   Total time for epoch: {}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            percent, loss.item(), timedelta(seconds=time_for_batch), timedelta(seconds=time_for_current_epoch))
        # print_and_log(log_file_path, log)
        print(log)

        log_loss = loss.item()

        # loss_epoch_and_percent - last two digits - Percent of epoch completed
        logger.scalar_summary("loss_epoch_and_percent", log_loss, (epoch*100)+(100. * batch_idx / len(train_loader)))

    logger.scalar_summary("loss", log_loss, epoch)

    time_for_epoch = int(time.time() - t)
    print('Total time for epoch: {}'.format(timedelta(seconds=time_for_epoch)))
    # print_and_log(log_file_path, 'Total time for epoch: {}'.format(timedelta(seconds=time_for_epoch)))

    if epoch % confi.model_save_interval == 0 or epoch == confi.epochs:
        save_model(confi, confi.model_type, model_dir, model, log_file_path, epoch)
        save_model(confi, confi.criterion_type, model_dir, loss_criterion, log_file_path, epoch)
     

def test( model, device, test_loader, total_loss, loss_criterion, log_file_path, logger, epoch):
    if test_loader == None:
        return
    model.eval()
    correct = 0
    if epoch % confi.test_interval == 0 or epoch == confi.epochs:
        model.eval()
        t = time.time()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)

                feats = model(data)

                if confi.criterion_type == 'arcface':
                    logits = loss_criterion(feats, target)
                    outputs = logits
                # elif confi.criterion_type == 'arcface2':
                #     logits = loss_criterion(feats, target)
                #     outputs = logits
                elif confi.criterion_type == 'cosface':
                    logits, _ = loss_criterion(feats, target)
                    outputs = logits
                elif confi.criterion_type == 'combined':
                    logits = loss_criterion(feats, target)
                    outputs = logits
                elif confi.criterion_type == 'centerloss':
                    _, outputs = loss_criterion(feats, target)

                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == target.data).sum()
                

        accuracy = 100. * correct / len(test_loader.dataset)
        nonaccuracy = 100.0 - accuracy 
        log = '\nTest set:, Accuracy: {}/{} ({:.0f}%)\n'.format(correct, len(test_loader.dataset),accuracy)
        print(log)
        log = '\t {} \t |'.format(epoch)+'({}-{}/{}) [(acc) {}% :(non acc) {}%] \t |'.format (correct,len(test_loader.dataset)- correct ,len(test_loader.dataset),accuracy, 100.0 - accuracy) 
        # log = \t {} \t |'.format(1)+'({}-{}/{}) [(acc) {}% : (non acc){}%] \t |'.format (1,10000 - 1 ,10000,55, 45)
        logger.scalar_summary("accuracy", accuracy, epoch)
        time_for_test = int(time.time() - t)
        print(time_for_test)
        print('Total time for test: {}'.format(timedelta(seconds=time_for_test)))
        # print_and_log(log_file_path, 'Total time for test: {}'.format(timedelta(seconds=time_for_test)))
        return log

def evaluate(validation_data_dic, model, device, log_file_path, logger, distance_metric, epoch):
    if epoch % confi.evaluate_interval == 0 or epoch == confi.epochs:
        embedding_size = confi.features_dim

        for val_type in confi.validations:
            dataset = validation_data_dic[val_type+'_dataset']
            loader = validation_data_dic[val_type+'_loader']

            model.eval()
            t = time.time()
            print('\n\nRunnning forward pass on {} images'.format(val_type))

            tpr, fpr, accuracy, val, val_std, far = evaluate_forward_pass(model, 
                                                                        loader, 
                                                                        dataset, 
                                                                        embedding_size, 
                                                                        device,
                                                                        lfw_nrof_folds=confi.evaluate_nrof_folds, 
                                                                        distance_metric=distance_metric, 
                                                                        subtract_mean=confi.evaluate_subtract_mean)
            auc = metrics.auc(fpr, tpr)
            log =  "(acc) %2.5f+-%2.5f "%(np.mean(accuracy),np.std(accuracy)) + '(AUC): %1.3f' % auc
            print("Evaluate accuracy: " + log)
            # print_and_log(log_file_path, '\nEpoch: '+str(epoch))
            # print_and_log(log_file_path, 'Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
            
            # print_and_log(log_file_path, 'Area Under Curve (AUC): %1.3f' % auc)
            # eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
            # print('Equal Error Rate (EER): %1.3f' % eer)
            # time_for_val = int(time.time() - t)
            # print_and_log(log_file_path, 'Total time for {} evaluation: {}'.format(val_type, timedelta(seconds=time_for_val)))
                
            logger.scalar_summary(val_type +"_accuracy", np.mean(accuracy), epoch)
            return log


def main():

    # Dirs
    subdir = datetime.strftime(datetime.now(), '%Y-%m-%d___%H-%M-%S')
    out_dir = os.path.join(os.path.expanduser(confi.out_dir), subdir)
    if not os.path.isdir(out_dir):  # Create the out directory if it doesn't exist
        os.makedirs(out_dir)
    model_dir = os.path.join(os.path.expanduser(out_dir), 'model')
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    tensorboard_dir = os.path.join(os.path.expanduser(out_dir), 'tensorboard')
    if not os.path.isdir(tensorboard_dir):  # Create the tensorboard directory if it doesn't exist
        os.makedirs(tensorboard_dir)

    # Write arguments to a text file
    write_arguments_to_file(confi, os.path.join(out_dir, 'arguments.txt'))
        
    # Store some git revision info in a text file in the log directory
    src_path,_ = os.path.split(os.path.realpath(__file__))
    store_revision_info(src_path, out_dir, ' '.join(sys.argv))

    log_file_path = os.path.join(out_dir, 'history_log.txt')
    logger = Logger(tensorboard_dir)

    ################### Pytorch: ###################
    print_and_log(log_file_path, "Pytorch version:  " + str(torch.__version__))
    use_cuda = torch.cuda.is_available()
    print_and_log(log_file_path, "Use CUDA: " + str(use_cuda))
    print_and_log(log_file_path, "Cuda Version:  " + str(torch.version.cuda))
    print_and_log(log_file_path, "cudnn enabled:  " + str(torch.backends.cudnn.enabled))
    print_and_log(log_file_path, "Use APEX: " + str(APEX_AVAILABLE))
    if APEX_AVAILABLE:
            print_and_log(log_file_path, "APEX level: " + str(confi.apex_opt_level))
    device = torch.device("cuda" if use_cuda else "cpu")

    ####### Data setup
    print('Data directory: %s' % confi.data_dir)
    train_loader, test_loader = get_data(confi, device)

    ######## Validation Data setup
    validation_paths_dic = {"LFW" : confi.lfw_dir}
    print_and_log(log_file_path, "Validation_paths_dic: " + str(validation_paths_dic))
    print_and_log(log_file_path, "Training_paths_dic: "   + str(confi.data_dir))
    validation_data_dic = {}
    for val_type in confi.validations:
        print_and_log(log_file_path, 'Init dataset and loader for validation type: {}'.format(val_type))
        dataset, loader = get_evaluate_dataset_and_loader(root_dir=validation_paths_dic[val_type], 
                                                                type=val_type, 
                                                                num_workers=confi.num_workers, 
                                                                input_size=confi.input_size, 
                                                                batch_size=confi.evaluate_batch_size)
        validation_data_dic[val_type+'_dataset'] = dataset
        validation_data_dic[val_type+'_loader'] = loader
    

    ####### Model setup
    print('Model type: %s' % confi.model_type)
    model = get_model(confi.model_type, confi.input_size)
    if confi.model_path != None:
        if use_cuda:
            model.load_state_dict(torch.load(confi.model_path))
        else:
            model.load_state_dict(torch.load(confi.model_path, map_location='cpu'))
    model = model.to(device)

    if confi.total_loss_type == 'softmax':
        total_loss = nn.CrossEntropyLoss().to(device)
    elif confi.total_loss_type == 'focal':
        total_loss = FocalLoss().to(device)
    else:
        raise AssertionError('Unsuported total_loss_type {}. We only support:  [\'softmax\', \'focal\']'.format(confi.total_loss_type))

    ####### Criterion setup
    print('Criterion type: %s' % confi.criterion_type)
    if confi.criterion_type == 'arcface':
        distance_metric = 1
        loss_criterion = ArcFaceLossMargin(num_classes=train_loader.dataset.num_classes, feat_dim=confi.features_dim, device=device, s=confi.margin_s, m=confi.margin_m).to(device)
    # elif confi.criterion_type == 'arcface2':
    #     distance_metric = 1
    #     loss_criterion = ArcFaceLossMargin2(num_classes=train_loader.dataset.num_classes, feat_dim=confi.features_dim, device=device, s=confi.margin_s, m=confi.margin_m).to(device)
    elif confi.criterion_type == 'cosface':
        distance_metric = 1
        loss_criterion = CosFaceLossMargin(num_classes=train_loader.dataset.num_classes, feat_dim=confi.features_dim, device=device, s=confi.margin_s, m=confi.margin_m).to(device)
    elif confi.criterion_type == 'combined':
        distance_metric = 1
        loss_criterion = CombinedLossMargin(num_classes=train_loader.dataset.num_classes, feat_dim=confi.features_dim, device=device, s=confi.margin_s, m1=confi.margin_m1, m2=confi.margin_m2).to(device)
    elif confi.criterion_type == 'centerloss':
        distance_metric = 0
        loss_criterion = CenterLoss(device=device, num_classes=train_loader.dataset.num_classes, feat_dim=confi.features_dim, use_gpu=use_cuda)
    else:
        raise AssertionError('Unsuported criterion_type {}. We only support:  [\'arcface\', \'cosface\', \'combined\', \'centerloss\']'.format(confi.criterion_type))

    if confi.loss_path != None:
        if use_cuda:
            loss_criterion.load_state_dict(torch.load(confi.loss_path))
        else:
            loss_criterion.load_state_dict(torch.load(confi.loss_path, map_location='cpu'))


    if confi.optimizer_type == 'sgd_bn':
        ##################
        if confi.model_type.find("IR") >= 0:
            model_params_only_bn, model_params_no_bn = separate_irse_bn_paras(
                model)  # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
            _, loss_params_no_bn = separate_irse_bn_paras(loss_criterion)
        else:
            model_params_only_bn, model_params_no_bn = separate_resnet_bn_paras(
                model)  # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
            _, loss_params_no_bn = separate_resnet_bn_paras(loss_criterion)

        optimizer = optim.SGD([{'params': model_params_no_bn + loss_params_no_bn, 'weight_decay': confi.weight_decay}, 
                            {'params': model_params_only_bn}], lr = confi.lr, momentum = confi.momentum)

    elif confi.optimizer_type == 'adam':
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': loss_criterion.parameters()}],
                                         lr=confi.lr, betas=(confi.beta1, 0.999))
    elif confi.optimizer_type == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': loss_criterion.parameters()}],
                                        lr=confi.lr, momentum=confi.momentum, weight_decay=confi.weight_decay)
    else:
        raise AssertionError('Unsuported optimizer_type {}. We only support:  [\'sgd_bn\',\'adam\',\'sgd\']'.format(confi.optimizer_type))


    if APEX_AVAILABLE:
        if confi.apex_opt_level==0:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level="O0", loss_scale=1.0
            )
        elif confi.apex_opt_level==1:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level="O1", loss_scale="dynamic"
            )
        elif confi.apex_opt_level==2:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level="O2", 
                keep_batchnorm_fp32=True, loss_scale="dynamic"
            )
        else:
            raise AssertionError('Unsuported apex_opt_level {}. We only support:  [0, 1, 2]'.format(confi.apex_opt_level))
    print_and_log(log_file_path,"[" + str(datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')) + "] ")
    print_and_log(log_file_path,"---------------------------------------------------------------------------")
    print_and_log(log_file_path, "\t Epoch \t |Test closeset \t \t \t \t | Evaluate openset \t \t \t") 
    for epoch in range(1, confi.epochs + 1):
        schedule_lr(confi, log_file_path, optimizer, epoch)
        logger.scalar_summary("lr", optimizer.param_groups[0]['lr'], epoch)

        train(model, device, train_loader, total_loss, loss_criterion, optimizer, log_file_path, model_dir, logger, epoch)
        log = test( model, device, test_loader, total_loss, loss_criterion, log_file_path, logger, epoch)
        log1 = evaluate( validation_data_dic, model, device, log_file_path, logger, distance_metric, epoch)
        print_and_log(log_file_path, log + log1)
if __name__ == '__main__':
    main()
