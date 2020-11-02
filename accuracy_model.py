
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
from PIL import Image
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms as T
import torchvision
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate
import argparse
from evaluate_helpers import *
from helpers import get_model

from pdb import set_trace as bp

import os
import numpy as np

from scipy import interpolate
import math


def calculate_accuracy(threshold, dist, actual_issame):
    predict_issame = np.less(dist, threshold)
    tp = np.sum(np.logical_and(predict_issame, actual_issame))
    # fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    tn = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    # fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    acc = float(tp+tn)/dist.size
    return acc


def evaluate_forward_pass_accuracy(model, lfw_loader, lfw_dataset, embedding_size, device, distance_metric, subtract_mean):

    nrof_images = lfw_dataset.nrof_embeddings

    emb_array = np.zeros((nrof_images, embedding_size))
    lab_array = np.zeros((nrof_images,))
    with torch.no_grad():
        for i, (data, label) in enumerate(lfw_loader):

            data, label = data.to(device), label.to(device)

            feats = model(data)
            emb = feats.cpu().numpy()
            lab = label.detach().cpu().numpy()

            lab_array[lab] = lab
            emb_array[lab, :] = emb

            if i % 10 == 9:
                print('.', end='')
                sys.stdout.flush()
        print('')
    embeddings = emb_array

    assert np.array_equal(lab_array, np.arange(nrof_images))==True, 'Wrong labels used for evaluation, possibly caused by training examples left in the input pipeline'
    accuracy = evaluate(embeddings, lfw_dataset.actual_issame, distance_metric=distance_metric, subtract_mean=subtract_mean)
    
    return accuracy




def evaluate(embeddings, actual_issame, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    accuracy = calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), distance_metric=distance_metric, subtract_mean=subtract_mean)
    
    return accuracy

def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric 
        
    return dist

def calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, distance_metric=0, subtract_mean=False):
    assert(embeddings1.shape[0] == embeddings2.shape[0])
    assert(embeddings1.shape[1] == embeddings2.shape[1])
    nrof_pairs = min(len(actual_issame), embeddings1.shape[0])
    nrof_thresholds = len(thresholds)

    if subtract_mean:
        mean = np.mean(np.concatenate([embeddings1, embeddings2]), axis=0)
    else:
        mean = 0.0
    dist = distance(embeddings1-mean, embeddings2-mean, distance_metric)
    
    # Find the best threshold for the fold
    acc_train = np.zeros((nrof_thresholds))
    for threshold_idx, threshold in enumerate(thresholds):
        acc_train[threshold_idx] = calculate_accuracy(threshold, dist, actual_issame)
    best_threshold_index = np.argmax(acc_train)
    
    accuracy = calculate_accuracy(thresholds[best_threshold_index], dist, actual_issame)

    return accuracy

