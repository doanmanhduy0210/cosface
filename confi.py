# out 
out_dir = ''
root_dir = '/home/minglee/Documents/aiProjects/facerec/data/lfw_112'
# training
epochs  = 5
# data 
input_size = [112,112]
data_dir = '/home/minglee/Documents/aiProjects/facerec/data/CASIA-WebFace_160'
num_workers = 8
batch_size =  32 


batch_size_test = 32
evaluate_batch_size = 32
validation_set_split_ratio = 0.05
evaluate_set_split_ratio   = 0.2
min_nrof_val_images_per_class = 0
# model 
model_path = None
model_type = 'IR_50'
# 'ResNet_50'
features_dim = 512 
# optimizer 
optimizer_type = 'sgd_bn'
lr = 0.1
lr_schedule_steps = [30, 55, 75]
lr_gamma = 0.1
beta1 = 0.05
weight_decay = 0.005
momentum = 0.9
# loss 
total_loss_type = 'softmax'
criterion_type = 'cosface'
loss_path = None
margin_s =  32.0

margin_m = 0.5
margin_m1 = 0.2
margin_m2 = 0.35
apex_opt_level = 2 
# intervals 
model_save_interval = 1
model_save_latest_path = None
test_interval = 1
evaluate_interval = 1
# validation 

validations = ['LFW', 'CALFW', 'CPLFW', 'CFP_FF', 'CFP_FP']
lfw_dir = '/home/minglee/Documents/aiProjects/facerec/data/lfw_112'
# '/home/minglee/Documents/aiProjects/facerec/data/__MACOSX/lfw_112/._pairs_LFW.txt'
calfw_dir = '/home/minglee/Documents/aiProjects/facerec/data/calfw_112'
cplfw_dir = '/home/minglee/Documents/aiProjects/facerec/data/cplfw_112'
cfp_ff_dir = '/home/minglee/Documents/aiProjects/facerec/data/cfp_112'
cfp_fp_dir = '/home/minglee/Documents/aiProjects/facerec/data/cfp_112'

evaluate_subtract_mean = False
evaluate_batch_size = 100
evaluate_nrof_folds = 10






