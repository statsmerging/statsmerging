import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import sys
# sys.path.append('/media/brcao/eData4TB05/ranjith/ModelMerging/')
# print(sys.paths)
from task_vectors import TaskVector
from eval import eval_single_dataset
from args import parse_arguments

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

exam_datasets = ['Cars', 'RESISC45', 'EuroSAT', 'GTSRB'] #, 'MNIST', 'DTD'] # SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD


#/media/brcao/eData4TB05/ranjith/NAFNet/models/task_vectors_checkpoints/ViT-B-32/Cars/finetuned.pt
#/media/brcao/eData4TB05/ranjith/NAFNet/models/task_vectors_checkpoints/ViT-B-32/DTD/finetuned.pt

# Epoch 127 RESISC45 ACC: 0.8674603174603175
# Epoch 127 EuroSAT ACC: 0.8518518518518519
# Epoch 127 GTSRB ACC: 0.9566904196357878
# Epoch 127 MNIST ACC: 0.99
# Epoch 127 DTD ACC: 0.6728723404255319
# Epoch 127 Avg ACC: 0.8677749858746978

# exam_datasets =  ['Cars', 'GTSRB']
exam_datasets =  ['RESISC45','EuroSAT']
exam_datasets = ['SVHN', 'EuroSAT']
exam_datasets = ['RESISC45', 'EuroSAT','Cars', 'GTSRB']

model = 'ViT-B-32'
args = parse_arguments()
args.data_location = '/home/brcao/Repos/merge_model/Datasets/mm/ModelMergingBaseline16Datasets/'
# '/media/brcao/eData4TB05/brcao/Data/datasets/mm/'
args.model = model
args.save = 'checkpoints-comp/' + model
args.logs_path = 'logs-comp/' + model
pretrained_checkpoint = '/home/brcao/Repos/merge_model/Datasets/models/task_vectors_checkpoints/ViT-B-32/zeroshot.pt'



# /media/brcao/eData4TB05/ranjith/NAFNet/models/task_vectors_checkpoints/ViT-B-32/head_SUN397.pt
str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
log = create_log_dir(args.logs_path, 'log_{}_task_arithmetic.txt'.format(str_time_))

task_vectors = [
    TaskVector(pretrained_checkpoint, '/home/brcao/Repos/merge_model/Datasets/models/task_vectors_checkpoints/ViT-B-32/'+dataset_name+'/finetuned.pt') for dataset_name in exam_datasets
]

task_vector_sum = sum(task_vectors)

scaling_coef_ = 0.3

image_encoder = task_vector_sum.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef_)
log.info('*'*20 + 'scaling_coef:' + str(scaling_coef_) + '*'*20)

accs = []
for dataset in exam_datasets:
    metrics = eval_single_dataset(image_encoder, dataset, args)
    log.info(str(dataset) + ':' + str(metrics.get('top1')*100)+'%')
    accs.append(metrics.get('top1')*100)
log.info('Avg ACC:' + str(np.mean(accs)) + '%')
