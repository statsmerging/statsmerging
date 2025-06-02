import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import time
import logging
import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F

from task_vectors import TaskVector
from eval import eval_single_dataset_preprocess_head
from args import parse_arguments
from heads import get_classification_head
from datasets.registry import get_dataset
from datasets.common import get_dataloader, maybe_dictionarize, get_dataloader_shuffle

# Set memory optimization settings
torch.backends.cudnn.benchmark = True
torch.backends.cuda.max_split_size_mb = 128

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images):
        return self.model(images)

class LayerwiseAlphaPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_tasks, num_layers):
        super(LayerwiseAlphaPredictor, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_tasks),
                nn.Softmax(dim=-1)
            ) for _ in range(num_layers)
        ])

    def forward(self, x):
        return torch.stack([net(x) for net in self.layers], dim=1)

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        names = name.split(".")
        obj = mod
        for n in names[:-1]:
            obj = getattr(obj, n)
        setattr(obj, names[-1], p)

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def make_functional(mod):
    orig_params = tuple(mod.parameters())
    names = []
    for name, _ in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def create_log_dir(path, filename='log.txt'):
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(os.path.join(path, filename))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

class StatsMergingPP(nn.Module):
    def __init__(self, paramslist, model, names, exam_datasets, args):
        super(StatsMergingPP, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.exam_datasets = exam_datasets
        self.args = args

        num_layers = len(paramslist[0])
        self.alpha_predictor = LayerwiseAlphaPredictor(3, 128, len(paramslist), num_layers).to(args.device)

        for dataset_name in exam_datasets:
            classification_head = get_classification_head(args, dataset_name)
            self.add_module(f'classifier_{dataset_name}', classification_head.to(args.device))

    def get_model_stats(self):
        with torch.no_grad():
            stds, vars, mean_vecs = [], [], []
            for i in range(len(self.paramslist)):
                layer_means = [p.mean() for p in self.paramslist[i]]
                mean_vecs.append(torch.stack(layer_means))

            for layer_params in zip(*self.paramslist):
                stacked = torch.stack(layer_params, dim=0)
                stds.append(stacked.std(dim=0).mean())
                vars.append(stacked.var(dim=0).mean())

            mean_vecs = torch.stack(mean_vecs)
            cos_sim = F.cosine_similarity(mean_vecs[None, :], mean_vecs[:, None], dim=-1).mean()

            return torch.tensor([torch.stack(stds).mean(), torch.stack(vars).mean(), cos_sim], device=self.args.device).unsqueeze(0)

    def get_alphas(self):
        return self.alpha_predictor(self.get_model_stats())  # shape: [1, L, T]

    def get_classification_head(self, dataset_name):
        return getattr(self, f'classifier_{dataset_name}')

    def get_image_encoder(self):
        with torch.no_grad():
            alphas = self.get_alphas()[0]  # [L, T]
            merged_params = []
            for l, params in enumerate(zip(*self.paramslist)):
                weighted_param = torch.zeros_like(params[0])
                for alpha, p in zip(alphas[l], params):
                    weighted_param.add_(alpha * p)
                merged_params.append(weighted_param)
            load_weights(self.model, self.names, merged_params)
            return self.model.to(self.args.device)

    def forward(self, x, dataset_name, training=False):
        if training:
            alphas = self.get_alphas()[0]
        else:
            with torch.no_grad():
                alphas = self.get_alphas()[0]
        merged_params = []
        for l, params in enumerate(zip(*self.paramslist)):
            weighted_param = torch.zeros_like(params[0], device=self.args.device)
            for alpha, p in zip(alphas[l], params):
                weighted_param = weighted_param + alpha * p if training else weighted_param.add_(alpha * p)
            merged_params.append(weighted_param)
        load_weights(self.model, self.names, merged_params)
        features = self.model(x.to(self.args.device))
        return self.get_classification_head(dataset_name)(features)

def main():
    # exam_datasets = ['EuroSAT', 'RESISC45']
    exam_datasets = ['Cars','RESISC45','SUN397', 'EuroSAT','SVHN', 'GTSRB', 'MNIST', 'DTD']
    model_name = 'ViT-B-32'
    args = parse_arguments()

    args.data_location = 'Datasets/mm/ModelMergingBaseline16Datasets/'
    args.model = model_name
    args.save = f'checkpoints-Layer-wise-8model/{model_name}'
    args.logs_path = f'logs_plot-8model/{model_name}'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    str_time = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log = create_log_dir(args.logs_path, f'log_{str_time}_StatsMergingPP.txt')
    args.log = log

    pretrained_path = 'Datasets/models/task_vectors_checkpoints/ViT-B-32/zeroshot.pt'
    pretrained_model = torch.load(pretrained_path)
    model = ModelWrapper(pretrained_model).to(args.device)
    orig_params, names = make_functional(model)

    task_vectors = [
        TaskVector(pretrained_path, f'Datasets/models/task_vectors_checkpoints/{model_name}/{d}/finetuned.pt')
        for d in exam_datasets
    ]

    paramslist = [tuple(p.detach().to(args.device).requires_grad_() for p in orig_params)]
    paramslist += [tuple(p.detach().to(args.device).requires_grad_() for p in tv.vector.values()) for tv in task_vectors]
    torch.cuda.empty_cache()

    StatsMergingPP = StatsMergingPP(paramslist, model, names, exam_datasets, args).to(args.device)

    optimizer = torch.optim.Adam(StatsMergingPP.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    # Initial eval
    log.info("Initial evaluation:")
    total_acc = 0.
    for dataset_name in exam_datasets:
        image_encoder = StatsMergingPP.get_image_encoder()
        classifier = StatsMergingPP.get_classification_head(dataset_name)
        metrics = eval_single_dataset_preprocess_head(image_encoder, classifier, dataset_name, args)
        total_acc += metrics['top1']
        log.info(f"{dataset_name} ACC: {metrics['top1']}")
    log.info(f"Avg ACC: {total_acc/len(exam_datasets)}\n")

    best_avg_acc, best_epoch, best_model_state = -1, -1, None

    for epoch in range(650):
        StatsMergingPP.train()
        total_loss = 0.

        for dataset_name in exam_datasets:
            dataset = get_dataset(dataset_name, pretrained_model.val_preprocess, location=args.data_location, batch_size=80)
            dataloader = get_dataloader_shuffle(dataset)
            for data in tqdm.tqdm(dataloader, desc=f"Epoch {epoch} {dataset_name}"):
                data = maybe_dictionarize(data)
                x, y = data['images'].to(args.device), data['labels'].to(args.device)
                optimizer.zero_grad()
                outputs = StatsMergingPP(x, dataset_name, training=True)
                loss = F.cross_entropy(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(StatsMergingPP.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

        log.info(f"Epoch {epoch} Total Loss: {total_loss}")
        scheduler.step(total_loss)


    # If you move this inside, you would get same as our log results but very time consuming as you increase number of models, 
    #  suggested to keep this after some 150 above epoch for best outputs.
    StatsMergingPP.eval()
    with torch.no_grad():
        alphas = StatsMergingPP.get_alphas()[0].cpu().numpy()
        log.info(f"Alphas: {alphas.tolist()}")
        total_acc, accs_this_epoch = 0., {}
        for dataset_name in exam_datasets:
            image_encoder = StatsMergingPP.get_image_encoder()
            classifier = StatsMergingPP.get_classification_head(dataset_name)
            metrics = eval_single_dataset_preprocess_head(image_encoder, classifier, dataset_name, args)
            acc = metrics['top1']
            accs_this_epoch[dataset_name] = acc
            total_acc += acc
            log.info(f"Epoch {epoch} {dataset_name} ACC: {acc}")
        avg_acc = total_acc / len(exam_datasets)
        log.info(f"Epoch {epoch} Avg ACC: {avg_acc}\n")

        if avg_acc > best_avg_acc:
            best_avg_acc = avg_acc
            best_epoch = epoch
            best_model_state = {
                'epoch': epoch,
                'model': StatsMergingPP.state_dict(),
                'optimizer': optimizer.state_dict(),
                'alphas': alphas,
                'avg_acc': avg_acc,
                'dataset_accs': accs_this_epoch
            }
            torch.save(best_model_state, args.save+"/best_model.pt")

if __name__ == "__main__":
    main()

