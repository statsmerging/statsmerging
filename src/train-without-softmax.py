import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from task_vectors import TaskVector
from eval import eval_single_dataset_preprocess_head
from args import parse_arguments
from heads import get_classification_head
from datasets.registry import get_dataset
from datasets.common import get_dataloader_shuffle, maybe_dictionarize

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
torch.backends.cudnn.benchmark = True
torch.backends.cuda.max_split_size_mb = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')
    def forward(self, images):
        return self.model(images)

class LayerwiseAlphaPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_additional_tasks, num_layers):
        super().__init__()
        self.num_additional_tasks = num_additional_tasks
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_additional_tasks),
                nn.ReLU(), #Optional this actually didn't affected the performance.
                # nn.Softmax(dim=-1)
            )
            self._init_weights(layer)
            self.layers.append(layer)

    def _init_weights(self, layer):
        with torch.no_grad():
            for name, param in layer.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    param.fill_(0.0)

    def forward(self, x):
        return torch.stack([layer(x[i]) for i, layer in enumerate(self.layers)], dim=0)

def make_functional(mod):
    orig_params, names = tuple(mod.parameters()), []
    for name, _ in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name)
    return orig_params, names

def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])

def load_weights(mod, names, params):
    for name, p in zip(names, params):
        parts = name.split(".")
        obj = mod
        for n in parts[:-1]:
            obj = getattr(obj, n)
        setattr(obj, parts[-1], p)

def create_log_dir(path, filename='log.txt'):
    os.makedirs(path, exist_ok=True)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(path, filename))
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        logger.addHandler(fh)
        logger.addHandler(ch)
    return logger

class StatsMerging(nn.Module):
    def __init__(self, paramslist, model, names, exam_datasets, args):
        super().__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        self.exam_datasets = exam_datasets
        self.args = args
        num_layers = len(paramslist[0])
        self.alpha_predictor = LayerwiseAlphaPredictor(
            input_dim=6, hidden_dim=256,
            num_additional_tasks=len(paramslist)-1,
            num_layers=num_layers
        ).to(args.device)

        for dataset_name in exam_datasets:
            head = get_classification_head(args, dataset_name)
            self.add_module(f'classifier_{dataset_name}', head.to(args.device))

    def get_alphas(self):
        layer_inputs = []
        for layer_idx in range(len(self.paramslist[0])):
            try:
                weights = [task[layer_idx] for task in self.paramslist[1:]]
                stacked = torch.stack(weights, dim=0)
                flat = stacked.view(len(weights), -1).float()

                mean = flat.mean().unsqueeze(0)
                std = flat.std().unsqueeze(0)
                var = flat.var().unsqueeze(0)
                magnitude = flat.norm(p=2).unsqueeze(0)

                try:
                    u, s, v = torch.svd(flat)
                    top_sv = s[0].unsqueeze(0)
                    rank = (s > 1e-3).float().sum().unsqueeze(0)
                except:
                    top_sv = torch.zeros(1, device=self.args.device)
                    rank = torch.zeros(1, device=self.args.device)

                stats = torch.cat([mean, std, var, magnitude, top_sv, rank], dim=0)
            except:
                stats = torch.zeros(6, device=self.args.device)
            layer_inputs.append(stats)

        inputs = torch.stack(layer_inputs, dim=0)
        additional_alphas = self.alpha_predictor(inputs.to(self.args.device))
        full_alphas = torch.ones(len(self.paramslist[0]), len(self.paramslist), device=self.args.device)
        full_alphas[:, 1:] = additional_alphas
        return full_alphas.unsqueeze(0)

    def get_classification_head(self, dataset_name):
        return getattr(self, f'classifier_{dataset_name}')

    def get_image_encoder(self):
        with torch.no_grad():
            alphas = self.get_alphas()[0].detach()
            merged_params = []
            for l, params in enumerate(zip(*self.paramslist)):
                weighted_param = params[0].detach().clone()
                for alpha, p in zip(alphas[l, 1:], params[1:]):
                    weighted_param.add_(alpha * p)
                merged_params.append(weighted_param)
            load_weights(self.model, self.names, merged_params)
            return self.model.to(self.args.device)

    def forward(self, x, dataset_name, training=False):
        with torch.no_grad():
            alphas = self.get_alphas()[0].detach()
            merged_params = []
            for l, params in enumerate(zip(*self.paramslist)):
                weighted_param = params[0].detach().clone()
                for alpha, p in zip(alphas[l, 1:], params[1:]):
                    weighted_param.add_(alpha * p)
                merged_params.append(weighted_param)
            load_weights(self.model, self.names, merged_params)
        features = self.model(x)
        return self.get_classification_head(dataset_name)(features)


def main():
    exam_datasets = ['RESISC45', 'GTSRB', 'EuroSAT', 'Cars', 'SUN397', 'DTD', 'SVHN', 'MNIST']
    model_name = 'ViT-B-32'
    args = parse_arguments()
    args.data_location = '/home/brcao/Repos/merge_model/Datasets/mm/ModelMergingBaseline16Datasets/'
    args.model = model_name
    args.save = f'checkpoints-Layer-wise-8models/{model_name}'
    args.logs_path = f'logs-plot-8model/{model_name}'
    args.device = device

    log = create_log_dir(args.logs_path, f'log_{time.strftime("%Y%m%d_%H%M%S")}_StatMLP.txt')
    args.log = log

    pretrained_path = f'/home/brcao/Repos/merge_model/Datasets/models/task_vectors_checkpoints/{model_name}/zeroshot.pt'
    pretrained_model = torch.load(pretrained_path)
    model = ModelWrapper(pretrained_model).to(args.device)
    orig_params, names = make_functional(model)

    task_vectors = [TaskVector(
        pretrained_path,
        f'/home/brcao/Repos/merge_model/Datasets/models/task_vectors_checkpoints/{model_name}/{d}/finetuned.pt'
    ) for d in exam_datasets]

    paramslist = [tuple(p.detach().to(args.device).requires_grad_() for p in orig_params)]
    paramslist += [tuple(p.detach().to(args.device).requires_grad_() for p in tv.vector.values()) for tv in task_vectors]
    torch.cuda.empty_cache()

    StatsMerging = StatsMerging(paramslist, model, names, exam_datasets, args).to(args.device)
    optimizer = torch.optim.AdamW(StatsMerging.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # log.info("Initial evaluation:")
    # encoder = StatsMerging.get_image_encoder()
    # log.info(f"Initial alphas:\n{StatsMerging.get_alphas()[0].detach().cpu()}")
    # for d in exam_datasets:
    #     classifier = StatsMerging.get_classification_head(d)
    #     metrics = eval_single_dataset_preprocess_head(encoder, classifier, d, args)
    #     log.info(f"{d} ACC: {metrics['top1']:.2f}")

    best_avg_acc = -1
    num_batches = 1

    for epoch in range(500):
        start = time.time()
        StatsMerging.train()
        total_loss = 0.0
        processed_batches = 0
        accumulated_loss = torch.tensor(0.0, device=args.device)

        for d in exam_datasets:
            dataset = get_dataset(d, pretrained_model.val_preprocess, location=args.data_location, batch_size=4)
            dataloader = get_dataloader_shuffle(dataset)
            for batch_i, data in enumerate(dataloader):
                if batch_i >= num_batches:
                    break
                data = maybe_dictionarize(data)
                x, y = data['images'].to(args.device), data['labels'].to(args.device)
                out = StatsMerging(x, d, training=True)
                loss = F.cross_entropy(out, y)
                accumulated_loss += loss
                total_loss += loss.item()
                processed_batches += 1

        optimizer.zero_grad()
        accumulated_loss.backward()
        optimizer.step()
        scheduler.step()

        avg_loss = total_loss / processed_batches
        log.info(f"Epoch {epoch} Loss: {avg_loss:.4f} LR: {scheduler.get_last_lr()[0]:.2e} Time: {time.time()-start:.1f}s")

        if (epoch + 1) % 50 == 0:
            StatsMerging.eval()
            encoder = StatsMerging.get_image_encoder()
            alphas = StatsMerging.get_alphas()[0].detach().cpu()
            log.info(f"Alphas at epoch {epoch + 1}:\nZeroshot weights: {alphas[:, 0].mean():.4f}\nAdditional tasks weights:\n{alphas[:, 1:]}")

            total_acc = 0.0
            with torch.no_grad():
                for d in exam_datasets:
                    classifier = StatsMerging.get_classification_head(d)
                    metrics = eval_single_dataset_preprocess_head(encoder, classifier, d, args)
                    total_acc += metrics['top1']
                    log.info(f"{d}: {metrics['top1']:.5f}")

            avg_acc = total_acc / len(exam_datasets)
            log.info(f"Avg ACC: {avg_acc:.5f}\n")

            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                torch.save(StatsMerging.state_dict(), args.save + "/best_model.pt")
                log.info(f"New best model saved with avg acc {best_avg_acc:.4f}")

if __name__ == "__main__":
    main()
