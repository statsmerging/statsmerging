import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from tqdm import tqdm

from datasets.registry import get_dataset
from datasets.common import get_dataloader, maybe_dictionarize
from args import parse_arguments

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True


def get_loader(name, preprocess, location, batch_size=64, args=None):
    dataset = get_dataset(name, preprocess, location=location, batch_size=batch_size)
    return (
        get_dataloader(dataset, is_train=True, args=args),
        get_dataloader(dataset, is_train=False, args=args),
        len(dataset.classnames)
    )


def get_resnet50(num_classes):
    model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)




def distill(teacher, student, loader, epochs=10, T=4.0, alpha=0.7, lr=1e-3):
    teacher.eval()
    student.train()
    student.to(device)

    opt = torch.optim.Adam(student.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"[Epoch {epoch+1}]"):
            data = maybe_dictionarize(batch)
            x, y = data['images'].to(device), data['labels'].to(device)

            with torch.no_grad():
                teacher_logits = teacher(x)
                
            student_logits = student(x)
            
            # Ensure both logits have the same number of classes
            if teacher_logits.shape[1] != student_logits.shape[1]:
                # If dimensions don't match, we need to project teacher logits to student's dimension
                # This is a simple fix - you might need a more sophisticated approach
                if teacher_logits.shape[1] < student_logits.shape[1]:
                    # Pad teacher logits with zeros
                    diff = student_logits.shape[1] - teacher_logits.shape[1]
                    teacher_logits = F.pad(teacher_logits, (0, diff), "constant", 0)
                else:
                    # Truncate teacher logits
                    teacher_logits = teacher_logits[:, :student_logits.shape[1]]
            
            loss = alpha * ce_loss(student_logits, y) + (1 - alpha) * T**2 * kl_loss(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1)
            )

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(loader):.4f}")
    return student


def distill1(teacher, student, loader, epochs=10, T=4.0, alpha=0.7, lr=1e-3):
    teacher.eval()
    student.train()
    student.to(device)

    opt = torch.optim.Adam(student.parameters(), lr=lr)
    ce_loss = nn.CrossEntropyLoss()
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in tqdm(loader, desc=f"[Epoch {epoch+1}]"):
            data = maybe_dictionarize(batch)
            x, y = data['images'].to(device), data['labels'].to(device)

            with torch.no_grad():
                teacher_logits = teacher(x)

            student_logits = student(x)
   

            loss = alpha * ce_loss(student_logits, y) + (1 - alpha) * T**2 * kl_loss(
                F.log_softmax(student_logits / T, dim=1).to(torch.float32),
                F.softmax(teacher_logits / T, dim=1).to(torch.float32)
            )

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(loader):.4f}")
    return student


def evaluate(model, loader, name="Cars"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            data = maybe_dictionarize(batch)
            x, y = data['images'].to(device), data['labels'].to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"✅ {name} Accuracy: {100 * acc:.2f}%")
    return acc


def main():
    args = parse_arguments()
    args.device = device
    args.data_location = '/home/brcao/Repos/merge_model/Datasets/mm/ModelMergingBaseline16Datasets/'

    teacher_dataset = "EuroSAT"
    vit_teacher_path = f"/home/brcao/Repos/merge_model/Datasets/models/task_vectors_checkpoints/ViT-B-32/{teacher_dataset}/finetuned.pt"
    distilled_save_path = f"/home/brcao/Repos/merge_model/Datasets/models/ResNet50/{teacher_dataset}/distilled.pt"
    os.makedirs(os.path.dirname(distilled_save_path), exist_ok=True)

    print("🔁 Preparing Cars dataloaders...")
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    train_loader, test_loader, num_classes = get_loader(
        name=teacher_dataset,
        preprocess=preprocess,
        location=args.data_location,
        args=args
    )

    print("Loading ViT-B-32 Cars model...")
    vit_teacher = torch.load(vit_teacher_path, map_location=device)
    vit_teacher.eval()

    print("nitializing ResNet-50 student...")
    student = get_resnet50(num_classes)

    print("Starting distillation...")
    distilled_model = distill(vit_teacher, student, train_loader, epochs=10)

    print("Saving distilled ResNet50 model...")
    torch.save(distilled_model.state_dict(), distilled_save_path)
    print(f"Saved to: {distilled_save_path}")

    print("Final evaluation:")
    evaluate(distilled_model, test_loader,teacher_dataset)

"""
cars 83.19

"""

if __name__ == "__main__":
    main()


