import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.models as models 
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 데이터 전처리 설정 (증강 없이)
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet에 최적화된 입력 크기
    transforms.ToTensor(),
])

# 데이터셋 경로 설정
train_image_dir = '/home/GJ/Workspace/WILDFIRE/DATA/tiled_224_golden/yolo_train/train/'
val_image_dir = '/home/GJ/Workspace/WILDFIRE/DATA/tiled_224_golden/yolo_train/val/'

# ImageFolder를 사용하여 데이터셋 로드 (클래스 폴더 구조를 자동으로 인식)
train_dataset = ImageFolder(root=train_image_dir, transform=transform)
val_dataset = ImageFolder(root=val_image_dir, transform=transform)

# 데이터 로더 생성
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# ResNet34 모델 로드 및 최종 레이어 수정 (이진 분류)
resnet34 = models.resnet34(pretrained=True)
num_features = resnet34.fc.in_features
resnet34.fc = nn.Linear(num_features, 2)  # 이진 분류를 위해 출력 노드를 2개로 설정

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
resnet34.to(device)

# 손실 함수 및 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet34.parameters(), lr=0.0001)

# 훈련 및 검증 함수 정의
def train(model, criterion, optimizer, data_loader, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(data_loader.dataset)
    return epoch_loss

def validate(model, criterion, data_loader, device):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    epoch_loss = running_loss / len(data_loader.dataset)
    accuracy = correct_predictions.double() / len(data_loader.dataset)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    return epoch_loss, accuracy, conf_matrix

# Early Stopping 설정
patience = 50
early_stopping_counter = 0

# 손실 및 정확도 추적
train_losses = []
val_losses = []
val_accuracies = []

# 모델 훈련 및 검증 루프
num_epochs = 1000
best_accuracy = 0.0
best_epoch = 0

for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    train_loss = train(resnet34, criterion, optimizer, train_loader, device)
    val_loss, val_accuracy, conf_matrix = validate(resnet34, criterion, val_loader, device)
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f'Epoch {epoch}/{num_epochs - 1}, '
          f'Train Loss: {train_loss:.4f}, '
          f'Validation Loss: {val_loss:.4f}, '
          f'Validation Accuracy: {val_accuracy:.4f}')
    
    # 혼동 행렬 출력
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=[0, 1])
    disp.plot()
    plt.show()
    
    # 가장 좋은 성능을 보인 모델 저장
    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        best_epoch = epoch
        torch.save(resnet34.state_dict(), f'best_resnet34_224_model_epoch_{epoch}.pth')
        early_stopping_counter = 0  # Early stopping counter 초기화
    else:
        early_stopping_counter += 1
    
    # Early stopping 조건 확인
    if early_stopping_counter >= patience:
        print("Early stopping triggered.")
        break

    # 손실 및 정확도 그래프 저장
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')

    # 정확도를 같은 그래프에 추가, 오른쪽 y축 사용
    ax2 = plt.gca().twinx()
    ax2.plot([acc.cpu().numpy() for acc in val_accuracies], label='Validation Accuracy', color='green')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc='upper right')

    plt.title('Training & Validation Loss and Accuracy vs. Epoch')
    plt.savefig('loss_accuracy_vs_epoch_224.png')

print(f'Best model saved from epoch {best_epoch} with validation accuracy of {best_accuracy:.4f}.')
