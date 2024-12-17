import torch
import torch.nn as nn
import torch.optim as optim
from functions.CustomDataset import CustomDataset
from functions.model import TimeSeriesTransformer
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
import matplotlib.pyplot as plt

HOME_PATH =  r'./dataset'

WINDOW_SIZE = 30

history = {'loss':[], 'accuracy':[], 'val_loss':[], 'val_accuracy':[]}
dataset = CustomDataset(window_size=WINDOW_SIZE, Folder_dir=f'{HOME_PATH}')

val_ratio = 0.2
val_size = int(val_ratio * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)

def train(model_name, train_loader, device, optimizer, loss_func, epoch, num_classes):
    model_name.train()

    Train_total_loss = 0
    Train_correct_predictions = 0

    for (x_train, y_train) in tqdm(train_loader):
        x_train = x_train.to(device)
        y_train = y_train.long().to(device)

        y_train = y_train.view(-1)

        if y_train.min() < 0 or y_train.max() >= num_classes:
            raise ValueError(f"Invalid label values: min={y_train.min()}, max={y_train.max()}")

        y_predict = model_name(x_train)

        loss = loss_func(y_predict, y_train)

        Train_total_loss += loss.item()

        _, indices = torch.max(y_predict, dim=1)

        Train_correct_predictions += (indices == y_train).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_train_loss = Train_total_loss / len(train_loader.dataset)
    Train_accuracy = 100. * Train_correct_predictions / len(train_loader.dataset)

    history['loss'].append(avg_train_loss)
    history['accuracy'].append(Train_accuracy)
    
    print(f'Train Epoch {epoch}: Average loss: {avg_train_loss:.6f}, Accuracy: {Train_accuracy:.2f}%')

def evaluate(model_name, test_loader, device, loss_func, num_classes):
    model_name.eval()

    correct = 0
    val_loss = 0

    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            y_test = y_test.long().to(device)

            y_test = y_test.view(-1)

            if y_test.min() < 0 or y_test.max() >= num_classes:
                raise ValueError(f"Invalid label values: min={y_test.min()}, max={y_test.max()}")

            y_pred = model_name(x_test)

            val_loss += loss_func(y_pred, y_test).item()

            _, indices = torch.max(y_pred, dim=1)

            correct += (indices == y_test).sum().item()

    avg_val_loss = val_loss / len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    history['val_loss'].append(avg_val_loss)
    history['val_accuracy'].append(accuracy)

    print(f'Validation set: Average loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

num_classes = 20

model = TimeSeriesTransformer(224, 64, 8, 1, 20, 30, 1).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

summary(model, input_size=(64, WINDOW_SIZE, 224))

epochs = 100

for epoch in range(1, epochs + 1):
    print(f"Epoch {epoch}/{epochs}")

    train(
        model_name=model,
        train_loader=train_dataloader,
        optimizer=optimizer,
        loss_func=criterion,
        device=device,
        epoch=epoch,
        num_classes=num_classes
    )

    evaluate(
        model_name=model,
        test_loader=val_dataloader,
        loss_func=criterion,
        device=device,
        num_classes=num_classes
    )

def plot_training_history(history):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history['accuracy'], label = 'Train_Accuracy')
    plt.plot(history['val_accuracy'], label = 'Validation_Accuracy')
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel('Accuracy')
    plt.legend(loc = 'best')

    plt.subplot(1,2,2)
    plt.plot(history['loss'], label = 'Train_Loss')
    plt.plot(history['val_loss'], label = 'Validation_Loss')
    plt.title('Model Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc = 'best')

    plt.show()
plot_training_history(history)
torch.save(model, './dataset/model_final.pt')