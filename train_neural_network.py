import datetime
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
from torch import optim
from torch.utils.tensorboard import SummaryWriter

class ChessValueDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
       return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

#def load_data(file_path="processed_complex/dataset_1M_layer_76.npz", use_memmap=False):
def load_data(file_path="processed_medium/dataset_1M_layer_22.npz", use_memmap=False):
#def load_data(file_path="processed_simple/dataset_1M_layer_7.npz", use_memmap=False):
    if use_memmap:
        print("USING MMAP_MODE=R")
        dat = np.load(file_path, mmap_mode='r')  # Use memory-mapped file to avoid loading entire dataset into RAM
    else:
        dat = np.load(file_path)

    print("TRAINING ON DATASET: ", file_path)

    X = dat['X']
    Y = dat['Y']

    print(Y)
    return X, Y

def create_datasets(split_ratio=0.8):
    X, Y = load_data()
    total_count = len(X)
    train_count = int(total_count * split_ratio)
    val_count = total_count - train_count
    return random_split(ChessValueDataset(X, Y), [train_count, val_count])

class ComplexNeuralNetwork(nn.Module):
    def __init__(self):
        super(ComplexNeuralNetwork, self).__init__()
        self.a1 = nn.Conv2d(76, 100, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(100)
        self.a2 = nn.Conv2d(100, 100, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(100)
        self.a3 = nn.Conv2d(100, 200, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(200)
        self.dropout3 = nn.Dropout(0.2)

        self.b1 = nn.Conv2d(200, 200, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(200)
        self.b2 = nn.Conv2d(200, 200, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(200)
        self.b3 = nn.Conv2d(200, 400, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(400)
        self.dropout6 = nn.Dropout(0.2)

        self.c1 = nn.Conv2d(400, 400, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(400)
        self.c2 = nn.Conv2d(400, 400, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(400)
        self.c3 = nn.Conv2d(400, 800, kernel_size=3, stride=2, padding=1)
        self.bn9 = nn.BatchNorm2d(800)
        self.dropout9 = nn.Dropout(0.2)

        self.last = nn.Linear(800, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.a1(x)))
        x = F.relu(self.bn2(self.a2(x)))
        x = F.relu(self.bn3(self.a3(x)))
        x = self.dropout3(x)

        x = F.relu(self.bn4(self.b1(x)))
        x = F.relu(self.bn5(self.b2(x)))
        x = F.relu(self.bn6(self.b3(x)))
        x = self.dropout6(x)

        x = F.relu(self.bn7(self.c1(x)))
        x = F.relu(self.bn8(self.c2(x)))
        x = F.relu(self.bn9(self.c3(x)))
        x = self.dropout9(x)

        x = x.view(-1, 800)
        x = self.last(x)

        return F.tanh(x)

class MediumNeuralNetwork(nn.Module):
    def __init__(self):
        super(MediumNeuralNetwork, self).__init__()
        self.a1 = nn.Conv2d(22, 50, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(50)
        self.a2 = nn.Conv2d(50, 50, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(50)
        self.a3 = nn.Conv2d(50, 100, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(100)
        self.dropout3 = nn.Dropout(0.2)

        self.b1 = nn.Conv2d(100, 100, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(100)
        self.b2 = nn.Conv2d(100, 100, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(100)
        self.b3 = nn.Conv2d(100, 200, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(200)
        self.dropout6 = nn.Dropout(0.2)

        self.c1 = nn.Conv2d(200, 200, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(200)
        self.c2 = nn.Conv2d(200, 200, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(200)
        self.c3 = nn.Conv2d(200, 400, kernel_size=3, stride=2, padding=1)
        self.bn9 = nn.BatchNorm2d(400)
        self.dropout9 = nn.Dropout(0.2)

        self.last = nn.Linear(400, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.a1(x)))
        x = F.relu(self.bn2(self.a2(x)))
        x = F.relu(self.bn3(self.a3(x)))
        x = self.dropout3(x)

        x = F.relu(self.bn4(self.b1(x)))
        x = F.relu(self.bn5(self.b2(x)))
        x = F.relu(self.bn6(self.b3(x)))
        x = self.dropout6(x)

        x = F.relu(self.bn7(self.c1(x)))
        x = F.relu(self.bn8(self.c2(x)))
        x = F.relu(self.bn9(self.c3(x)))
        x = self.dropout9(x)

        x = x.view(-1, 400)
        x = self.last(x)

        return F.tanh(x)

class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super(SimpleNeuralNetwork, self).__init__()
        self.a1 = nn.Conv2d(7, 25, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(25)
        self.a2 = nn.Conv2d(25, 25, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(25)
        self.a3 = nn.Conv2d(25, 50, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(50)
        self.dropout3 = nn.Dropout(0.2)

        self.b1 = nn.Conv2d(50, 50, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(50)
        self.b2 = nn.Conv2d(50, 50, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(50)
        self.b3 = nn.Conv2d(50, 100, kernel_size=3, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(100)
        self.dropout6 = nn.Dropout(0.2)

        self.c1 = nn.Conv2d(100, 100, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(100)
        self.c2 = nn.Conv2d(100, 100, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(100)
        self.c3 = nn.Conv2d(100, 200, kernel_size=3, stride=2, padding=1)
        self.bn9 = nn.BatchNorm2d(200)
        self.dropout9 = nn.Dropout(0.2)

        self.last = nn.Linear(200, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.a1(x)))
        x = F.relu(self.bn2(self.a2(x)))
        x = F.relu(self.bn3(self.a3(x)))
        x = self.dropout3(x)

        x = F.relu(self.bn4(self.b1(x)))
        x = F.relu(self.bn5(self.b2(x)))
        x = F.relu(self.bn6(self.b3(x)))
        x = self.dropout6(x)

        x = F.relu(self.bn7(self.c1(x)))
        x = F.relu(self.bn8(self.c2(x)))
        x = F.relu(self.bn9(self.c3(x)))
        x = self.dropout9(x)

        x = x.view(-1, 200)
        x = self.last(x)

        return F.tanh(x)



if __name__ == "__main__":
  #  model_directory = 'models_complex'  # Define directory to save models_complex
    model_directory = 'models_medium'  # Define directory to save models_complex
  #  model_directory = 'models_simple'  # Define directory to save models_complex

    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_directory = f'./{model_directory}/{current_time}'
    os.makedirs(session_directory, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE USED: ", device)

    train_dataset, val_dataset = create_datasets(split_ratio=0.8)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

  #  model = ComplexNeuralNetwork().to(device)
    model = MediumNeuralNetwork().to(device)
  #  model = SimpleNeuralNetwork().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    floss = nn.MSELoss()
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    writer = SummaryWriter()

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    epochs = 100
    print("NUMBER OF EPOCHS: ", epochs)

    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        train_loss = 0
        for data, target in train_loader:
            target = target.unsqueeze(-1)
            data, target = data.to(device), target.to(device)
            data = data.float()
            target = target.float()

            optimizer.zero_grad()
            output = model(data)
            loss = floss(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        writer.add_scalar('Loss/Train', train_loss, epoch)  # Log training loss to TensorBoard

        val_loss = 0
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                target = target.unsqueeze(-1)
                data, target = data.to(device), target.to(device)
                data = data.float()
                target = target.float()

                output = model(data)
                loss = floss(output, target)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        writer.add_scalar('Loss/Validation', val_loss, epoch)  # Log validation loss to TensorBoard
        scheduler.step(val_loss)  # Adjust learning rate based on validation loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            model_path = os.path.join(session_directory, f'epoch_{epoch}_val_loss_{val_loss:.4f}_train_loss_{train_loss:.4f}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Model saved")
        else:
            patience_counter += 1
            if patience_counter > patience:
                print("EARLY STOPPING DUE TO PATIENCE COUNTER")
                break

        epoch_time = time.time() - start_time
        print(f'Epoch {epoch}: Training Loss {train_loss:.6f}, Validation Loss {val_loss:.6f}, Time: {epoch_time:.2f} sec')
        print("PATIENCE COUNTER: ", patience_counter)
        writer.add_scalars('Loss/Train-Validation', {'Train': train_loss, 'Validation': val_loss}, epoch)

    writer.close()