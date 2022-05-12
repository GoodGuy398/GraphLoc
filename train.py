import os
import torch
import numpy as np
from tqdm import tqdm
from utils import load_data
from model import Model

# path
Dataset_Path = './data/'
Result_Path = './result/'

# Seed
SEED = 2333
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.set_device(3)
    torch.cuda.manual_seed(SEED)

# Model parameters
NUMBER_EPOCHS = 1000
BATCH_SIZE = 256


def evaluate(model,val_features, val_graphs, val_labels):
    
    model.eval()
    epoch_loss_valid = 0.0
    exact_match = 0
    for i in tqdm(range(len(val_labels))):
        with torch.no_grad():
            
            sequence_features = torch.from_numpy(val_features[i])
            sequence_graphs = torch.from_numpy(val_graphs[i])
            labels = torch.from_numpy(np.array([int(float(val_labels[i]))]))

            sequence_features = torch.squeeze(sequence_features)
            sequence_graphs = torch.squeeze(sequence_graphs)

            if torch.cuda.is_available():
                features = sequence_features.cuda()
                graphs = sequence_graphs.cuda()
                y_true = labels.cuda()
            
            y_pred = model(features, graphs)
            if(torch.max(y_pred, 1)[1] == y_true):
                exact_match += 1
            loss = model.criterion(y_pred, y_true)
            epoch_loss_valid += loss.item()
    epoch_loss_valid_avg = epoch_loss_valid / len(val_labels)
    acc = exact_match / len(val_labels)
    return acc, epoch_loss_valid_avg


def train(model, epoch):
    train_features, train_graphs, train_labels = load_data(Dataset_Path + "deeploc_train_seq", Dataset_Path + "deeploc_train_label", Dataset_Path + "pssm/", Dataset_Path + "graph/")
    val_features, val_graphs, val_labels = load_data(Dataset_Path + "deeploc_test_seq", Dataset_Path + "deeploc_test_label", Dataset_Path + "pssm/", Dataset_Path + "graph/")
    best_acc = 0
    best_epoch = 0
    cur_epoch = 0
    print("epoch:" + str(0))
    print("========== Evaluate Valid set ==========")
    valid_acc, epoch_loss_valid_avg  = evaluate(model,val_features, val_graphs, val_labels)
    print("valid acc:", valid_acc)
    print("valid loss:", epoch_loss_valid_avg)
    best_acc = valid_acc

    for epoch in range(epoch):
        model.train()
        for i in tqdm(range(len(train_labels))):
                       
            sequence_features = torch.from_numpy(train_features[i])
            sequence_graphs = torch.from_numpy(train_graphs[i])
            labels = torch.from_numpy(np.array([int(float(train_labels[i]))]))
            
            sequence_features = torch.squeeze(sequence_features)
            sequence_graphs = torch.squeeze(sequence_graphs)

            if torch.cuda.is_available():
                features = sequence_features.cuda()
                graphs = sequence_graphs.cuda()
                y_true = labels.cuda()

            y_pred = model(features, graphs)
            loss = model.criterion(y_pred, y_true)
            loss /= BATCH_SIZE
            loss.backward()

            if(i % BATCH_SIZE == 0):
                model.optimizer.step()
                model.optimizer.zero_grad()

        print("epoch:" + str(epoch+1))
        print("========== Evaluate Valid set ==========")
        valid_acc, epoch_loss_valid_avg = evaluate(model,val_features, val_graphs, val_labels)
        print("valid acc:", valid_acc)
        print("valid loss:", epoch_loss_valid_avg)
        if best_acc < valid_acc:
            best_acc = valid_acc
            best_epoch = epoch + 1
            cur_epoch = 0
            torch.save(model.state_dict(), os.path.join('./model/best_model.pkl'))
        else:
            cur_epoch += 1
            if(cur_epoch > 200):
                break
    print("Best epoch at", str(best_epoch))
    print("Best acc at", str(best_acc))
    

def main():
    model = Model()
    model.load_state_dict(torch.load('./model/best_model.pkl'))
    if torch.cuda.is_available():
        model.cuda()
    train(model, NUMBER_EPOCHS)


if __name__ == "__main__":
    main()