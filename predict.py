import torch
import numpy as np
from sklearn import metrics
from tqdm import tqdm
from model import Model
from utils import load_data,calculate

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

def evaluate(model,val_features, val_graphs, val_labels):
    
    model.eval()
    preds = torch.zeros(len(val_labels),10).cuda()
    confusion = torch.zeros(10,10).cuda()
    subcellular = ["Nucleus", "Cytoplasm", "Secreted", "Mitochondrion", "Membrane", "Endoplasmic", "Plastid", "Golgi_apparatus", "Lysosome", "Peroxisome"]
    f1 = open(Result_Path + "sub_cellular_prediction.txt", "w")

    auc = []
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

            a = y_true.item()

            y_pred = model(features, graphs)

            for j in range(10):
                preds[i][j] = y_pred[0][j]
            b = torch.max(y_pred, 1)[1].item()
            f1.write("prediction: " + str(b) + " " + subcellular[b] + "\n")
            confusion[a][b] += 1
    f1.close()
    q = preds.cpu()
    for i in range(10):
        labels = []
        f3 = open("./data/deeploc_test_label", "r")
        for j in range(len(val_labels)):
            label = f3.readline().strip()
            if(int(label[0])==i):
                labels.append(1)
            else:
                labels.append(0)
        rocauc1 = metrics.roc_auc_score(labels, q[:,i])
        auc.append(rocauc1)
        f3.close()
    acc, table = calculate(confusion)
    print("acc:", round(acc,4))
    print("Subcellular".ljust(17, ' ') + "MCC    " + "Precision  " + "Recall  " + "accuracy  " + "AUC")
    for i in range(10):
        print(subcellular[i].ljust(17, ' ') + str(round(table[i][0],3)).ljust(7, ' ') + str(round(table[i][1],3)).ljust(11, ' ') +  str(round(table[i][2],3)).ljust(8, ' ') + str(round(table[i][3],3)).ljust(10, ' ') + str(round(auc[i],3)))


def main():
    model = Model()
    model.load_state_dict(torch.load('./model/best_model.pkl'))
    if torch.cuda.is_available():
        model.cuda()
    val_features, val_graphs, val_labels = load_data(Dataset_Path + "deeploc_test_seq", Dataset_Path + "deeploc_test_label", Dataset_Path + "pssm/", Dataset_Path + "graph/")
    evaluate(model,val_features, val_graphs, val_labels)

if __name__ == "__main__":
    main()