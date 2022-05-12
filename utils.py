from Bio import SeqIO
from math import sqrt
import numpy as np

def convertSampleToPhysicsVector_pca(seq):
    """
    Convertd the raw data to physico-chemical property
    PARAMETER
    seq: "MLHRPVVKEGEWVQAGDLLSDCASSIGGEFSIGQ" one fasta seq
        X denoted the unknow amino acid.
    probMatr: Probability Matrix for Samples. Shape (nb_samples, 1, nb_length_of_sequence, nb_AA)
    """
    letterDict = {}
    letterDict["A"] = [0.008, 0.134, -0.475, -0.039, 0.181]
    letterDict["R"] = [0.171, -0.361, 0.107, -0.258, -0.364]
    letterDict["N"] = [0.255, 0.038, 0.117, 0.118, -0.055]
    letterDict["D"] = [0.303, -0.057, -0.014, 0.225, 0.156]
    letterDict["C"] = [-0.132, 0.174, 0.070, 0.565, -0.374]
    letterDict["Q"] = [0.149, -0.184, -0.030, 0.035, -0.112]
    letterDict["E"] = [0.221, -0.280, -0.315, 0.157, 0.303]
    letterDict["G"] = [0.218, 0.562, -0.024, 0.018, 0.106]
    letterDict["H"] = [0.023, -0.177, 0.041, 0.280, -0.021]
    letterDict["I"] = [-0.353, 0.071, -0.088, -0.195, -0.107]
    letterDict["L"] = [-0.267, 0.018, -0.265, -0.274, 0.206]
    letterDict["K"] = [0.243, -0.339, -0.044, -0.325, -0.027]
    letterDict["M"] = [-0.239, -0.141, -0.155, 0.321, 0.077]
    letterDict["F"] = [-0.329, -0.023, 0.072, -0.002, 0.208]
    letterDict["P"] = [0.173, 0.286, 0.407, -0.215, 0.384]
    letterDict["S"] = [0.199, 0.238, -0.015, -0.068, -0.196]
    letterDict["T"] = [0.068, 0.147, -0.015, -0.132, -0.274]
    letterDict["W"] = [-0.296, -0.186, 0.389, 0.083, 0.297]
    letterDict["Y"] = [-0.141, -0.057, 0.425, -0.096, -0.091]
    letterDict["V"] = [-0.274, 0.136, -0.187, -0.196, -0.299]
    letterDict["X"] = [0, -0.00005, 0.00005, 0.0001, -0.0001]
    letterDict["-"] = [0, 0, 0, 0, 0, 1]
    AACategoryLen = 5  
    l = len(seq)
    probMatr = np.zeros((l, AACategoryLen))
    AANo = 0
    for AA in seq:
        if not AA in letterDict:
            probMatr[AANo] = np.full(AACategoryLen, 0)
        else:
            probMatr[AANo] = letterDict[AA]

        AANo += 1
    return probMatr


def convertSampleToBlosum62(seq):
    letterDict = {}
    letterDict["A"] = [4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0]
    letterDict["R"] = [-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3]
    letterDict["N"] = [-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3]
    letterDict["D"] = [-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3]
    letterDict["C"] = [0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1]
    letterDict["Q"] = [-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2]
    letterDict["E"] = [-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2]
    letterDict["G"] = [0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3]
    letterDict["H"] = [-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3]
    letterDict["I"] = [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3]
    letterDict["L"] = [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1]
    letterDict["K"] = [-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2]
    letterDict["M"] = [-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1]
    letterDict["F"] = [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1]
    letterDict["P"] = [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2]
    letterDict["S"] = [1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2]
    letterDict["T"] = [0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0]
    letterDict["W"] = [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3]
    letterDict["Y"] = [-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1]
    letterDict["V"] = [0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4]
    AACategoryLen = 20  
    l = len(seq)
    probMatr = np.zeros((l, AACategoryLen))
    AANo = 0
    for AA in seq:
        if not AA in letterDict:
            probMatr[AANo] = np.full(AACategoryLen, 0)
        else:
            probMatr[AANo] = letterDict[AA]

        AANo += 1
    return probMatr


def readPSSM(pssmfile):
    pssm = []
    with open(pssmfile, 'r') as f:
        count = 0
        for eachline in f:
            count += 1
            if count <= 3:
                continue
            if not len(eachline.strip()):
                break
            line = eachline.split()
            pssm.append(line[2: 22])  
    return np.array(pssm)


def load_data(seq_file, labelfile, pssmdir, graphdir):
    labels = []
    features = []
    graphs = []
    f = open(labelfile, "r")
    num = 0
    print("Load data.")
    for seq_record in list(SeqIO.parse(seq_file, "fasta")):

        pssmfile= pssmdir + str(seq_record.id) +"_pssm.txt"
        pssm = readPSSM(pssmfile)
        pssm = pssm.astype(float)
        Blosum62 = convertSampleToBlosum62(seq_record.seq)
        PhyChem = convertSampleToPhysicsVector_pca(seq_record.seq)
        feature = np.concatenate((PhyChem, Blosum62, pssm), axis=1)
        features.append(feature)
        
        label = f.readline().strip()
        labels.append(label)
        
        graph = np.load(graphdir + seq_record.id + ".npy")
        graphs.append(graph)

        num += 1
        if(num % 500 == 0):
            print("load " + str(num) + " sequences")

    f.close()
    return features, graphs, labels


def calculate(a):
    y_pred = a.cpu().numpy()
    table = np.zeros((y_pred.shape[0],4))
    pt = 0
    num = 0
    for i in range(y_pred.shape[0]):
            TP = y_pred[i, i]
            pt += TP
            FN = np.sum(y_pred[i, :]) - TP
            FP = np.sum(y_pred[:, i]) - TP
            TN = np.sum(y_pred) - TP - FP - FN
            accuracy = round((TP + TN )/ (TP + FP + TN + FN), 3) if TP + FP + TN + FN != 0 else 0.
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            mcc_num = round(((TP * TN) - (FP * FN)) / sqrt((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)), 3) if ((TP + FP)*(TP + FN)*(TN + FP)*(TN + FN)) != 0 else 0.
            table[i][0] = mcc_num
            table[i][1] = Precision
            table[i][2] = Recall
            table[i][3] = accuracy
    num = np.sum(y_pred)
    return pt/num, table