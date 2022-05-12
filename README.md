# GraphLoc
A Protein Subcellular Localization Predictor developed by Graph Convolutional Network and Predicted Contact Map
The source code for our paper Protein Subcellular Localization Prediction Model Based on Graph Convolutional Network
1. Dependencies
The code has been tested under Python 3.6，related packages are in env.yaml and pip.txt
2. How to retrain the GraphLoc model and test
Step 1: Download all sequence features
Please go to the path ./data/pssm/pssm file link.txt and ./data/graph/graph file link.txt and download pssm files.zip and graph files.zip
Step 2: Decompress all .zip files
Please unzip 2 zip files and put them into the corresponding paths.
./data/graph/graph files.zip -> ./data/graph/
./data/pssm/pssm files.zip -> ./data/pssm/
Step 3: Run the training code
Run the following python script and it will take several hours to train the model.

$ python train.py
A trained model will be saved in the folder ./model

Step 4: Run the test code
Run the following python script and it will be finished in a few seconds.

$ python predict.py

The results will be saved in the folder ./result
