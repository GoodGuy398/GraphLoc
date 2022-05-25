# GraphLoc


1. Dependencies

    The code has been tested under Python 3.6，related packages are in env.yaml and pip.txt

2. How to retrain the GraphLoc model and test

    Step 1: Download all sequence features
    
    Please go to the path ./data/pssm/pssm file link.txt and ./data/graph/graph file link.txt and download pssm.tar.gz and graph.tar.gz

    Step 2: Decompress all .tar.gz files
    
    Please decompress 2 tar.gz files and put them into the corresponding paths.
    
        ./data/graph/graph.tar.gz -> ./data/graph/
        ./data/pssm/pssm.tar.gz -> ./data/pssm/

    Step 3: Run the training code
    
    Run the following python script and it will take several hours to train the model.

        $ python train.py

    A trained model will be saved in the folder ./model

    Step 4: Run the test code
    Run the following python script and it will be finished in a few seconds.

        $ python predict.py

    The results will be saved in the folder ./result
