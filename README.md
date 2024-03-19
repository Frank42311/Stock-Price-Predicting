About 100 different features are generated then reduced using PCA or integration by classifications.
Built GRU models to train and predict, got 97.3% acc on test set.

Details
Algorithms: LSTM + ANN
Input: Open, High, Low, Close, Vol, 8 PCA indicators and 5 chart patterns integration
Input sequential data length: 60 trading days
Hyper parameters: 
	batch_size 120; epochs 200; 
	activation function relu; optimizer adam; 
	loss function mse
  LR: ReduceLROnPlateau, Early stop 30
  Adam, MSE

Model structure 

![image](https://github.com/Frank42311/Stock-Price-Predicting/assets/137829542/7c7271dc-078e-4f23-a2ac-45c05f345a7f)

Train Loss

![image](https://github.com/Frank42311/Stock-Price-Predicting/assets/137829542/e2b1c63f-dbb2-440d-b010-63aea83bba52)

Predicting on never used dataset

![image](https://github.com/Frank42311/Stock-Price-Predicting/assets/137829542/8d60ae03-a144-477a-9e91-693f1afbcd32)
