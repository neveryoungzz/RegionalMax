# RegionalMax
Recurrent neural network learning of regional max
Method:
Here we embed both the sequence data and the scalar data to a default 8 dimension vector, and dump the embeded sequence data to a LSTM model with 30 dimension hidden states: c and h. The initial states of c is set to all zeros, while the initial states of h is set to the embeded scalar (the rest vectors are set zeros). The final state of the c is dumpped into a 1-layer MLP with 10 outputs, and the outputs of the MLP are used as the model's prediction. The multi-class cross entropy  is used as the loss function. Adam optimizer with the 0.01 learning rate is used to optimize the model. In the test dataset, we noticed that the prediction accuracy is above 0.97 with 50 training epochs. 
