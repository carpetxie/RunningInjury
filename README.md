# PyTorch Running Injury Classification

This repository contains the code and resources for a research project conducted under the guidance of Dr. Xia Ning at The Ohio State University in 2023. The project explores the use of deep learning models, specifically Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks, to predict running injuries based on ground reaction force (GRF) data. The study aims to enhance the accuracy of injury prediction by leveraging time-series data from running biomechanics.

The dataset used in this study includes the x-coordinate vector of GRFs collected during foot strikes, represented as sequences of 9,000 data points. These sequences were labeled with binary indicators to mark the occurrence or absence of injuries. Two models, an RNN and an LSTM, were developed using the PyTorch framework. The RNN model included a single recurrent layer, while the LSTM model incorporated additional fully connected layers and dropout to mitigate overfitting. Both models underwent hyperparameter optimization using the Optuna library.

The results demonstrated that the RNN model outperformed the LSTM in terms of accuracy, precision, and recall, suggesting that simpler architectures may be more effective for this type of injury prediction task. While the study shows promise, it also highlights the need for more comprehensive approaches that consider multiple risk factors to improve predictive capabilities in real-world applications.

**The dataset used was from: **__

Fukuchi RK, Fukuchi CA, Duarte M. A public dataset of running biomechanics and the effects of running speed 
on lower extremity kinematics and kinetics. PeerJ. 2017;5:e3298. 
