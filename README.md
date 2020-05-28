# Code repository for GNN-Bet
This repository contains implementation for paper "Fast Approximations of Betweenness Centrality using Graph Neural Networks" which is accepted as short research paper in CIKM 2019.

Code is written in Python and the proposed model is implemented using Pytorch.

**Main package Requirements**: Pytorch, networkit, networkx and scipy.
Use of conda environment is recommended but not necessary.
Experiments in paper used PyTorch(0.4.1) and Python(3.7).

For easy check, we have provided small synthetic dataset. The given dataset consists of 30 scalefree graphs with number of nodes varying from 1000 to 10,000. Graphs are stored as networkx graphs and betweenness centrality values are stored in dictionaries. 20 graphs are used for training and 10 graphs for testing. 20 adjacency matrices are created from each training graph by randomly permutating the node sequences. Hence we have total 400 samples for training.

**Running the code**:

    python main.py

outputs the average Kendall's Tau score of test graphs.

**Addition-1**
I have added a Jupyter notebook to show some interesting observations. We simplify the model to single layer and train it to rank nodes based on degree centrality. We see that the trained model can easily rank the nodes with respect to degree centrality in new different types of graphs without being provided any explicit information. These observations are not discussed in the paper.

**Note (PyTorch 1.0 or higher)**:  
This code was written and tested on PyTorch (0.4.1), so it has some incompatibilities with newer versions. With PyTorch versions (1.0 or higher), this code may give inconsistence performance. This is because of some of the changes in newer versions cause problems with this code. One reason is dropout not acting as intended in the code (See [https://discuss.pytorch.org/t/moving-from-pytorch-0-4-1-to-1-0-changes-model-output/41760/3](https://discuss.pytorch.org/t/moving-from-pytorch-0-4-1-to-1-0-changes-model-output/41760/3)).
For example, changing the dropout code in `model.py`,
```
score = F.dropout(score_temp,self.dropout)
```
to (for newer PyTorch versions) by adding `self.training`
```
score = F.dropout(score_temp,self.dropout,self.training)
```
improves the performance similar to original results. In addition to this fix, I found results varying a bit (between older and newer versions) even with same random seed. I will look into it and provide a patch for newer PyTorch versions.



