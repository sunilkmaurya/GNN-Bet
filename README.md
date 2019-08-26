# Code repository for GNN-Bet
This repository contains implementation for paper "Fast Approximations of Betweenness Centrality using Graph Neural Networks" which is accepted as short research paper in CIKM 2019.

Code is written in Python and the proposed model is implemented using Pytorch.

**Main package Requirements**: torch, networkit, networkx and scipy.
Use of conda environment is recommended but not necessary.

For easy check, we have provided small synthetic dataset. The given dataset consists of 30 scalefree graphs with number of nodes varying from 1000 to 10,000. Graphs are stored as networkx graphs and betweenness centrality values are stored in dictionaries. 20 graphs are used for training and 10 graphs for testing. 20 adjacency matrices are created from each training graph by randomly permutating the node sequences. Hence we have total 400 samples for training.

**Running the code**:

    python main.py

outputs the average Kendall's Tau score of test graphs.