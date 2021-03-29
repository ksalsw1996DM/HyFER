# HyFER : Hypergraph neural network Framework for Efficient implementation and Reproduciblity

This drive shows the code / implementation for our framework HyFER:

User can test the benchmarking performance in hyperGNN for node classification task like this:

Specify dataset name, set latent variable dimension, number of layers, and experiment numbers :

`python node_classification.py --dataset_name dblp --model hnhn --dim_vertex 100 --dim_edge 100 --dim_query 64 --exp_num 10 --gpu 3`

User can also test the benchmarking performance in hyperGNN for hyperedge prediction task like this:

Users should additionally specify aggregation methods, negative sampling methods, and whether to use given negative sample or generate new one:

`python edge_prediction.py --dataset_name dblp --model hnhn --dim_vertex 100 --dim_edge 100 --dim_query 64 --aggr mean --nsampler MNS --gpu 3`

For all hyperparameter and model selection, please observe utils.py or run:

`python node_classification.py --h`