# GraphGlue: Multi-Domain Transferable Graph Gluing for Building Graph Foundation Models


## Get Started


To run the pretraining, please using the following command:
```shell
python main.py --run_type pretrain \
 --pretrain_single_graph_data ${SINGLE_GRAPH_DATASETS_LIST} \
 --pretrain_multi_graph_data ${MULTI_GRAPH_DATASETS_LIST}
```
You need to replace ```${SINGLE_GRAPH_DATASETS_LIST}``` 
and ```${MULTI_GRAPH_DATASETS_LIST}```
with lists of graph dataset names. 
For instance, ```[ogbn-arxiv, Reddit, FB15k_237]```  and ```[PROTEINS, HIV]```.