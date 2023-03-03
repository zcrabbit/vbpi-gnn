# vbpi-gnn
Code for [learnable topological features for phylogenetic inference via graph neural networks](https://openreview.net/forum?id=hVVUY7p64WL)

Please cite our paper if you find the code useful
```
@inproceedings{
zhang2023learnable,
title={Learnable Topological Features For Phylogenetic Inference via Graph Neural Networks},
author={Cheng Zhang},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=hVVUY7p64WL}
}
```


## Mini demo

Use command line
```python
python run_tde.py --gnn_type edge --batch_size 20 --hL 2 --hdim 100 --maxIter 200000
python main.py --dataset DS1  --brlen_model gnn --gnn_type edge --hL 2 --hdim 100 --maxIter 400000 --empFreq --psp

```
You can also load the data, set up and train the model on your own. See more details in [main.py](https://github.com/zcrabbit/vbpi-gnn/blob/main/main.py) and [run_tde.py](https://github.com/zcrabbit/vbpi-gnn/blob/main/run_tde.py).
