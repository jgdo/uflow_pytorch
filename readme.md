# UFlow implementation in PyTorch

This is an unofficial implementation of https://github.com/google-research/google-research/tree/master/uflow

Currently in very early stage, only basic loss functions implemented.

For training with minecraft data, the ep1_pickle_doc.pkl must be placed into dataset/UFlow_data folder (see dataset.py)

Training: 
```
python3 train.py
```

Torch checkpoint will be stored at save/checkpoint.pt

Testing:
```
python3 show_results.py
```

Use --help for flags or check out uflow_flags.py
