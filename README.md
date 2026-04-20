# HATR - Hierarchical Audio-Text Representation Model

HATR is a hierarchical-aware multimodal classifier for audio and text embeddings, designed to utilize the label hierarchy during training.

## Quick start

1. Set the input and output paths in `config.yaml`. Make sure the paths point to the correct directories or files before running the model.

2. Run create_full_dataframe.py

3. Run train.py

## Features
- **Multimodal:** Uses both audio and text input (embeddings)  
- **Hierarchical top-class aware training:** Supports hierarchical supervision (through the loss)
- **Attention-based fusion:** Learns to weight modalities dynamically  
- **Residual-based deep classifier:** Stacked residual blocks
- **Embedding augmentation:** Gaussian noise + random masking
- **Visualization**: Outputs additional diagrams for understanding the model's results

## Paper 
```
@inproceedings{anastasopoulou2025hierarchical,
  title = {Hierarchical and Multimodal Learning for Heterogeneous Sound Classification},
  author = {Anastasopoulou, Panagiota and Dal R{\'i}, Francesco Ardan and Serra, Xavier and Font, Frederic},
  booktitle = {Proc. {{Workshop}} on {{Detection}} and {{Classification}} of {{Acoustic Scenes}} and {{Events}} ({{DCASE}})},
  year = {2025}
}
```

## Credits
This code is based on the work of Ardan Dal Ri and was further conceptualized, modified, and refactored by Panagiota Anastasopoulou.
