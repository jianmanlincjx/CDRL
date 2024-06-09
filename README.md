# CDRL

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org)).
- `pip install -r requirements.txt`
- Download the MEAD dataset from ([here](https://wywu.github.io/projects/MEAD/MEAD.html)).
- Download the pre-trained weights ([here](https://drive.google.com/file/d/1W_qa9xxXTCXo_44PX_oRDLlJQ3F8uXJk/view?usp=sharing)) (" backbone.pth ") and place it under "./pretrain/backbone.pth"


## Preprocessing
The obtained MEAD dataset is first preprocessed with 'dataloader/align_face.py':

```bash
python ./dataloader/align_face.py
```

## Training CCRL
cd CCRL
python train.py


## Training CDRL
cd CDEL
python train.py
