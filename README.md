# CDRL

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org)).
- `pip install -r requirements.txt`
- Download the MEAD dataset from ([here](https://wywu.github.io/projects/MEAD/MEAD.html)).
- Download the pre-trained weights ([here](https://drive.google.com/file/d/1W_qa9xxXTCXo_44PX_oRDLlJQ3F8uXJk/view?usp=sharing)) (" backbone.pth ") and place it under "./pretrain/backbone.pth"


## Preprocessing
The obtained MEAD dataset is first preprocessed with 'dataloader/align_face.py':

```bash
python align_face.py
```
Get audio feature corresponding to image:

```bash
python w2f.py
```


## Training CCRL
```bash
cd CCRL 
python train.py
```


## Training CDRL
```bash
cd CDRL 
python train.py
```

## The integration of CDRL into NED (Please switch to the NED_CDRL branch.)
First, follow the data preparation and processing workflow of NED.  
### Supervising the training of the first-stage emotion manipulator
```bash
bash run_manipulator.sh
```

### Supervising the training of the second-stage renderer.
```bash
bash run_render.sh
```

### Testing metrics on the test set.
```bash
cd metrics
bash eval_crossID_driven.sh
```
