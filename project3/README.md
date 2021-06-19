# Project 3 - Scene Text Recognition

The Scene Text Recognition (SCR) task can be divided into two sub-tasks:
first 

- detect/segment the area where text may appear 

and then

- recognize the text information in the detected/segmented area.

## EAST

EAST is used for detection task.

### Training

Trained for 600 epochs, took roughly 10 days on a 2080Ti GPU.

Code in `train_EAST.py`.

```
[2021-06-07 14:34:30,512] [INFO] Epoch is [1/600], mini-batch is [1/991], time consumption is 0.43097448, batch_loss is 9.25167656
[2021-06-07 14:34:31,709] [INFO] Epoch is [1/600], mini-batch is [2/991], time consumption is 0.12628007, batch_loss is 7.04520941
...
...
...
[2021-06-18 00:11:07,076] [INFO] epoch_loss is 0.45232999, epoch_time is 1529.87673354
[2021-06-18 00:11:07,143] [INFO] Epoch model saved to /home/jupyter/sjl/neural-network-deep-learning/project3/pths/EAST/model_epoch_600.pth
```

## CRNN

CRNN is used for recognition task.

### Extract All Characters

run `$ python ./code/extract_chars.py` to extract all characters in the training set.
The result is saved as `./code/vocab.txt` in utf-8 encoding.