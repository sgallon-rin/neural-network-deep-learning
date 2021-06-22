# Project 3 - Scene Text Recognition

The Scene Text Recognition (SCR) task can be divided into two sub-tasks:
first 

- detect/segment the area where text may appear 

and then

- recognize the text information in the detected/segmented area.

The training set has 7932 images, and the test set has 1990 images.

## EAST

EAST is used for detection task.

Codes refer to [SakuraRiven/EAST](https://github.com/SakuraRiven/EAST).

### Training

Run `$ python ./code/train_EAST.py`.

Trained for 600 epochs, took roughly 10 days on a 2080Ti GPU.

```
[2021-06-07 14:34:30,512] [INFO] Epoch is [1/600], mini-batch is [1/991], time consumption is 0.43097448, batch_loss is 9.25167656
[2021-06-07 14:34:31,709] [INFO] Epoch is [1/600], mini-batch is [2/991], time consumption is 0.12628007, batch_loss is 7.04520941
...
...
...
[2021-06-18 00:11:07,076] [INFO] epoch_loss is 0.45232999, epoch_time is 1529.87673354
[2021-06-18 00:11:07,143] [INFO] Epoch model saved to /home/jupyter/sjl/neural-network-deep-learning/project3/pths/EAST/model_epoch_600.pth
```

### Detection

Detect the text position in the test set and save in `./data/EAST_gt/`.

Run `$ python ./code/eval_EAST.py`. 

## CRNN

CRNN is used for recognition task.

Codes refer to [Sierkinhane/CRNN_Chinese_Characters_Rec](https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec).

### Extract All Characters in Training set

run `$ python ./code/extract_chars.py` to extract all characters in the training set.
The result is saved as `./code/vocab.txt` in utf-8 encoding.

### Make OCR Dataset

Prepare the dataset for recognition task. 
Cut the training picture according to labeled text positions, 
and save each picture as several labeled mini-pictures in `./data/train/text_img/`. 
(72838 mini-pictures in total.)
The labels are saved in a single txt as `./data/train/txt/train_own.txt`

### Training

Run `$ python ./code/train_CRNN.py`

Trained for 100 epochs (due to time limitation), took one day.

```
...
[2021-06-22 02:57:04,473] [INFO] Start training epoch [99/100]
...
[#correct:39175 / #total:72834]
Test loss: 1.0573, accuray: 0.5379
[2021-06-22 03:07:10,902] [INFO] Epoch 100, train loss: 0.5379
[2021-06-22 03:07:10,949] [INFO] Epoch state dict saved to /home/jupyter/sjl/neural-network-deep-learning/project3/pths/CRNN/OWN/crnn/2021-06-21-10-09/checkpoints/checkpoint_99_acc_0.5379.pth
```

### Generate Result to Submit

Run `$ python .py`