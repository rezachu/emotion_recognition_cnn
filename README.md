# emotion_recognition_cnn

#### *Author @rezachu*

## I. Introduction
- This is a CNN Speech Emotion Recognition Model I found on [GitHub](https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer). 
- There is no paper to reference according to the repository author. 

- Please run the notebook named: **CNN_emotion_recognition**


## II. Package Required

Please refers to the requirement file.
- Requirement File: [requirements.txt](/uploads/13620b71da276aad1d42a6e7608d3ffe/requirements.txt)

## III. Preparation: Understanding the Data from Repo

Data Set: [The Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS)](https://zenodo.org/record/1188976#.XN0fwnUzZhE)

- 12 Actors & 12 Actresses recorded speech and song version respectively.
- Actor no.18 does not have song version data.
- Emotion `Disgust`, `Neutral` and `Surprised` are not included in the song version data.

Total Class:

| Emotion | Speech Sample Count | Song Sample Count | Summed Count |
| ---- | ---- | ---- | ---- |
| Neutral | 96 | 92 | 188 |
| Calm | 192 | 184 | 376 |
| Happy | 192 | 184 | 376 |
| Sad | 192 | 184 | 376 |
| Angry | 192 | 184 | 376 |
| Fearful | 192 | 184 | 376 |
| Disgust | 192 | 0 | 192 |
| Surprised | 192 | 0 | 192 |
| Total | 1440 | 1012 | 2452 |

### Sample Distribution:

- Originally, there are 16 target classes (8 emotions and each emotion split to male and female.) in total for 1440 samples (Speech Only). The author removed the `disgust`, `surprised` and `neutral` from both gender which reduced the target classes to 10.

- Please Create a `./model/` folder and put all of the data inside.

## IV.Preparation: Understanding the Model

**Default Architecture:**

```
model = Sequential()
model.add(Conv1D(256, 5,padding='same', input_shape=(216,1)))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same'))
model.add(Activation('relu'))
#model.add(Conv1D(128, 5,padding='same'))
#model.add(Activation('relu'))
#model.add(Conv1D(128, 5,padding='same'))
#model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Conv1D(128, 5,padding='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
```

- Its reported performance is `Validation Acc : 72%` on 10 classes prediction. 
- Model result cannot be replicated since the model weight provided cannot be loaded since the author updated the notebook and the model architecture does not match the model weight.
- He did not isolate a test set, so the validation set is identical with his test set. There is a data leakage problem.
- As a result, I have to re-sample the data to `Train, Valid and Test` sets and trained male and female model respectively. 


## V. Key Findings


## VI. Backlog
- [(2 May Thur) Run Speech Emotion Recognition Model](#1)
- [(3 May Fri) Experiment on the Speech Emotion Recognition Model](#5)
- [(6 - 10 May ) Continue the experiment on speech emotion recognition](#7)
- [(6- 10 May) Speech / Text Emotion Recognition PPT](#8)
- [(14 May Tue) Further Training on SER model.](#14)
- [(15 May Wed) Further Experiment on SER](#16)
- [(16 May Thur) SER Data Labeling PPT](#18)
- [(17 May Thur) SER Project Summary](#20)



