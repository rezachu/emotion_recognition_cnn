# Speech Emotion Recognition with Convolution Neural Network
#### *Author @rezachu*

## I. Introduction
- This is a CNN Speech Emotion Recognition Model I found on [GitHub](https://github.com/MITESHPUTHRANNEU/Speech-Emotion-Analyzer). 
- There is no paper to reference according to the repository author. 

## II. Package Required

- Please refers to the requirement file: [requirements.txt](./requirements.txt)

## III. To Run
- Please run the notebook named: **CNN_emotion_recognition.ipynb**
- Please create a `./data/` folder and put all of the data inside.
- Please create a `./model/` folder and set it as the model weight saving directory.

## IV. Preparation: Understanding the Data from Repo

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


## V.Preparation: Understanding the Model

**Model Architecture:**

```
# Model 
model = Sequential()
model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1],1))) #1
model.add(Activation('relu'))
model.add(Conv1D(256, 8, padding='same')) #2
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 8, padding='same')) #3
model.add(Activation('relu')) 
model.add(Conv1D(128, 8, padding='same')) #4
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same')) #5
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same')) #6
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(64, 8, padding='same')) #7
model.add(Activation('relu'))
model.add(Conv1D(64, 8, padding='same')) #8
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(target_class)) #9
model.add(Activation('softmax'))
opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
```

## VI. Project Summary
- Please refers to my Blog on Medium: [Speech Emotion Recognition with Convolution Neural Network](https://medium.com/@rezachu/speech-emotion-recognition-with-convolution-neural-network-1e6bb7130ce3)

