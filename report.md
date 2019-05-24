## I. Baseline Model
- I tried to replicate the model result with the original code provided. 
- Setting:
  - Removed `neutral`, `disgust` and `surprised`.
  - Categorized to Male and Female with 5 emotions respectively.
  - 10 emotions in total.
  - valid set = test set.

- Model Architecture:

```
model = Sequential()
model.add(Conv1D(256, 5,padding='same', input_shape=(216,1)))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same'))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
opt = keras.optimizers.rmsprop(lr=0.00001, decay=1e-6)
```
- **Data Splitting**
  - Train Set Shape: (1504, 216, 1)
  - Valid Set Shape: (376, 216, 1)
  - Test Set Shape: (320, 216, 1)
  - Noted that the Testing Set are included in the `Train / Valid` sets in the default setting.
  - **There is a data leakage problem.**

| Performance | Train Valid Loss | Confusion Matrix |
| ---- | ---- | ---- |
| Validation Acc: 50.27% <br><br> Testing Accuracy: 61.25% <br><br> Testing F1 Score: 60.04% | ![image](/uploads/a580388224e55cd94da4f61428418a42/image.png) | ![image](/uploads/a304aa334aab9dd47efa7e4e3b72649e/image.png) |

#### Baseline with new splitting
- Actor no. 1 - 20 are used for `Train / Valid` sets with 8:2 splitting ratio. 
- Actor no. 21 - 24 are isolated for testing usage.
- **Data Splitting**
  - Train Set Shape: (1248, 216, 1)
  - Valid Set Shape: (312, 216, 1)
  - Test Set Shape: (320, 216, 1) - (Isolated)

| Performance | Train Valid Loss | Confusion Matrix |
| ---- | ---- | ---- |
| Validation Acc: 47.12% <br><br> Testing Accuracy: 36.563% <br><br> Testing F1 Score: 34.52% | ![image](/uploads/1f99a19c201c6b3abaa1ccb1c5925440/image.png) | ![image](/uploads/d138e0b57ba603a020a3c1324d06a361/image.png) |


## II. Data Selection & Data Splitting
- Since classifying the emotion with 10 classes are very challenging, thus, I decided to reduce the class number to 5 which include the male data only.
- I isolated actor no. 21 & 23 to be the test set, and the rest would be the train/ valid set with 8:2 Stratified Split.  
- There are three approaches of data selection.
  - a). Use five emotions noted above.
  - b). Re-sample to 2 class - `Positive` & ` Negative`.
  - c). Re-sample to 3 class - `Positive` `Neutral` & ` Negative`.

- **2 class:**
  - **Positive**: `happy`, `calm`.
  - **Negative**: `angry`, `fearful`, `sad`.
  
- **3 class:**
  - **Positive**: `happy`.
  - **Neutral**:  `calm`, `neutral`.
  - **Negative**: `angry`, `fearful`, `sad`.

- **5 class:**: `happy`, `calm`, `angry`, `fearful`, `sad`.

**Male Dataset**
- Train Set = 640 samples from actor 1- 10.
- Valid Set = 160 samples from actor 1- 10.
- Test Set = 160 samples from actor 11- 12.

**Female Dataset**
- Train Set = 608 samples from actress 1- 10.
- Valid Set = 152 samples from actress 1- 10.
- Test Set = 160 samples from actress 11- 12.


## IV. Model Tuning

- Here is the final model architecture.
  - Added two `Conv1D` layers.
  - Added two `BatchNormalization` layers.
  - Added one `MaxPooling1D` layer.
  - Implemented `Dropout` one more time.
  - Changed Dropout rate.
  - Changed optimizer to `SGD`
- Added Learning Rate Reduce technique.
- Added save best model only on min `val_loss`. 
- This architecture applied to all of the following experiment.

```
model = Sequential()
model.add(Conv1D(256, 8, padding='same',input_shape=(X_train.shape[1],1)))
model.add(Activation('relu'))
model.add(Conv1D(256, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(128, 8, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(MaxPooling1D(pool_size=(8)))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Conv1D(64, 8, padding='same'))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('softmax'))
opt = keras.optimizers.SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=False)
```

## V. Baseline Performance

#### Male 5 class

| Performance | Train Valid Loss | Confusion Matrix |
| ---- | ---- | ---- |
| Validation Acc: 51.25% <br><br> Testing Accuracy: 46.25% <br><br> Testing F1 Score: 45.38% | ![image](/uploads/4a4e37d726b0cc12fb0d31291de98c6d/image.png) | ![image](/uploads/f5a1f14c5819970d547ac15593075af3/image.png) |

#### Female 5 class

| Performance | Train Valid Loss | Confusion Matrix |
| ---- | ---- | ---- |
| Validation Acc: 64.47% <br><br> Testing Accuracy: 43.125% <br><br> Testing F1 Score: 43.33% | ![image](/uploads/d31f4536641e7cfb23b03d7279c5af71/image.png)  | ![image](/uploads/c9639611a81bf2fef4d3ef70527643e8/image.png) |

#### Male 2 class

| Performance | Train Valid Loss | Confusion Matrix |
| ---- | ---- | ---- |
| Validation Acc: 74.38% <br><br> Testing Accuracy: 65.625% <br><br> Testing F1 Score: 64.28% | ![image](/uploads/78b244e03177c9fb4ff79e51775e5c69/image.png) | ![image](/uploads/4a05ca6a0f5e36ebc15b95f9a3d28b1c/image.png) |

#### Male 3 class 

| Performance | Train Valid Loss | Confusion Matrix |
| ---- | ---- | ---- |
| Validation Acc: 60.23% <br><br> Testing Accuracy: 58.5% <br><br> Testing F1 Score: 52.67% | ![image](/uploads/5a394eb34725c1b38985b1b69e68d9d4/image.png) | ![image](/uploads/07247a1a86a6fd117eb814e5222c97cf/image.png) |

## VI. Augmentation

### Male 5 class
##### Dynamic Value Change

| Performance | Train Valid Loss | Confusion Matrix |
| ---- | ---- | ---- |
| Validation Acc: 83.75% <br><br> Testing Accuracy: 38.125% <br><br> Testing F1 Score: 38.46% | ![image](/uploads/0c5c9af8108b0d0904f44e7f5117b586/image.png) | ![image](/uploads/80fb985ab48fa3712f2990c91b486e28/image.png) |

##### Pitch Tuning

| Performance | Train Valid Loss | Confusion Matrix |
| ---- | ---- | ---- |
| Validation Acc: 75% <br><br> Testing Accuracy: 40% <br><br> Testing F1 Score: 38.8% | ![image](/uploads/b0bb323ee1ece5b5e14d22e6ca6e52f0/image.png) | ![image](/uploads/5ce154d08a7fb8c3341c4c35edd12ea8/image.png) |

##### Shifting

| Performance | Train Valid Loss | Confusion Matrix |
| ---- | ---- | ---- |
| Validation Acc: 72.19% <br><br> Testing Accuracy: 42.5% <br><br> Testing F1 Score: 41.94% | ![image](/uploads/175813455ab56d31d3c812403f9905bd/image.png) | ![image](/uploads/2b457dea7b5f55f02928fbc5d37f77df/image.png) |

##### White Noise Adding

| Performance | Train Valid Loss | Confusion Matrix |
| ---- | ---- | ---- |
| Validation Acc: 87.19% <br><br> Testing Accuracy: 40% <br><br> Testing F1 Score: 39.9% | ![image](/uploads/c7ac96933904d318b913c0398e60e583/image.png) | ![image](/uploads/706b07ae514d4768b4827430f58dd469/image.png) |

#### Mix
##### 2 Class: Noise Adding + Shifting

Model Weight: [baseline_2class_ps.h5](/uploads/3c17facdeecacad57bf3cabf94f36224/baseline_2class_ps.h5)

| Performance | Train Valid Loss | Confusion Matrix |
| ---- | ---- | ---- |
| Validation Acc: 85.52% <br><br> Testing Accuracy: 67.5% <br><br> Testing F1 Score: 66.75% | ![image](/uploads/0384164055a7f9459ba621c4c6f6940f/image.png) | ![image](/uploads/329ac972b827e6b625b013b6095a77cf/image.png) |

##### 2 Class: Pitch Tuning + Noise Adding

| Performance | Train Valid Loss | Confusion Matrix |
| ---- | ---- | ---- |
| Validation Acc: 86.46% <br><br> Testing Accuracy: 62.5% <br><br> Testing F1 Score: 59.74% | ![image](/uploads/966a96bbc143381596ef4764ee3af980/image.png) | ![image](/uploads/c229808c77e951c8c047d8751aef2007/image.png) |

##### 5 Class: Noise Adding + Shifting

| Performance | Train Valid Loss | Confusion Matrix |
| ---- | ---- | ---- |
| Validation Acc: 91.25% <br><br> Testing Accuracy: 45% <br><br> Testing F1 Score: 45.26% | ![image](/uploads/2c58c6ecbc413c01e6a4cf643c1b410f/image.png) | ![image](/uploads/c97239840dccb8b8271f1e5b23325b59/image.png) |



#### Male 2 class

| Performance | Train Test Loss | Confusion Matrix |
| ---- | ---- | ---- |
| Validation Acc: 77.34% <br><br> Testing Accuracy: 75% <br><br> Testing F1 Score: 61.39% | ![image](/uploads/16b9d05df0675afb444b34be15b1b2f3/image.png) | ![image](/uploads/fed4b58cdc156b9b91f8dee7b15ccaa2/image.png) |

#### Male 3 class 

| Performance | Train Test Loss | Confusion Matrix |
| ---- | ---- | ---- |
| Validation Acc: 68.18% <br><br> Testing Accuracy: 60.795% <br><br> Testing F1 Score: 54.18% | ![image](/uploads/63d218ed0d265ff85c585ac5bc394b9b/image.png) | ![image](/uploads/383208e2f66a9f3b6f23ae5b5983e2e0/image.png) |

## VII. Insight
