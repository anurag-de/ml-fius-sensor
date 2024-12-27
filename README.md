# Enhancement of Reliability and Accuracy Red Pitaya Sensor System for Object Detection

This project focuses on improving the reliability and accuracy of a Red Pitaya sensor system for object detection using advanced ultrasonic signal analysis and machine learning.

# Overview
The core of this application lies in precisely detecting the position of the first echo, which represents the reflection of a pulsed ultrasonic beam. The Ultrasonic Proximity sensor achieves this by converting high-frequency sound waves into an analog-to-digital conversion (ADC) data signal. To enhance accuracy, a machine learning (ML) algorithm compares calculated distances with actual distances. A confusion matrix analyzes the deviation between the calculated and actual distances, offering insights into the algorithm's performance. Furthermore, an ML-driven maximum peak detection approach employing a convolutional neural network (CNN) is implemented to refine peak detection precision and bolster the overall reliability of the sensor system.

# Addressing Noise Interference in Peak Detection
A significant challenge is the presence of noise in the ultrasonic sensor measurements, which can lead to the detection of false peaks. Standard Python signal processing libraries like SciPy may mistakenly identify a noisy peak instead of the actual peak. To overcome this, signal processing has been integrated with deep learning. The ADC data for a single signal is divided into multiple windows of 300 values. Each window is then manually annotated to indicate the presence of a real peak.

# Utilizing PSD Spectrograms for Peak Identification
Power spectral density (PSD) spectrograms are employed for their ability to provide a time-signal representation, revealing how power distribution across different frequency components changes over time. This aids in identifying high-frequency noise and extracting relevant features from the signals. PSD spectrograms of fixed dimensions (300 x 300 pixels) and grayscale colormap are generated for windows containing real peaks and those without peaks. These spectrograms serve as input for training a CNN model.

# CNN Model for Binary Classification
A dataset of approximately 1600 manually annotated spectrogram samples (800 for peak and 800 for non-peak) is used to train a CNN model using the Keras library. The model performs binary classification to categorize spectrograms as 'peak' or 'non-peak'. The CNN model architecture comprises four convolution blocks, each followed by a ReLU activation function and a max pooling layer.  A flatten layer, a dense layer with 512 neurons, and a final output layer with one neuron for binary classification complete the model. The model is compiled with the Adam optimizer, binary crossentropy loss function, and accuracy metric. Early stopping is implemented to halt the training process if validation loss does not improve after 3 epochs. Training is conducted for 10 epochs with a batch size of 64.

```python

# Define CNN model for binary image classification
model = Sequential()

# First conv block with 32 filters
model.add(Conv2D(32, (3, 3), padding='same', input_shape=(300, 300, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second conv block with 64 filters
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third conv block with 128 filters
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fourth conv block with 256 filters
model.add(Conv2D(256, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer to convert the 3D outputs to 1D vector
model.add(Flatten())
# Dense layer with 512 neurons
model.add(Dense(512, activation='relu'))
# Output layer with 1 neuron (for binary classification)
model.add(Dense(1, activation='sigmoid'))

```
# Testing the Model
During testing, the same windowing approach is applied to the test signal. Spectrograms are generated for all windows, and the trained CNN model identifies the window most likely to contain a real peak. The real peak is then derived from this window. 
