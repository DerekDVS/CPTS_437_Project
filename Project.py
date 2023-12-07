########################
# step 1 import models #
########################

# import models
import os                                                                           # operating system commands
from PIL import Image                                                               # Image loading
import numpy as np                                                                  # number processing
import matplotlib.pyplot as plt                                                     # used for gui
import seaborn as sns                                                               # used for heatmap
from tensorflow.keras.preprocessing.image import ImageDataGenerator                 # used for processing images I got
from tensorflow.keras.models import Sequential                                      # CNN modeling
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense            # Keras modeling
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score # used for developing confucian matrix
from sklearn import metrics                                                         # metrics for models (accuracy)
from sklearn.preprocessing import LabelEncoder                                      # used for getting labels


##############################
# Step 2: Data Preprocessing #
##############################

# the paths to each dataset
train_path = "437_Custom_Data/train"
val_path = "437_Custom_Data/valid"
test_paths = ["437_Custom_Data/test_normal_bg", "437_Custom_Data/test_blue_bg", "437_Custom_Data/test_red_bg"]
test_names = ["Normal Background", "Blue Background", "Red Background"]

# Define parameters for the model and data preprocessing
train_batch_size, valid_batch_size, test_batch_size = 100, 25, 25
img_height, img_width = 150, 150
epoch_count = 20

# set augmentation technqiues for training images
train_datagen = ImageDataGenerator(
    rescale = 1/255,            # Rescales the pixel to 0 to 1 range
    rotation_range = 40,        # Randomly rotates the image 40 degrees
    width_shift_range = 0.2,    # Randomly shifts the width 20%
    height_shift_range = 0.2,   # Randomly shifts the height 20%
    shear_range = 0.2,          # Randomly shears the image 20% range
    zoom_range = 0.2,           # Randomly zooms a range of 20%
    horizontal_flip = True      # Randomly flips the image
)

# set augmentation technqiues for validation images
validation_datagen = ImageDataGenerator(
    rescale = 1.0/255.0         # Rescales the pixel to 0 to 1 range
)

# set augmentation technqiues for test images
test_datagen = ImageDataGenerator(
    rescale = 1.0/255.0         # Rescales the pixel to 0 to 1 range
)

# Load and augment training images
train_generator = train_datagen.flow_from_directory(
    train_path,                             # path to the training data
    target_size = (img_width, img_height),  # specifies the resize dimensions
    batch_size = train_batch_size,          # determines the number of images in each batch that will be fed to the model
    class_mode = 'categorical'              # return type labels one hot encoded labels in this case
)

# Load and augment validation images
validation_generator = validation_datagen.flow_from_directory(
    val_path,                               # path to the validation data
    target_size = (img_width, img_height),  # specifies the resize dimensions
    batch_size = valid_batch_size,          # determines the number of images in each batch that will be fed to the model
    class_mode = 'categorical'              # return type labels one hot encoded labels in this case
)

# create array to hold the test generators
test_generators = []

# add each test_generator
for i, test_path in enumerate(test_paths):
    # Load and augment testing images
    test_generators.append(
        test_datagen.flow_from_directory(
            test_path,                              # path to the testing data
            target_size = (img_width, img_height),  # specifies the resize dimensions
            batch_size = test_batch_size,           # determines the number of images in each batch that will be fed to the model
            class_mode = 'categorical'              # return type labels one hot encoded labels in this case
        )
    )

############################
# Step 3: Model Generation #
############################

# Build a simple CNN model
model = Sequential()                                                                      # Initializes a sequential model where layers are added
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)))  # Adds 2D Conv layer with 32 filters using ReLu activation
model.add(MaxPooling2D((2, 2)))                                                           # Adds max pooling layer which reduces spatial dimensions

# increase convolutional and max pooling layers which increase the depth of the cnn
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# flattens the 3D output to a 1D vector
model.add(Flatten())

# adds the fully connected layer through relu activation
model.add(Dense(512, activation='relu'))

# adds output layer of 8 (one for each class) neurons for mult class classification
model.add(Dense(8, activation='softmax'))

# Compile the model
model.compile(
    optimizer='adam',                 # adam optimization algorithm
    loss='categorical_crossentropy',  # loss function for training for mult class classification problems
    metrics=['accuracy']              # evaluation metric during training for accuracy
)

###########################
# Step 4: Model Training #
###########################

# Train the model
trained_model = model.fit(
    train_generator,                                                        # the generator of training images
    steps_per_epoch = train_generator.samples // train_batch_size,          # the amount of iterations of reading the batches
    epochs = epoch_count,                                                   # total amount of rounds
    validation_data = validation_generator,                                 # utilize a validation generator to prevent overfitting
    validation_steps = validation_generator.samples // valid_batch_size     # the amount of iterations of reading the batches
)

# get accuracy of graph during training and validation
train_accuracy = trained_model.history['accuracy']
val_accuracy = trained_model.history['val_accuracy']

############################
# Step 5: Model Evaluation #
############################

# iterate through the generators and make evaluations on 
for i, test_generator in enumerate(test_generators):
    loss, accuracy = model.evaluate(
                        test_generator,                                   # the test generator
                        steps= test_generator.samples // test_batch_size  # the amount of iterations of reading the batches
                    )

    # print accuracy info
    print(f"{test_names[i]}\nEvaluation Accuracy: {accuracy * 100:.2f}%\nEvaluation Loss: {loss * 100:.2f}%\n")


###########################
# Dtep 6 Model Prediction #
###########################

# stores all testing matrix data
confusion_matrixs = []

# iterate through the generators and make tests on 
for i, test_generator in enumerate(test_generators):

    # get the amount of testing steps
    test_steps = test_generator.samples // test_batch_size 

    # predict the model based on the testing data
    predictions = model.predict(
        test_generator,         # load images from the generator 
        steps = test_steps,     # the amount of images to load per step
        verbose = 1             # display loading process
    )

    # convert to class labels
    predicted_labels = np.argmax(predictions, axis=1)

    # get the true labels of the class from the generator
    true_labels = test_generator.classes

    # Calculate accuracy
    accuracies = []
    accuracy = accuracy_score(true_labels, predicted_labels)
    accuracies.append(accuracy)

    # Print accuracies
    for i, accuracy in enumerate(accuracies):
        print(f"{test_names[i]} Accuracy: {accuracy * 100:.2f}%")

    # get the actual names of the labels
    label_names = list(test_generator.class_indices.keys())

    # Generate matrix
    confusion_matrixs.append(confusion_matrix(true_labels, predicted_labels))

    # Print classification report
    report = classification_report(true_labels, predicted_labels, target_names=label_names)
    print(test_names[i])
    print(report)

##########################
# Step 7: Model Analysis #
##########################

# set the plot figure=
fig = plt.figure(figsize=(16, 16))

# define the grid layout
gs = fig.add_gridspec(2, 3)

# set axis locations
axis1 = fig.add_subplot(gs[0, :])
axis2 = fig.add_subplot(gs[1, 0])
axis3 = fig.add_subplot(gs[1, 1])
axis4 = fig.add_subplot(gs[1, 2])

# set the first axis
axis1.plot(train_accuracy, label='Train Accuracy', color='blue')
axis1.plot(val_accuracy, label='Validation Accuracy', color='orange')
axis1.set_title('Model Training Accuracy Data')
axis1.set_xlabel('Epoch')
axis1.set_ylabel('Accuracy')

# set variables to showcase on axis
heatmap_axis = [axis2, axis3, axis4]
colors = ['Greens', 'Blues', 'Reds']

# iterate through the matrixs and make show heatmap
for i, confusion_matrix in enumerate(confusion_matrixs):
    # create heat map of most picked elements
    sns.heatmap(
        confusion_matrix,            # the matrix to display
        xticklabels=label_names,     # The name of the x's
        yticklabels=label_names,     # The name of the y's
        annot=True,                  # Show total images picked
        fmt='d',                     # Set the format to integers 
        cmap=colors[i],              # Set the heatmap from light to dark green
        ax = heatmap_axis[i]         # set the assigned axis
    )

    # set the graph labels
    title = 'Prediction Lables vs True Labels on ' + test_names[i]
    heatmap_axis[i].set_title(title)
    heatmap_axis[i].set_xlabel('Predicted labels')
    heatmap_axis[i].set_ylabel('True labels')

# show the graph
plt.tight_layout()  # adjust spacing to fit both graphs
plt.show()