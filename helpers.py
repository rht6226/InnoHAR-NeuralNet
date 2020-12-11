 # Imports
from statistics import mean 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import itertools
import matplotlib
import numpy as np

## Mean Model Helpers
# Mean Accuracy calculator
def get_mean_accuracy(modelList):
    return mean([modelDict['accuracy'] for modelDict in modelList])
   
# Find the model whose performence is closest to the average accuracy
def get_average_model(modelList):
    mean_accuracy = get_mean_accuracy(modelList)
    index = min(range(len(modelList)), key=lambda i: abs(modelList[i]['accuracy'] - mean_accuracy))
    return modelList[index]['model'], modelList[index]['history']


## Plotting Helpers

# Plotting Accuracy
def plot_model_accuracy(details, history, figureSize=[8, 8]):
    matplotlib.rcParams['figure.figsize'] = figureSize
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy for {}'.format(details))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# Plotting Loss
def plot_model_loss(details, history, figureSize=[8, 8]):
    matplotlib.rcParams['figure.figsize'] = figureSize
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss for {}'.format(details))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# Plotting Confusion Matrix

def plot_cm(model, test_data, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, figureSize=[10, 10]):

    classes = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
    testX, testy = test_data
    matplotlib.rcParams['figure.figsize'] = figureSize

    # Predict Y, And get Confusion Matrix
    predictedY = model.predict(testX)
    predictedY = np.argmax(predictedY, axis = 1)
    testy = np.argmax(testy, axis=1)
    
    cm = confusion_matrix(y_true=testy, y_pred=predictedY)

    # Plotting
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Print Classification Report

def print_classification_report(model, test_data, title=''):
    classes = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']
    testX, testy = test_data
    # Get predicted and true values
    predictedY = model.predict(testX)
    y_pred = np.argmax(predictedY, axis = 1)
    y_true = np.argmax(testy, axis=1)

    print("Classification Report for '{}': \n".format(title))
    print(classification_report(y_true, y_pred, target_names=classes, digits=5))
