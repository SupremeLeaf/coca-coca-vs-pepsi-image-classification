import tensorflow as tf


def load_and_prep_image(filename, img_shape=224, scale=True):
  """
  Reads in an image from filename, turns it into a tensor and reshapes into
  (224, 224, 3).

  Parameters
  ----------
  filename (str): string filename of target image
  img_shape (int): size to resize target image to, default 224
  scale (bool): whether to scale pixel values to range(0, 1), default True
  """
  # Read in the image
  img = tf.io.read_file(filename)
  # Decode it into a tensor
  img = tf.image.decode_jpeg(img)
  # Resize the image
  img = tf.image.resize(img, [img_shape, img_shape])
  if scale:
    # Rescale the image (get all values between 0 and 1)
    return img/255.
  else:
    return img
    
def pred_and_plot(model, filename, class_names):
  """
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.
  """
  # Import the target image and preprocess it
  img = load_and_prep_image(filename)

  # Make a prediction
  pred = model.predict(tf.expand_dims(img, axis=0))

  # Get the predicted class
  if len(pred[0]) > 1: # check for multi-class
    pred_class = class_names[pred.argmax()] # if more than one output, take the max
  else:
    pred_class = class_names[int(tf.round(pred)[0][0])] # if only one output, round

  # Plot the image and predicted class
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False)
  

import datetime

def create_tensorboard_callback(dir_name, experiment_name):
  """
  Creates a TensorBoard callback instand to store log files.

  Stores log files with the filepath:
    "dir_name/experiment_name/current_datetime/"

  Args:
    dir_name: target directory to store TensorBoard log files
    experiment_name: name of experiment directory (e.g. efficientnet_model_1)
  """
  log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir=log_dir
  )
  print(f"Saving TensorBoard log files to: {log_dir}")
  return tensorboard_callback


# Plot the validation and training data separately
import matplotlib.pyplot as plt

def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.

  Args:
    history: TensorFlow model History object (see: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/History)
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend();


def compare_histories(original_history, new_history, initial_epochs=5):
    """
    Compares two TensorFlow model History objects.
    
    Args:
      original_history: History object from original model (before new_history)
      new_history: History object from continued model training (after original_history)
      initial_epochs: Number of epochs in original_history (new_history plot starts from here) 
    """
    
    # Get original history measurements
    acc = original_history.history["accuracy"]
    loss = original_history.history["loss"]

    val_acc = original_history.history["val_accuracy"]
    val_loss = original_history.history["val_loss"]

    # Combine original history with new history
    total_acc = acc + new_history.history["accuracy"]
    total_loss = loss + new_history.history["loss"]

    total_val_acc = val_acc + new_history.history["val_accuracy"]
    total_val_loss = val_loss + new_history.history["val_loss"]

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
    
    
 def compare_images():
  plt.figure()
  plt.subplot(1, 2, 1)
  pepsi_image = view_random_image(target_dir='coke_and_pepsi_all_images/Train/', target_class='pepsi')
  plt.subplot(1, 2, 2)
  coke_image = view_random_image(target_dir='coke_and_pepsi_all_images/Train/', target_class='coca_cola')
  plt.show()


from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Activation, Rescaling
from tensorflow.keras.optimizers import Adam

# Build the model
def base_model():
  model_0 = Sequential([
  Rescaling(1.0/255),                      
  Conv2D(filters=10,
         kernel_size=3,
         strides=1,
         input_shape=(224, 224, 3)),
  Activation(activation='relu'),
  Conv2D(filters=10,
         kernel_size=3,
         strides=1),
  Activation(activation='relu'),
  MaxPool2D(),
  Conv2D(filters=10,
         kernel_size=3,
         strides=1),
  Activation(activation='relu'),
  Conv2D(filters=10,
         kernel_size=3,
         strides=1),
  Activation(activation='relu'),
  MaxPool2D(),
  Flatten(),
  Dense(1, activation='sigmoid')       
  ])
  return model_0
  

def data_augmentation_image_comparison(class_names: list):
  """
  Takes an image from a specified directory and comapres the 
  original image to an augmented version counterpart of itself.
  Args:
    class_names: A list of class names
    
  Returns: Two images of the same image (one with augmentation and the other without)
    
  """
  # Create the target image 
  target_class = random.choice(class_names)
  target_dir = 'coke_and_pepsi_all_images/Train/' + target_class
  print(target_dir)
  random_image = random.choice(os.listdir(target_dir))
  random_image_path = target_dir + '/' + random_image
  print(random_image_path)

  # Read in the random image 
  plt.figure(figsize=(10, 10))
  img = mpimg.imread(random_image_path)
  plt.imshow(img)
  plt.title(f'Original random image from class: {target_class}')
  plt.axis(False)

  # Plot our augmented random image 
  augmented_image = data_augmentation_layer(img)
  plt.figure(figsize=(10, 10))
  plt.imshow(augmented_image)
  plt.title(f'Augmented random image from class: {target_class}')
  plt.axis(False)