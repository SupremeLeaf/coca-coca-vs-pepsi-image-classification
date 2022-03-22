# Summary
We build a **binary image classification** machine learning model.

The data used consists of 100 training images and 20 testing images for both classes (**coke** and **pepsi**).
Given the small dataset the overall objective was to just start with a small base-line model and then get progressively larger and more complicated in order to try and improve the model's accuracy as much as possible given what we had to work with. 

# Overview of Model WorkFlow

Model_0 was a fairly simple Keras `Sequential` Model that consisted of layers: `Conv2D`, `MaxPool2d`, `Dense` `Activation` and `GlobalMaxPooling2D`. 

We then built `model_1` which had the same architecture as `model_0` except that **data augmentation** was added.

`Model_2` upped the game and introduced **transfer learning** with using `keras.applications.EfficientNetB0`. Using this new framework we managed to substantially **decrease the loss** and **increase the accuracy** on the test data. 

The issue however is that the model was badly **overfitting** on the training data so in order to combat this we first lowered the **learning rate** by 10x and then introuduced **data augmentation** into the mix. 

The accuracy on the testing data didn't end up very good in the end (40%) but we did manage to increase the accuracy from less than 0.10 to about 0.40 using the power of transfer learning. Additionally we even managed to make some accurate predictions on custom data.