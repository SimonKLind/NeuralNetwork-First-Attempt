# NeuralNetwork

This is a failed attempt at a neural network 
for image classification with the cifar-10 set.

After taking a good look into what was wrong with this, there were quite a few errors that really added up. Possibly the biggest error was that the layers were made completely wrong, obviously BatchNorm is supposed to be after a fully connected layer, before the actual ReLU.

Then the gradient for BatchNorm was wrong as well, I forgot to take into account the effect an input x has through its change in the mean and variance, so it only applied the gradient of one output Yi with respect to Xi.

There was also a slight bug in the gradient of the SVM, The gradient with respect to input Xi should be 1, not Xi. And it should also take into account that the correct label gets a gradent of -1 per other label that doesn't fit the margin.

The SoftMax layer should also be fixed, this version does nothing to prevent exponentials from blowing up...
