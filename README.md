# Classifying frog calls into 9 species using CNNs trained on spectrograms

In this project, I put together code that chunks .wav files into 4 or 10 second segment .wav files. These files are then converted into spectrograms, both mel scale and logarithmic, which will be used for further analysis. 

A convolutional neural network is then trained on the spectrograms to classify them into 9 species. Various methods are used for data augmentation, including standardization and normalization of data. Regularization techniques including the L2-regularizer and Dropout are used in each layer to handle overfitting.

This project was put together as part of the UMW mentorship CS class at Riverbend High School. It was conducted with mentors Sally Burkley and Brian Harnish, as well as additional support from fellow students and team-members Guido Visioni and Derek Ooten.
