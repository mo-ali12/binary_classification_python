import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv('diabetes.csv')     # reading dataset

print(dataset.head(2))               # viewing 2 rows and all columns of dataset

print(dataset.describe(include='all'))    # viewing all the dataset to determine what type of data we are dealing with

# sns.pairplot(dataset, hue='Class', vars = dataset.Age[:-1])

# plt.show()

X = dataset.iloc[: , 0:8]
y = dataset.iloc[:,8]
print(X.head(2))

# now standardize the features so that we have mean of 0 and standard deviation 1.

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X = sc.fit_transform(X)

print(X)

# now model selection i.e 30% for test and 70% training

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# using sequential model from keras for building neural network

from keras import Sequential

# dense is used for layers building

from keras.layers import Dense

classifier = Sequential()

# First Hidden Layer

classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal', input_dim=8))  # relu (basically a activation funcytion) , we have 4 nodes in hidden layer

# Second Hidden Layer

classifier.add(Dense (4, activation='relu',kernel_initializer='random_normal'))

#Output Layer

classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal')) # we used sigmoid activation beacuse we are doing binray classification

# now we will complie our neural network use binary_crossentropy for loss function (actual-predicted) as we have binary problem
 # for optimization we use ADAM which is  RMS + Momentum where momentum takes in account the previous gradients to smooth gradient descent.

classifier.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

# fitting the data to model we just created

classifier.fit(X_train, y_train,batch_size=10, epochs=100)


