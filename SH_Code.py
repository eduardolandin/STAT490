from tensorflow import keras
from numpy import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import Adam
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

######################################################
#DATA
######################################################
n = 1000
X = zeros(n)
Y = zeros(n)
Z = zeros(n)
X = sort(random.uniform(0., 1., n))
Z = X**2
for i in range(0, n):
    Y[i] = X[i]**2 + random.normal(0, 0.2, 1)

######################################################
#NETWORK
######################################################
num = 10
msemse = 50 * ones(num)
kiki = 0
while kiki < num:
    print(kiki)
    z = relu
    # increasing Glorot initialization
    wgt1 = RandomUniform(minval=0., maxval=1., seed=None)
    wgt2 = RandomUniform(minval=-1., maxval=0., seed=None)
    wgt3 = RandomUniform(minval=-sqrt(6.) / sqrt(10.), maxval=0., seed=None)
    wgt4 = RandomUniform(minval=0., maxval=sqrt(6.) / sqrt(10.), seed=None)

    # Adam optimizer
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.00021)
    # earlystop if MSE does not decrease
    earlystop = keras.callbacks.EarlyStopping(monitor='loss', patience=400, min_delta=0.00000001, verbose=1,
                                              mode='auto')
    callbacks = [earlystop]

    # DNN
    model = Sequential()
    model.add(Dense(5, input_dim=1, kernel_initializer=wgt1, bias_initializer=wgt2, activation=z))
    model.add(Dense(5, kernel_initializer=wgt4, bias_initializer=wgt3, activation=z))
    model.add(Dense(5, kernel_initializer=wgt4, bias_initializer=wgt3, activation=z))
    model.add(Dense(1, kernel_initializer=wgt1, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer=opt)
    model.fit(X, Y, epochs=1500, callbacks=callbacks, verbose=0)
    predicted = model.predict(X)
    msemse[kiki] = sum((Y - predicted[:, 0]) ** 2)
    print(msemse[kiki] / n)
    if msemse[kiki] < 50:  # <0.000002 to reproduce simulations from paper
        kiki = num
    kiki = kiki + 1

mseDNN = min(msemse) / n
print("MSE DNN", mseDNN)
##to reproduce simulations from paper
# mse4[rep]=mseDNN

plt.plot(X, Z, X, predicted)
plt.ylim((-0.1, 1.1))
plt.title('${DNN}$')
plt.show()

# plot
fig = plt.figure(figsize=(20, 4), dpi=80)
#fig.patch.set_facecolor('white')
#ax = fig.add_subplot(141)
#ax.plot(X, Z, X, y_hatMARS)
#plt.ylim((-0.1, 1.1))
#plt.title('${MARS}$')
#ax = fig.add_subplot(142)
#ax.plot(X, Z, X, y_hatHOMARS)
#plt.ylim((-0.1, 1.1))
#plt.title('$HO$-$MARS$')
#ax = fig.add_subplot(143)
#ax.plot(X, Z, plotx, ploty)
#plt.ylim((-0.1, 1.1))
#plt.title('${FS}$')
ax = fig.add_subplot(144)
ax.plot(X, Z, X, predicted)
plt.ylim((-0.1, 1.1))
plt.title('${DNN}$')
plt.show()
##to reproduce simulations from paper
# print "MARS", sum(mse1)/100
# print "HOMARS", sum(mse2)/100
# print "FS", sum(mse3)/100
# print "DNN", sum(mse4)/100
# print "SDE MARS", sqrt(sum((mse1-sum(mse1)/100)**2)/(100-1))
# print "SDE HOMARS", sqrt(sum((mse2-sum(mse2)/100)**2)/(100-1))
# print "SDE FS", sqrt(sum((mse3-sum(mse3)/100)**2)/(100-1))
# print "SDE DNN", sqrt(sum((mse4-sum(mse4)/100)**2)/(100-1))
