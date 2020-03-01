import numpy as np
from sklearn.model_selection import GridSearchCV
from processData import *
from model import CNN
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

images, labels = load_images()
p = np.random.permutation(range(len(images)))
images, labels = images[p], labels[p]
X = images.astype('float32')
Y = labels.astype('float32')

print(X.shape)
print(Y.shape)


# Build function for epochs and batch_size grid searches
def create_model():
    model = CNN()
    optimizer = tf.keras.optimizers.SGD(0.001)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_object,
                  metrics=['accuracy'])
    return model


# Grid search to determine the optimal number of epochs
def grid_search_epochs():
    model = KerasClassifier(build_fn=create_model)
    epochs = [100, 200, 300, 400, 500]
    param_grid = dict(epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        n_jobs=1, scoring=['accuracy'], refit='accuracy', cv=3)
    grid_result = grid.fit(X, Y)
    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_accuracy']
    stds = grid_result.cv_results_['std_test_accuracy']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


# Grid search to determine the optimal batch_size given a number of epochs


def grid_search_batch_size(epochs=400):
    model = KerasClassifier(build_fn=create_model, epochs=epochs)
    batch_size = [16, 32, 64, 128]
    param_grid = dict(batch_size=batch_size)
    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        n_jobs=1, scoring=['accuracy'], refit='accuracy', cv=3)
    grid_result = grid.fit(X, Y)
    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_accuracy']
    stds = grid_result.cv_results_['std_test_accuracy']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


# Build function for optimizer grid search
def create_model_optimizer(optimizer=tf.keras.optimizers.SGD(0.001)):
    model = CNN()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_object,
                  metrics=['accuracy'])
    return model


# Grid search to determine the best optimizer given a number of epochs and a batch_size
def gridSearchOptimizer(epochs=400, batch_size=64):
    model = KerasClassifier(
        build_fn=create_model_optimizer, epochs=epochs, batch_size=batch_size)
    optimizer = ['SGD', 'RMSprop', 'Adagrad',
                 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    param_grid = dict(optimizer=optimizer)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, scoring=[
                        'accuracy'], refit='accuracy', cv=3)
    grid_result = grid.fit(X, Y)
    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_accuracy']
    stds = grid_result.cv_results_['std_test_accuracy']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


# Grid search for the initialization method of weights
def create_model_learning_rate(learning_rate=0.001):
    model = CNN()
    optimizer = tf.keras.optimizers.SGD(learning_rate)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_object,
                  metrics=['accuracy'])
    return model


# Grid search to determine the optimal learning rate given the number of epochs and the batch_size
def gridSearchLearningRate(epochs=300, batch_size=64):
    model = KerasClassifier(
        build_fn=create_model_learning_rate, epochs=epochs, batch_size=batch_size)
    learning_rate = [0.0001, 0.001, 0.01, 0.1]
    param_grid = dict(learning_rate=learning_rate)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1,
                        scoring=['accuracy'], refit='accuracy', cv=2)
    grid_result = grid.fit(X, Y)
    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_accuracy']
    stds = grid_result.cv_results_['std_test_accuracy']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


gridSearchLearningRate()


def grid_search_weight_init():
    model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=8)
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero',
                 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    param_grid = dict(init_mode=init_mode)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    grid_result = grid.fit(X, Y)
    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))
