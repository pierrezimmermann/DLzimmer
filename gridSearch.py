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


def create_model():
    model = CNN()
    optimizer = tf.keras.optimizers.SGD(0.001)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(optimizer=optimizer, loss=loss_object,
                  metrics=['accuracy'])
    return model


# Grid search to determine the optimal size of batch and the number of epochs


def grid_search_epochs():
    model = KerasClassifier(build_fn=create_model)
    epochs = [100, 200, 300, 400, 500]
    param_grid = dict(epochs=epochs)
    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        n_jobs=1, scoring=['accuracy'], refit='accuracy', cv=3)
    grid_result = grid.fit(X, Y)
    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


grid_search_epochs()


def grid_search_batch_size():
    model = KerasClassifier(build_fn=create_model, epochs=500)
    batch_size = [16, 32, 64, 128]
    param_grid = dict(batch_size=batch_size)
    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        n_jobs=1, scoring=['accuracy'], refit='accuracy')
    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

# Replcae values of epoch and batch_size with results of previous gridsearch

# Grid search to determine the best optimizer with the number of epochs and the batch_size
# determined by the above grid search


def gridSearchOptimizer():
    model = KerasClassifier(build_fn=create_model, epochs=100, batch_size=8)
    optimizer = ['SGD', 'RMSprop', 'Adagrad',
                 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    param_grid = dict(optimizer=optimizer)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    grid_result = grid.fit(X, Y)
    print("Best: %f using %s" %
          (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

# Grid search for the initialization method of weights


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
