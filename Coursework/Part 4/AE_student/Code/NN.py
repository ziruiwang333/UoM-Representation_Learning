import numpy as np
import math
from RBM import RBM
import matplotlib.pyplot as plt


class Layer():
    def __init__(self, prev_layer):
        self.ID = None
        self.n = prev_layer.n + 1
        self.input_shape = prev_layer.output_shape
        self.output_shape = prev_layer.output_shape
        self.sparse = False
        return

    def forward(self, x):
        return x

    def backward(self, x):
        return x


class InputLayer(Layer):
    def __init__(self, _input_shape):
        self.ID = "Input"
        self.n = 0
        self.input_shape = _input_shape
        self.output_shape = _input_shape
        return


class Dense(Layer):
    def __init__(self, prev_layer, units, seed):
        super().__init__(prev_layer)
        self.ID = "Dense"
        self.np_rand = np.random.RandomState(seed=seed)
        self.output_shape = (self.input_shape[0], units)
        xavier_bound = np.sqrt(6) / (np.sqrt(self.input_shape[1] + units))  # xavier/golrot weight initialization
        self.w = np.reshape(self.np_rand.uniform(-xavier_bound, xavier_bound, units * self.input_shape[1]),
                            newshape=(self.input_shape[1], units))
        self.b = np.zeros(units)
        self.dw = np.zeros((self.input_shape[1], units))
        self.last_dw = self.dw
        self.db = np.zeros((units))
        self.last_db = self.db

    def forward(self, x):
        self.saved_x = x
        return x @ self.w + self.b

    def backward(self, d_out):
        dx = d_out @ (self.w.T)
        # self.prev_dw = self.dw
        self.dw = (self.saved_x.T) @ d_out
        self.db = np.sum(d_out, axis=0)
        return dx

    def update(self, scaled_dw, scaled_db, momentum):
        new_dw = scaled_dw + momentum * self.last_dw
        self.w -= new_dw
        self.last_dw = new_dw
        new_db = scaled_db + momentum * self.last_db
        self.b -= scaled_db
        self.last_db = new_db


class ReLU(Layer):
    def __init__(self, prev_layer):
        super().__init__(prev_layer)
        self.ID = "ReLU"

    def forward(self, x):
        self.saved_x = x
        return np.maximum(x, 0)

    def backward(self, d_out):
        return np.where(self.saved_x < 0, 0, self.saved_x)


class Sigmoid(Layer):
    def __init__(self, prev_layer):
        super().__init__(prev_layer)
        self.ID = "Sigmoid"

    def enable_sparsity(self, beta, p, eps=0.0001, verbose=False):
        """
        enables KL sparsity in the layer
        beta: weight of sparsity penalty
        p: level of sparsity
        eps: prevents small numbers to appear in denominators
        """
        self.sparse = True
        self.beta = beta
        self.p = p
        self.s_eps = eps
        # self.v=verbose
        # self.kl_div=0

    def forward(self, x):
        self.saved_x = np.copy(x)
        s = 1.0 / (1 + np.exp(-x))
        if (self.sparse is True):
            self.saved_mean_activation = np.mean(s, axis=0)
        return s

    def backward(self, d_out):
        s = 1.0 / (1 + np.exp(-self.saved_x))
        if self.sparse is False:
            return s * (1 - s)
        else:
            s_mean = self.saved_mean_activation
            sparse_term = self.beta * (((1 - self.p) / (1 - s_mean + self.s_eps)) - (self.p / (s_mean + self.s_eps)))
            return s * (1 - s), sparse_term


class Tanh(Layer):
    def __init__(self, prev_layer):
        super().__init__(prev_layer)
        self.ID = "Tanh"

    def forward(self, x):
        self.saved_x = x
        return np.tanh(x)

    def backward(self, d_out):
        tanh = np.tanh(self.saved_x)
        return 1 - (tanh * tanh)


class Linear(Layer):
    def __init__(self, prev_layer):
        super().__init__(prev_layer)
        self.ID = "Linear"

    def forward(self, x):
        self.saved_x = x
        return x

    def backward(self, d_out):
        return np.ones((self.saved_x.shape))


class MSE_loss():
    def loss(self, y_true, y_pred):
        self.diff = (y_pred - y_true)
        return np.mean(self.diff * self.diff, axis=0)

    def loss_prim(self):
        return self.diff


ActivationDict = {"linear": Linear, "relu": ReLU, "sigmoid": Sigmoid, "tanh": Tanh}


class Network():
    def __init__(self, layer_units, activations=["relu"], seed=0, copy_net=None):
        self.loss_fn = MSE_loss()
        self.lr = 0.1
        self.lr_decay = 0.0
        self.train_err_hist = []
        self.val_err_hist = []
        self.tied_layers = None
        self.batch_size = None  # not needed for NN matrix multiplications
        if copy_net is None:
            if not (isinstance(activations, list)):
                activations = [activations]
            if (len(activations) == 1):
                activations = activations * (len(layer_units) - 1)
            elif (len(activations) != len(layer_units) - 1):
                raise ValueError("Error: layer and activation list length mismatch!")
            self.arch = layer_units
            self.activation_types = activations
        else:
            assert isinstance(copy_net,Network)
            self.arch = copy_net.arch
            self.activation_types = copy_net.activation_types
        self.np_rand = np.random.RandomState(seed=seed)
        self.output_size = self.arch[-1]
        self.input_size = self.arch[0]
        n_layers = len(self.arch)
        self.layers = [InputLayer((self.batch_size, self.input_size))]
        for i in range(n_layers - 2):
            self.layers.append(Dense(self.layers[-1], self.arch[i + 1], seed=seed + i))
            self.layers.append(ActivationDict[self.activation_types[i]](self.layers[-1]))
        # final layer
        self.layers.append(Dense(self.layers[-1], self.arch[-1], seed=seed + n_layers))
        self.layers.append(ActivationDict[self.activation_types[-1]](self.layers[-1]))
        #copying
        if not(copy_net is None):
            for n_layer in range(1,n_layers):
                if copy_net.layers[n_layer].ID == "Dense":
                    copy_dense_weights_from_net(copy_net.layers[n_layer], self.layers[n_layer])
        return


    def tie_layer_weights(self, layer_a, layer_b):
        """
        Tie layers with ids layer_a and layer_b.
        Only ONE PAIR of layers can be tied per network!!!
        """
        assert layer_a != layer_b
        assert self.layers[layer_b].w.shape == (self.layers[layer_a].w.T).shape
        self.layers[layer_b].w = self.layers[layer_a].w.T
        self.tied_dw = None
        self.second_tied_layer_scaled_db = None
        # append values so that the first layer is under the 0 index
        self.tied_layers = [min(layer_a, layer_b), max(layer_a, layer_b)]

    def set_loss_fn(self, f):
        self.loss_fn = f

    def set_lr(self, lr, lr_decay=0.0, momentum=0.0, weight_decay=0.0):
        self.lr = lr
        self.lr_decay = lr_decay
        self.momentum = momentum
        self.weight_decay = weight_decay

    def decay_lr(self):
        self.lr *= (1 - self.lr_decay)

    def get_layer_output(self, x, n):
        """
        Get the output of the n-th layer after forward propagating the X input.
        :param x: input data, numpy array
        :param n: int, the layer id
        :return: numpy array corresponding to the layer's output
        """
        i = 0
        for layer in self.layers:
            x = layer.forward(x)
            if (i == n):
                break
            else:
                i += 1
        return x

    def predict(self, x):
        """
        Get prediction from data.
        :param x: input numpy array
        :return:
        """
        return self.get_layer_output(x, len(self.layers))

    def evaluate(self, x, y_true, return_mean=False, metric=None):
        """
        Evaluate the network performance on the given data
        :param x: input data
        :param y_true: targets
        :param return_mean: bool, if to return the mean of the losses or just the losses
        :param metric: possible custom function to evaluate loss
        :return:
        """
        y_pred = self.predict(x)
        if (metric is None):
            loss = self.loss_fn.loss(y_true, y_pred)
        else:
            loss = metric(y_true, y_pred)
        if (return_mean is False):
            return loss
        else:
            return np.mean(loss)

    def train_batch(self, x, y_true):
        err = self.evaluate(x, y_true, return_mean=False)
        self.update_layers_SGD(self.lr, x.shape[0])
        return np.mean(err)

    def update_layers_SGD(self, lr, batch_size):
        dCost = self.loss_fn.loss_prim()
        for i in range(len(self.layers) - 1, 0, -1):  # start from the last layer
            layer = self.layers[i]
            if (layer.ID == "Dense"):
                dCost = layer.backward(dCost)
                scaled_dw = (layer.dw * self.lr / batch_size)
                scaled_db = (layer.db * self.lr / batch_size)
                if self.tied_layers is None:
                    layer.update(scaled_dw, scaled_db, self.momentum)
                else:  # if there is a pair of tied layers present
                    if (self.tied_layers[1] == i):  # comes first
                        self.tied_dw = np.copy(scaled_dw)
                        self.second_tied_layer_scaled_db = scaled_db
                    elif (self.tied_layers[0] == i):  # comes second
                        layer.update(self.tied_dw.T + scaled_dw, scaled_db, self.momentum)
                        self.layers[self.tied_layers[1]].update(self.tied_dw + scaled_dw.T, self.
                                                                second_tied_layer_scaled_db,
                                                                self.momentum)
                    else:
                        layer.update(scaled_dw, scaled_db, self.momentum)
            elif (layer.sparse == True and layer.ID == "Sigmoid"):
                mult, sparse_term = layer.backward(None)
                dCost = dCost * mult + np.tile(sparse_term, (dCost.shape[0], 1))
            else:
                dCost *= layer.backward(None)
        return dCost

    def get_summary(self):
        """
        Print the network summary (architecture).
        :return:
        """
        for layer in self.layers:  # start from the last layer
            print(str(layer.n) + ": " + layer.ID + "\t in:" + str(layer.input_shape) + "\t out:" + str(
                layer.output_shape))

    def fit(self, x, y_true, x_val=None, y_true_val=None, batch_size=8, epochs=10, patience=-1):
        """
        Train the Network object using SGD.
        :param x: numpy array used for training data (n_samples,n_features)
        :param y_true: numpy array used for training targets (n_samples,n_outputs)
        :param x_val: numpy array used for validation data (n_val_samples,n_features)
        :param y_true_val: numpy array used for validation targets (n_val_samples,n_outputs)
        :param batch_size: int, batch size used for training
        :param epochs: int, number of epochs before the training is finished
        :param patience: int, number of epochs without improvement before the training is finished
        """
        def sample(indexes, n_samples):
            if_new_epoch = len(indexes) <= n_samples
            if (if_new_epoch):
                batch_ind = np.copy(indexes)
                indexes = np.arange(x.shape[0])
                batch_ind_2, indexes, _ = sample(indexes, n_samples - len(batch_ind))
                batch_ind = np.concatenate((batch_ind, batch_ind_2))
            else:
                removed_indices = self.np_rand.choice(len(indexes), n_samples, replace=False)
                batch_ind = np.copy(indexes[removed_indices])
                indexes = np.delete(indexes, removed_indices)
            return batch_ind, indexes, if_new_epoch

        n_samples = x.shape[0]
        indexes = np.arange(n_samples)
        epoch_counter = 0
        batch_counter = 0
        train_err = 0.0
        val_err = 0.0
        best_val_err = 10e9
        current_patience = 0
        while epoch_counter < epochs:
            batch_ind, indexes, if_new_epoch = sample(indexes, batch_size)
            train_err += self.train_batch(x[batch_ind], y_true[batch_ind])
            batch_counter += 1
            if if_new_epoch == True:
                # print("\r"+str(epoch_counter),end=": ")
                epoch_counter += 1
                # print("Train err: "+str(np.around(train_err*1.0/batch_counter,5)),end="\t")
                if not (x_val is None):  # if testing data present
                    val_err = np.mean(self.evaluate(x_val, y_true_val))
                    # print("Val err:"+str(np.around(val_err,5)),end="\t")
                    if patience >= 0:
                        if val_err < best_val_err:
                            best_val_err = val_err
                            current_patience = 0
                        else:
                            current_patience += 1
                        # print("Patience:"+str(current_patience),end="\r", flush=True)
                print("\r" + str(epoch_counter) + ":\t train err: " + str(np.around(train_err * 1.0 / batch_counter, 5))
                      + "\t val err: " + str(np.around(val_err, 5)) + "\t patience: " + str(current_patience),
                      flush=True, end="\t")
                self.train_err_hist.append(train_err * 1.0 / batch_counter)
                self.val_err_hist.append(val_err)
                train_err = 0
                self.decay_lr()
                if patience >= 0 and current_patience == patience:
                    print("\n Patience condition reached,best validation performance: " + str(best_val_err))
                    break
                batch_counter = 0
        return


# some helper functions
def copy_dense_weights_from_net(l_source, l_target):
    """
    Copy weights from source layer to target layer.
    """
    assert l_source.input_shape == l_target.input_shape
    assert l_source.output_shape == l_target.output_shape
    l_target.w = np.copy(l_source.w)
    l_target.b = np.copy(l_source.b)


def copy_dense_weights_from_rbm(rbm_source, l_target, if_encoder):
    """
    Copy RBM->NN
    :param rbm_source:
    :param l_target: target layer object
    :param if_encoder: encoding RBM layer or decoding
    :return:
    """
    if if_encoder == 1:
        l_target.w = np.copy(rbm_source.W)
        l_target.b = np.copy(rbm_source.b)
    else:
        l_target.w = np.copy(rbm_source.W.T)
        l_target.b = np.copy(rbm_source.a)


def pretrain_autoencoder(net, x, x_val, rbm_lr=0.001, rbm_use_gauss_visible=False,
                         rbm_use_gauss_hidden=True,
                         rbm_mom=0.5, rbm_weight_decay=0.0000, rbm_lr_decay=0.0,
                         rbm_batch_size=100,
                         rbm_epochs=100, rbm_patience=-1, verbose=1):
    final_arch = net.arch[1:math.ceil(len(net.arch) / 2.0)]  # without input layer
    n_dense_layers = len(final_arch)
    rbm_list = []
    # loop for training the RBMs
    for i in range(n_dense_layers):
        print("\nFine tuning layer number " + str(i))
        if i == 0:
            x_new = x
            x_val_new = x_val
        else:
            x_new = rbm_list[-1].get_h(x_new)
            x_val_new = rbm_list[-1].get_h(x_val_new)
        rbm = RBM(x_new.shape[1], final_arch[i], use_gaussian_visible_sampling=rbm_use_gauss_visible,
                  use_gaussian_hidden_sampling=rbm_use_gauss_hidden,
                  use_sample_vis_for_learning=False)
        rbm.set_lr(rbm_lr, rbm_lr_decay, momentum=rbm_mom,
                   weight_decay=rbm_weight_decay, )
        rbm.fit(x_new, x_val_new, batch_size=rbm_batch_size,
                epochs=rbm_epochs, patience=rbm_patience)
        rbm_list.append(rbm)
    rbm_iterator = 0
    rbm_iterator_direction = 1
    # loop to copy the weights from rbm to NN
    for n_layer in range(len(net.layers)):
        if net.layers[n_layer].ID == "Dense":
            copy_dense_weights_from_rbm(rbm_list[rbm_iterator], net.layers[n_layer], rbm_iterator_direction)
            if rbm_iterator == len(rbm_list) - 1 and rbm_iterator_direction == 1:
                rbm_iterator_direction = -1
            else:
                rbm_iterator += rbm_iterator_direction
    print("Pre training finished!")
    return rbm_list



def plot_results(model, x_test, hidden_units=None):
    # plot two plots for MNIST test dataset.
    # One heatmap plot and one reconstruction plot.
    if isinstance(model, Network):
        reco = model.predict(x_test)
    else:
        reco = model.reconstruct(x_test, force_sample_visible=False)
    perf_array = np.array([np.mean((reco[i] - x_test[i]) * (reco[i] - x_test[i])) for i in range(100)])
    mean = np.mean(perf_array)
    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(121)
    f = ax.imshow(np.reshape(perf_array, newshape=(10, 10)))
    f.set_cmap('Reds')
    f.axes.get_xaxis().set_visible(False)
    if hidden_units is None:
        plt.title("MSE: " + str(np.round(mean, 4)))
    else:
        plt.title("Units: " + str(hidden_units) + " MSE: " + str(np.round(mean, 4)))
    ax = plt.subplot(122)
    digits = np.zeros((10 * 28, 10 * 28))
    for i in range(100):
        digits[28 * (i // 10):28 * (i // 10) + 28, 28 * (i % 10):28 * (i % 10) + 28] = np.reshape(reco[i],
                                                                                                  newshape=(28, 28))
    f = ax.imshow(digits)
    f.axes.get_xaxis().set_visible(False)
    f.axes.get_yaxis().set_visible(False)
    plt.show()
    return mean
def plot_alphadigit_results(model, X, labels, n_columns=25):
    """
    Plotting alphadigit dataset reconstructions.
    :param model: Network object
    :param X: data
    :param labels: labels of the supplied data
    :param n_columns: number of images per row
    :return:
    """
    X=X[np.argsort(labels)]
    w=16
    h=20
    reco = model.predict(X)
    n_test = X.shape[0]
    perf_array = np.array([np.mean((reco[i] - X[i]) * (reco[i] - X[i])) for i in range(n_test)])
    mean = np.mean(perf_array)

    n_rows = math.ceil(1.0*n_test/n_columns)
    reco = np.reshape(reco,(n_test,h,w))
    x = np.reshape(X, (n_test, h, w))

    fig = plt.figure(figsize=(10, 5))
    ax = plt.subplot(121)
    arr1 = np.zeros((n_rows * h, n_columns * w))
    for i in range(n_test):
        row = i//n_columns
        col = i%n_columns
        arr1[h*row:h*(row+1),w*col:w*(col+1)] = np.reshape(reco[i],newshape=(h, w))
    f = ax.imshow(arr1)
    f.axes.get_xaxis().set_visible(False)
    f.axes.get_yaxis().set_visible(False)
    ax = plt.subplot(122)
    arr2 = np.zeros((n_rows * 20, n_columns * 16))
    for i in range(n_test):
        row = i // n_columns
        col = i % n_columns
        arr2[h * row:h * (row + 1), w * col:w * (col + 1)] = np.reshape(x[i], newshape=(h, w))
    f = ax.imshow(arr2)
    f.axes.get_xaxis().set_visible(False)
    f.axes.get_yaxis().set_visible(False)
    plt.show()
    return mean
