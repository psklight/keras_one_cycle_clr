try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt


class OneCycle(keras.callbacks.Callback):

    def __init__(
            self,
            lr_range,
            momentum_range=None,
            reset_on_train_begin=True,
            record_frq=10,
            verbose=False):

        super(OneCycle, self).__init__()

        self.lr_range = lr_range

        self.momentum_range = momentum_range
        if momentum_range is not None:
            err_msg = "momentum_range must be a 2-numeric tuple (m1, m2)."
            if not isinstance(momentum_range, (tuple,)) or len(momentum_range) != 2:
                raise ValueError(err_msg)

        self.reset_on_train_begin = reset_on_train_begin
        self.record_frq = record_frq
        self.verbose = verbose

        # helper tracker
        self.at_iteration = 0
        self.log = {}  # history in iterations
        self.log_ep = {}  # history in epochs
        self.stop_training = False

        # counter
        self.current_iter = 0

    def get_current_lr(self, n_iter=None):
        """
        A helper function to calculate a current learning rate based on current iteration number.

        :return lr: a current learning rate.
        """
        if n_iter is None:
            n_iter = self.n_iter

        x = float(self.current_iter) / n_iter
        # first half
        if x < 0.5:
            amp = self.lr_range[1] - self.lr_range[0]
            lr = (np.cos(x * np.pi * 2 - np.pi) + 1) * amp / 2.0 + self.lr_range[0]
        if x >= 0.5:
            amp = self.lr_range[1]
            lr = (np.cos((x - 0.5) * np.pi * 2) + 1) / 2.0 * amp
        return lr

    def get_current_momentum(self, n_iter=None):
        """
        A helper function to calculate a current momentum based on current iteration number.

        :return momentum: a current momentum.
        """
        if n_iter is None:
            n_iter = self.n_iter
        amplitude = self.momentum_range[1] - self.momentum_range[0]
        delta = (1 - np.abs(np.mod(self.current_iter, n_iter) * 2.0 / n_iter - 1)) * amplitude
        return delta + self.momentum_range[0]

    def set_momentum(self):
        keys = dir(self.model.optimizer)
        mom = self.get_current_momentum()
        if "momentum" in keys:
            K.set_value(self.model.optimizer.momentum, mom)
        if "rho" in keys:
            K.set_value(self.model.optimizer.rho, mom)
        if "beta_1" in keys:
            K.set_value(self.model.optimizer.beta_1, mom)

    @property
    def cycle_momentum(self):
        return self.momentum_range is not None

    def on_train_begin(self, logs={}):
        self.n_epoch = self.params['epochs']

        # find number of batches per epoch
        if self.params['batch_size'] is not None:  # model.fit
            self.n_bpe = int(np.ceil(self.params['samples'] / self.params['batch_size']))
        if self.params['batch_size'] is None:  # model.fit_generator
            self.n_bpe = self.params['samples']

        self.n_iter = self.n_epoch * self.n_bpe
        # this is a number of iteration in one cycle

        self.current_iter = 0

    def on_train_batch_begin(self, batch, logs={}):
        K.set_value(self.model.optimizer.lr, self.get_current_lr())
        if self.cycle_momentum:
            self.set_momentum()

    def on_train_batch_end(self, batch, logs={}):

        if self.verbose:
            print("lr={:.2e}".format(self.get_current_lr()), ",", "m={:.2e}".format(self.get_current_momentum()))

        # record according to record_frq
        if np.mod(int(self.current_iter), self.record_frq) == 0:
            self.log.setdefault('lr', []).append(self.get_current_lr())
            if self.cycle_momentum:
                self.log.setdefault('momentum', []).append(self.get_current_momentum())

            for k, v in logs.items():
                self.log.setdefault(k, []).append(v)

            self.log.setdefault('iter', []).append(self.at_iteration)

        # update current iteration
        self.current_iter += 1

        # consider termination
        if self.current_iter == self.n_iter:
            self.stop_training = True

    def on_epoch_end(self, epoch, logs={}):
        self.log_ep.setdefault('epoch', []).append(epoch)
        self.log_ep.setdefault('lr', []).append(
            K.get_value(self.model.optimizer.lr))

        for k, v in logs.items():
            self.log_ep.setdefault(k, []).append(v)

    def test_run(self, n_iter=None):
        """
        Visualize values of learning rate (and momentum) as a function of iteration (batch).

        :param n_iter: a number of cycles. If None, 1000 is used.
        """

        if hasattr(self, 'current_iter'):
            original_it = self.current_iter

        if n_iter is None:
            if hasattr(self, 'n_iter'):
                n_iter = self.n_iter
            else:
                n_iter = 1000
        n_iter = int(n_iter)

        lrs = np.zeros(shape=(n_iter,))
        if self.momentum_range is not None:
            moms = np.zeros_like(lrs)

        for i in range(int(n_iter)):
            self.current_iter = i
            lrs[i] = self.get_current_lr(n_iter)
            if self.cycle_momentum:
                moms[i] = self.get_current_momentum(n_iter)
        if not self.cycle_momentum:
            plt.plot(lrs)
            plt.xlabel('iterations')
            plt.ylabel('lr')
        else:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            plt.plot(lrs)
            plt.xlabel('iterations')
            plt.ylabel('lr')
            plt.subplot(1, 2, 2)
            plt.plot(moms)
            plt.xlabel('iterations')
            plt.ylabel('momentum')

        if hasattr(self, 'current_iter'):
            self.current_iter = original_it

