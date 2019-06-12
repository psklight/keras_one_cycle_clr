try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from .utils import set_lr, set_momentum


class CLR(keras.callbacks.Callback):
    """
    Based off https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/callbacks/cyclical_learning_rate.py.

    :param cyc: an integer for a number of cycles.
    :param lr_range: a tuple of starting (usually minimum) lr value and maximum (peak) lr value.
    :param momentum_range: a tuple of momentum values.
    :param phase_one_fraction: a fraction for phase I (increasing lr) in one cycle. Must between 0 to 1.
    :param amplitude_fn: a function to modify a cycle's amplitude (lr_max-lr_min). For example, amplitude_fn = lambda x: np.power(0.5, x) will halven the amplitude every cycle.
    :param reset_on_train_begin: True or False to reset counters when training begins.
    :param record_frq: integer > 0, a frequency in batches to record training loss.
    :param verbose: True or False to print progress.
    """

    def __init__(
            self,
            cyc,
            lr_range,
            momentum_range,
            phase_one_fraction=0.3,
            amplitude_fn=None,
            reset_on_train_begin=True,
            record_frq=10,
            verbose=False):

        super(CLR, self).__init__()

        self.cyc = cyc
        self.lr_range = lr_range
        self.momentum_range = momentum_range
        self.phase_one_fraction = phase_one_fraction

        if amplitude_fn is None:
            amplitude_fn = lambda x: 1.0
        if not callable(amplitude_fn):
            raise TypeError("``amplitude_fn`` must a be a function/callable object returning numeric.")
        self.amplitude_fn = amplitude_fn

        self.reset_on_train_begin = reset_on_train_begin
        self.record_frq = record_frq
        self.verbose = verbose

        # helper tracker
        self.current_iter = 0
        self.log = {}  # history in iterations
        self.log_ep = {}  # history in epochs
        self.stop_training = False

        self.reset()

    def reset(self):
        self.current_iter = 0

    def get_current_lr(self, n_iter=None):
        """
        A helper function to calculate a current learning rate based on current iteration number.

        :return lr: a current learning rate.
        """
        if n_iter is None:
            n_iter = self.n_iter

        x = float(self.current_iter) / n_iter
        # current cycle
        c = np.floor(x * self.cyc)
        # current cycle phase
        x = (x - c / self.cyc) * self.cyc

        amp_0 = self.lr_range[1] - self.lr_range[0]
        amp_mod = self.amplitude_fn(c)
        if x < self.phase_one_fraction:
            amp = amp_0 * amp_mod
            lr = (np.cos(x * np.pi / self.phase_one_fraction - np.pi) + 1) * amp / 2.0 \
                 + self.lr_range[0]
        if x >= self.phase_one_fraction:
            amp = amp_0 * amp_mod + self.lr_range[0]
            lr = (np.cos((x - self.phase_one_fraction) * np.pi / (1 - self.phase_one_fraction)) + 1) / 2.0 * amp
        return lr

    def get_current_momentum(self, n_iter=None):
        """
        A helper function to calculate a current momentum based on current iteration number.

        :return momentum: a current momentum.
        """
        if n_iter is None:
            n_iter = self.n_iter
        amp = self.momentum_range[1] - self.momentum_range[0]

        x = float(self.current_iter) / n_iter
        # current cycle
        c = np.floor(x * self.cyc)
        # current cycle phase
        x = (x - c / self.cyc) * self.cyc

        if x < self.phase_one_fraction:
            delta = (np.cos(x * np.pi / self.phase_one_fraction - np.pi) + 1) * amp / 2.0
        if x >= self.phase_one_fraction:
            delta = (np.cos((x - self.phase_one_fraction) * np.pi / (1 - self.phase_one_fraction)) + 1) / 2.0 * amp
        return delta + self.momentum_range[0]

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

        if self.reset_on_train_begin:
            self.reset

    def on_train_batch_begin(self, batch, logs={}):
        set_lr(self.model.optimizer, self.get_current_lr())
        if self.cycle_momentum:
            set_momentum(self.model.optimizer, self.get_current_momentum())

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

            self.log.setdefault('iter', []).append(self.current_iter)

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

    def find_n_epoch(self, dataset, batch_size=None):
        """
        A method to find a number of epochs to train in the sweep.

        :param dataset: If the training data is an ndarray (used with model.fit), ``dataset`` is the x_train. If the training data is a generator (used with model.fit_generator), ``dataset`` is the generator instance.
        :param batch_size: Needed only if ``dataset`` is x_train.
        :return epochs: a number of epochs needed to do a learning rate sweep.
        """
        if isinstance(dataset, keras.utils.Sequence):
            return int(np.ceil(self.step_size * 2.0 / len(dataset) * self.terminate_on_cycle))
        if isinstance(dataset, np.ndarray):
            if batch_size is None:
                raise KeyError("``batch_size`` must be provided.")
            else:
                return int(np.ceil(dataset.shape[0] / batch_size))

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

