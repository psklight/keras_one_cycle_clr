try:
    import tensorflow.keras as keras
    import tensorflow.keras.backend as K
except:
    import keras
    import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt


class CLR(keras.callbacks.Callback):
    """
    Based off https://github.com/keras-team/keras-contrib/blob/master/keras_contrib/callbacks/cyclical_learning_rate.py.

    :param base_lr: initial learning rate which is the lower boundary in the cycle.
    :param max_lr: upper boundary in the cycle. Together with ``base_lr`` it defines the initial amplitude (``max_lr``-``min_lr``) of a cycle. However, the actual amplitude can be modified using a specific ``amplitude_fn``.
    :param step_size: the number of batch training step of a half cycle.
    :param scale: either "log" (default) or "linear", determining whether a step in learning rate is uniform in a linear or a log scale.
    :param amplitude_fn: a function that can alter local amplitude. It must be in a form of ``f(n)`` and returns a number. For example, a constant amplitude function can be ``lambda x: 1`` (default). An exponential decay can be ``lambda x: 1/2**x``.
    :param amplitude_fn_mode: can only be 'cycle' or 'iteration'. This provides a meaning of ``n`` in ``amplitude_fn``. ``iteration`` means a batch training.
    :param terminate_on_cycle: a number of cycles to perform CLR training.
    :param reset_on_train_begin: True or False whether to reset CLR when training starts.
    :param record_frq: a number of iterations (batches) for performances/metrics to be recorded in ``history`` attribute.
    :param verbose: True or False whether a progress is printed.
    """

    def __init__(
            self,
            cyc,
            lr_range,
            mom_range,
            scale="log",
            amplitude_fn=None,
            amplitude_fn_mode='cycle',
            reset_on_train_begin=True,
            record_frq=10,
            verbose=False):

        super(CLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size

        if scale not in ["log", "linear"]:
            raise KeyError("``scale`` must be either ""log"" or ""linear"".")
        self.scale = scale

        if amplitude_fn is None:
            amplitude_fn = lambda x: 1.0
        if not callable(amplitude_fn):
            raise TypeError("``amplitude_fn`` must a be a function/callable object returning numeric.")
        self.amplitude_fn = amplitude_fn

        if amplitude_fn_mode not in ["cycle", "iteration"]:
            raise KeyError("``amplitude_fn_mode`` must be either ""cycle"" or ""iteration"".")
        self.amplitude_fn_mode = amplitude_fn_mode

        self.terminate_on_cycle = terminate_on_cycle
        self.reset_on_train_begin = reset_on_train_begin
        self.record_frq = record_frq
        self.verbose = verbose

        # helper tracker
        self.clr_iterations = 0.
        self.history = {}  # history in iterations
        self.history_epoch = {}  # history in epochs

        self.reset()

    def reset(self, new_base_lr=None, new_max_lr=None, new_step_size=None):
        """
        Resets cycle iterations. Optional boundary/step size adjustment.
        :param new_base_lr: a new base_lr
        :param new_max_lr: a new max_lr
        :param new_step_size: a new step_sze
        """
        if new_base_lr is not None:
            self.base_lr = new_base_lr
        if new_max_lr is not None:
            self.max_lr = new_max_lr
        if new_step_size is not None:
            self.step_size = new_step_size

        self.clr_iterations = 0.
        self.history = {}

    def clr(self):
        """
        A helper function to calculate a current learning rate based on current iteration number.

        :return lr: a current learning rate.
        """
        if self.scale == "log":
            amplitude = np.log10(self.max_lr) - np.log10(self.base_lr)
        if self.scale == "linear":
            amplitude = self.max_lr - self.base_lr

        period = 2 * self.step_size

        cycle = np.floor(self.clr_iterations / period)

        if self.amplitude_fn_mode == "cycle":
            amp_mod = self.amplitude_fn(cycle)
        if self.amplitude_fn_mode == "iteration":
            amp_mod = self.amplitude_fn(self.clr_iterations)

        dlr = (1 - np.abs(np.mod(self.clr_iterations, period) * 2.0 / period - 1)) * amplitude * amp_mod

        if self.scale == "log":
            return np.power(10.0, dlr + np.log10(self.base_lr))
        if self.scale == "linear":
            return dlr + self.base_lr

    def on_train_begin(self, logs={}):

        if self.reset_on_train_begin:
            self.reset()

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_begin(self, batch, logs=None):
        K.set_value(self.model.optimizer.lr, self.clr())
        if self.verbose:
            print("CLR {} iteration: lr = {}".format(int(self.clr_iterations), self.clr()))

    def on_batch_end(self, batch, logs={}):
        # record according to record_frq
        if np.mod(int(self.clr_iterations), self.record_frq) == 0:
            self.history.setdefault(
                'lr', []).append(
                K.get_value(
                    self.model.optimizer.lr))

            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

            self.history.setdefault('iter', []).append(self.clr_iterations)

        # consider termination
        if self.clr_iterations == self.terminate_on_cycle * self.step_size * 2:
            self.model.stop_training = True

        # update current iteration
        self.clr_iterations += 1

    def on_epoch_end(self, epoch, logs={}):
        self.history_epoch.setdefault('epoch', []).append(epoch)
        self.history_epoch.setdefault('lr', []).append(
            K.get_value(self.model.optimizer.lr))

        for k, v in logs.items():
            self.history_epoch.setdefault(k, []).append(v)

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

    def test_run(self, cyc=1):
        """
        :param cyc: a number of cycles
        :return lrs: learning rates
        """
        lrs = np.zeros(shape=(self.step_size*2*cyc,))
        for i in range(int(self.step_size*2*cyc)):
            self.clr_iterations = i
            lrs[i] = self.clr()
        plt.plot(lrs)
        plt.xlabel('iterations')
        plt.ylabel('lr')
        return lrs

