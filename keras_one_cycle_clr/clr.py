import tensorflow.keras as keras
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

class LrRangeTest(keras.callbacks.Callback):
    """
    A callback class for finding a learning rate.
    """

    def __init__(self,
                 lr_range = (1e-4, 10),
                 wd_list = [],
                 steps=100,
                 batches_per_step=10,
                 threshold_multiplier=5,
                 validation_data=None,
                 validation_batch_size=16,
                 batches_per_val = 10,
                 verbose=False):
        
        super(LrRangeTest, self).__init__()
        
        self.lr_range = lr_range
        
        self.wd_list = wd_list
            
        self.steps = steps
        self.batches_per_step = batches_per_step
        self.early_stop = False
        self.threshold_multiplier = threshold_multiplier
        self.validation_data = validation_data
        if validation_data is not None:
            self.use_validation = True
        else:
            self.use_validation = False
        self.validation_batch_size = validation_batch_size
        self.batches_per_val = batches_per_val
        self.verbose = verbose
        
        # generate a range of learning rates
        self.lr_values = np.power(10.0, 
                                  np.linspace(np.log10(lr_range[0]), np.log10(lr_range[1]), self.steps))
        
        # logs initialization
        self.lr = self.lr_values
        n_wd = len(self.wd_list) if len(self.wd_list)>0 else 1
        self.loss = np.zeros(shape=(self.lr_values.size, n_wd)) * np.nan
        if self.use_validation:
            self.val_loss = np.zeros_like(self.loss) * np.nan
            
        # non-reset counters
        self.current_wd = 0
        
    def _fetch_val_batch(self, batch):
        if isinstance(self.validation_data, (tuple,)):
            batch_size = self.validation_batch_size
            x = self.validation_data[0][batch*batch_size:(batch+1)*batch_size]
            y = self.validation_data[1][batch*batch_size:(batch+1)*batch_size]
            return (x, y)
        if isinstance(self.validation_data, (keras.utils.Sequence,)):
            return self.validation_data.__getitem__(batch)
        
    def _reset(self):
        """
        Reset counters, prepare for a new weight decay value.
        """
        self.model.optimizer.set_weights(self.model_org.optimizer.get_weights())
        self.model.set_weights(self.model_org.get_weights())
        self.current_step = 0
        self.current_batches_per_step = 0
        self.current_loss_val = 0
        self.best_loss = np.inf
        self.early_stop = False

    def on_train_begin(self, logs={}):
        # save current model for reset
        self.model.save("lr_range_test_original_stage.h5")
        self.model_org = keras.models.load_model("lr_range_test_original_stage.h5")
        # handle empty input wd_list
        if len(self.wd_list)==0:
            self.wd_list = [K.get_value(self.model.optimizer.decay)]
        self.current_wd = 0
        self._reset()

    def on_train_batch_begin(self, batch, logs):
        K.set_value(self.model.optimizer.lr, self.lr_values[self.current_step])
        K.set_value(self.model.optimizer.decay, self.wd_list[self.current_wd])

    def on_train_batch_end(self, batch, logs):
        
        self.current_loss_val += logs['loss']
        self.current_batches_per_step += 1

        if self.current_batches_per_step == self.batches_per_step:
            
            self.loss[self.current_step, self.current_wd] = self.current_loss_val / self.batches_per_step
            
            if self.use_validation:
                # calulate for validation set
                self.current_loss_val = 0.0
                if isinstance(self.validation_data, tuple):
                    batch_size = self.validation_batch_size
                    N = int(np.ceil(self.validation_data[0].shape[0]/batch_size))
                if isinstance(self.validation_data, keras.utils.Sequence):
                    N = len(self.validation_data)
                n_batch = min(self.batches_per_val, N)
                for i in range(n_batch):
                    data_batch = self._fetch_val_batch(i)
                    batch_size = data_batch[0].shape[0]
                    result = self.model.evaluate(x=data_batch[0], y=data_batch[1],
                                                 batch_size=batch_size,
                                                 verbose=False)
                    self.current_loss_val += result[0]
                    
                self.val_loss[self.current_step, self.current_wd] = self.current_loss_val/n_batch

            # verbose
            if self.verbose:
                if not self.use_validation:
                    print("wd={:.2e}".format(self.wd_list[self.current_wd]), ",",
                      "lr={:.2e}".format(self.lr_values[self.current_step]), ",",
                      "loss={:.2e}".format(self.loss[self.current_step - 1, self.current_wd]))
                if self.use_validation:
                    print("wd={:.2e}".format(self.wd_list[self.current_wd]), ",",
                      "lr={:.2e}".format(self.lr_values[self.current_step]), ",",
                      "loss={:.2e}".format(self.loss[self.current_step - 1, self.current_wd]), ",",
                      "val_loss={:.2e}".format(self.val_loss[self.current_step-1, self.current_wd]))

            self.current_batches_per_step = 0
            self.current_loss_val = 0.0
            self.current_step += 1
            
            # update best loss
            if not self.use_validation:
                latest_loss = self.loss[self.current_step-1, self.current_wd]
            else:
                latest_loss = self.val_loss[self.current_step-1, self.current_wd]
                
            self.best_loss = self.best_loss if self.best_loss < latest_loss else latest_loss
            
            # determine earlystop
            if latest_loss > self.best_loss * self.threshold_multiplier:
                self.early_stop = True

        # consider next wd value
        if self.current_step == self.lr_values.size or self.early_stop:
            self.current_wd += 1
            self._reset()
        
        # stop training when done with all weight decays, set everything back to before lr range test.
        if self.current_wd == len(list(self.wd_list)):
            self.model.set_weights(self.model_org.get_weights())
            K.set_value(self.model.optimizer.lr, 
                        K.get_value(self.model_org.optimizer.lr) )
            self.model.optimizer.set_weights(self.model_org.optimizer.get_weights())
            self.model.stop_training = True

    def find_n_epoch(self, dataset, batch_size=None):
        """
        A method to find a number of epochs to train in the sweep.

        :param dataset: If the training data is an ndarray (used with model.fit), ``dataset`` is the x_train. If the training data is a generator (used with model.fit_generator), ``dataset`` is the generator instance.
        :param batch_size: Needed only if ``dataset`` is x_train.
        :return epochs: a number of epochs needed to do a learning rate sweep.
        """
        n_wd = len(self.wd_list) if len(self.wd_list)>0 else 1
        if isinstance(dataset, keras.utils.Sequence):
            return int(np.ceil(self.steps * self.batches_per_step / len(dataset)) * n_wd)
        if isinstance(dataset, np.ndarray):
            if batch_size is None:
                raise ValueError("``batch_size`` must be provided.")
            else:
                return int(np.ceil(self.steps * self.batches_per_step /
                                   (dataset.shape[0] / batch_size)) * n_wd)

    def plot(self, **kwargs):
        
        y_scale = kwargs.setdefault('y_scale', "linear")
        x_scale = kwargs.setdefault('x_scale', "log")
        linestyle = kwargs.setdefault('linestle', '.-')
        
        plt.figure()
        if self.use_validation:
            loss = self.val_loss
            y_str = "val loss"
        else:
            loss = self.loss
            y_str = "train loss"
        # build legend
        legends = []
        for w in self.wd_list:
            legends.append("wd={:.1e}".format(w))
        lr = self.lr
        
        plt.plot(lr, loss, linestyle)
        
        plt.xlabel("lr")
        plt.ylabel(y_str)
        
        plt.xscale(x_scale)
        plt.yscale(y_scale)
        
        plt.legend(tuple(legends))
        plt.show()


class OneCycle(keras.callbacks.Callback):
    
    def __init__(
            self,
            lr_range,
            momentum_range = None,
            reset_on_train_begin=True,
            record_frq=10,
            verbose=False):

        super(OneCycle, self).__init__()

        self.lr_range = lr_range
        
        self.momentum_range = momentum_range
        if momentum_range is not None:
            err_msg = "momentum_range must be a 2-numericy tuple (m1, m2)."
            if not isinstance(momentum_range, (tuple,)) or len(momentum_range)!=2:
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
            
        x = float(self.current_iter)/n_iter
        # first half
        if x<0.5:
            amp = self.lr_range[1] - self.lr_range[0]
            lr = (np.cos(x*np.pi*2 - np.pi)+1)*amp/2.0 + self.lr_range[0]
        if x>=0.5:
            amp = self.lr_range[1]
            lr = (np.cos((x-0.5)*np.pi*2)+1)/2.0*amp
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
        if self.params['batch_size'] is not None: #model.fit
            self.n_bpe = int(np.ceil(self.params['samples']/self.params['batch_size']))
        if self.params['batch_size'] is None: #model.fit_generator
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
        :param cyc: a number of cycles
        :return lrs: learning rates
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
            plt.figure(figsize=(10,4))
            plt.subplot(1,2,1)
            plt.plot(lrs)
            plt.xlabel('iterations')
            plt.ylabel('lr')
            plt.subplot(1,2,2)
            plt.plot(moms)
            plt.xlabel('iterations')
            plt.ylabel('momentum')
        
        if hasattr(self, 'current_iter'):
            self.current_iter = original_it


def plot_from_history(history):
    """
    Plot losses in training history.

    :param history: a ``History`` callback instance from ``Model`` instance.
    """

    epoch = history.epoch
    val_exist = "val_loss" in history.history

    plt.plot(epoch, history.history["loss"], '.-', label="train")
    if val_exist:
        plt.plot(epoch, history.history["val_loss"], '.-', label="valid")

    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.legend()