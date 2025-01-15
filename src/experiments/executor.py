"""
This module handles the training, validation and testing of Pytorch models for
timeseries prediction.
The classes perform either batch learning or seq2seq learning.
If cuda is available the procedures are executed on GPU.
It requires a data loader, the model, training and data parameters
and a dictionary with all datasets the model shall be executed on.
---
The training can be done with or without teacher forcing.
Trains the model using the provided data and training parameters.
Logging with MLFlow.
Using teacher forcing the model is shown the ground truth during training.
That accelerates the convergence, however, the model has a high risk of
suffering an exposure bias (i.e. during testing or inference the model
lacks this input and might have difficulties relying on its predictions).
One should not use teacher forcing for testing!
"""
import time
import ast

import inspect
from typing import Callable

import torch
import numpy as np

from src.data_loader.wavemask.aug import augmentation
from src.util.train_utils import EarlyStopping


class LearningProcessor:
    """
    Handles Training and Testing of Pytorch Model with Batch or seq2seq.
    """

    def __init__(self, provide_series_method: Callable, model: torch.nn.Module,
                 params, data_dict: dict,
                 teacher_force: bool = False) -> None:
        """
        Arguments:
            provide_series_method: A callable function to provide data.
            model: A Pytorch model instance.
            params: Data and path configuration.Training parameters including
            a desired model name.
        """
        self.provide_series = provide_series_method
        self.model = model
        self.model_trained = None
        self.params = params
        self.data_dict = data_dict
        self.use_teacher_force = teacher_force
        self.variate_dim = -1 if self.params.features == 'MS' else 0
        # set device
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Optimizer initialization
        if params.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(model.parameters(),
                                             lr=self.params.learning_rate)
        elif params.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(model.parameters(),
                                               lr=self.params.learning_rate)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(),
                                              lr=self.params.learning_rate)
        # Loss function initialization
        if self.params.loss == "MAE":
            self.loss_criterion = torch.nn.L1Loss()
        elif self.params.loss == "Smooth":
            self.loss_criterion = torch.nn.SmoothL1Loss()
        else:
            self.loss_criterion = torch.nn.MSELoss()

    @staticmethod
    def inspect_model_args(func):
        sig = inspect.signature(func)
        required_args = [name for name, param in sig.parameters.items()
                         if param.default is inspect.Parameter.empty and
                         param.kind in [
                             inspect.Parameter.POSITIONAL_OR_KEYWORD,
                             inspect.Parameter.POSITIONAL_ONLY]]
        return required_args

    def get_training_data(self):
        training_set, training_loader = self.provide_series(
            params=self.params, flag="train", data_dict=self.data_dict)
        print("Train data loading completed!\n Data Set: ", training_set)
        validation_set, validation_loader = self.provide_series(
            params=self.params, flag="val", data_dict=self.data_dict)
        print("Validation data loading completed!\n Data Set: ",
              validation_set)

        return training_set, training_loader, validation_set, validation_loader

    def set_training_args(self):
        schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min',
            patience=self.params.scheduler_patience,
            factor=self.params.scheduler_factor)
        stop = EarlyStopping(patience=self.params.stop_patience,
                             delta=self.params.stop_delta, mode='min')
        return schedule, stop

    def process_batch(self, x, y=None, mark_x=None, mark_y=None,
                      testing=False):
        # with teacher forcing
        if self.use_teacher_force and not testing:
            decoder_batch = torch.zeros_like(y)
            decoder_batch[:, 1:, :] = y[:, :-1, :]
            # Assign batch_x's last timestep (for all features)
            # to the first timestep of decoder_batch
            decoder_batch[:, 0, :] = x[:, -1, :]
            model_out = self.model(x, decoder_batch, False)
        elif self.use_teacher_force and testing:
            model_out = self.model(x, x[:, -1, :], True)
        # without teacher forcing
        else:
            model_num_args = len(self.inspect_model_args(self.model.forward))
            if model_num_args == 1:
                model_out = self.model(x)
            elif model_num_args == 2:
                model_out = self.model(x, mark_x)
            elif model_num_args == 4:
                model_out = self.model(x, mark_x, y, mark_y)
            else:
                print(f"Number of model arguments is {model_num_args}, "
                      f"required are:  "
                      f"{self.inspect_model_args(self.model.forward)}")
                model_out = None
        return model_out

    def train(self):
        """
        Train the model using the provided data and training parameters.
        Logging with MLFlow.
        Early stopping and learning rate scheduling is performed.
        After every training epoch, the model is validated on an unseen
        validation set. The validation loss is used to
        define early stopping and learning rate adjustments.
        Would be nice to have a function that is able to deal with order
        switching if the arguments!
        """
        # ---LOAD DATA---
        _, train_loader, _, val_loader = self.get_training_data()

        # ---TRAINING SETTINGS---
        scheduler, early_stopping = self.set_training_args()

        # ---RUN EPOCHS---
        avg_train_loss = None
        start_time = time.time()
        print(f"\n{'=' * 46}\nTraining of {self.model.__class__.__name__} "
              f"successfully started.\n{'=' * 46}\n")
        if self.params.aug_type in {1, 2, 3, 4, 5}:
            print(f"Augmentation type is {self.params.aug_type}.")
        for epoch in range(self.params.epochs):
            train_loss, val_loss = [], []
            epoch_time = time.time()

            # ---TRAINING---
            self.model.train()
            for _, (train_batch_x, train_batch_y, train_mark_x, train_mark_y) \
                    in enumerate(train_loader):
                # avoid gradient accumulation
                self.optimizer.zero_grad()
                train_batch_x, train_batch_y = map(lambda x: x.float().to(
                    self.device), [train_batch_x, train_batch_y])
                # Separate training for WaveMask augmentation
                if self.params.aug_type in {1, 2, 3, 4, 5}:
                    outputs, train_batch_x, train_batch_y = (
                        self.train_wavemask(train_batch_x, train_batch_y,
                                            train_mark_x))
                else:
                    outputs = self.process_batch(train_batch_x, train_batch_y,
                                                 train_mark_x, train_mark_y)
                if outputs is None:
                    break
                outputs, train_batch_y = [x[:, -self.params.pred_len:,
                                          self.variate_dim:].to(self.device)
                                          for x in [outputs, train_batch_y]]
                if torch.isnan(outputs).any():
                    print(f"Outputs NaN: {torch.isnan(outputs).any()}")
                    continue
                # ---CALC LOSS---
                loss = self.loss_criterion(outputs, train_batch_y)
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            # ---VALIDATION---
            self.model.eval()
            with torch.no_grad():
                for _, (val_batch_x, val_batch_y,
                        val_mark_x, val_mark_y) in enumerate(val_loader):
                    val_batch_x, val_batch_y = map(
                        lambda x: x.float().to(self.device),
                        [val_batch_x, val_batch_y])
                    val_outputs = self.process_batch(val_batch_x, val_batch_y,
                                                     val_mark_x, val_mark_y)
                    val_outputs, val_batch_y = [x[:, -self.params.pred_len:,
                                                self.variate_dim:].to(
                        self.device) for x in [val_outputs, val_batch_y]]
                    val_loss_item = self.loss_criterion(val_outputs,
                                                        val_batch_y).item()
                    val_loss.append(val_loss_item)
            avg_val_loss = np.average(val_loss)

            # ---EARLY STOPPING---
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("No improvement-training was stopped early.")
                break

            # ---SCHEDULER STEP---
            lr_before = self.optimizer.param_groups[0]['lr']
            scheduler.step(avg_val_loss)
            lr_after = self.optimizer.param_groups[0]['lr']
            if lr_before != lr_after:
                print(f"Learning rate changed from {lr_before} to {lr_after}")

            # ---EPOCH SUMMARY---
            print(f"Epoch: {epoch + 1} | Time: {time.time() - epoch_time}")
            avg_train_loss = np.average(train_loss)
            print("Train loss was: ", avg_train_loss)

        # ---TRAINING SUMMARY---
        total_time = time.time() - start_time
        print(f"Total Time: {total_time} | Final aver. loss: {avg_train_loss}")

        # --- SET TRAINED MODEL---
        self.model_trained = self.model

    def test(self):
        """
        Test the trained model using test data. Logging with MLFlow.
        :return pred_values: is a list of numpy arrays with predicted sequences
        :return true_values: contains corresponding ground truth
        :return marks_x, marks_y: timestamps of the input and output sequences,
        resp.
        """
        if self.model_trained is None:
            raise ValueError("The model is in an invalid state.")

        # ---LOAD DATA---
        test_set, test_loader = self.provide_series(
            params=self.params, flag="test", data_dict=self.data_dict)
        print("Test data loading completed!\n Data Set: ", test_set)

        # --- TEST INIT---
        pred_values, true_values, marks_x, marks_y = [], [], [], []
        self.model = self.model_trained
        print("Model testing successfully started")

        # ---TEST LOOP---
        self.model.eval()
        with ((torch.no_grad())):
            for test_batch_x, test_batch_y, test_mark_x, test_mark_y in test_loader:
                test_batch_x, test_batch_y = map(lambda x: x.float().to(
                    self.device), [test_batch_x, test_batch_y])
                self.model.to(self.device)
                outputs = self.process_batch(test_batch_x, test_batch_y,
                                             test_mark_x, test_mark_y, True)
                outputs, test_batch_y = [x[:, -self.params.pred_len:,
                                         self.variate_dim:]
                                         for x in [outputs, test_batch_y]]

                # ---CREATE OUTPUT (PREDICTION&TRUTH)---
                # detach tensor from computational graph and move it to cpu
                pred_values += [outputs.detach().cpu().numpy()]
                true_values += [test_batch_y.detach().cpu().numpy()]
                marks_x += [test_mark_x.numpy()]
                marks_y += [test_mark_y.numpy()]
        return pred_values, true_values, marks_x, marks_y

    def augment_and_sample(self, xy, batch_x, batch_y, sampling_rate):
        batch_x2 = xy[:, :self.params.seq_len, :]
        batch_y2 = xy[:, -self.params.seq_overlap - self.params.pred_len:, :]
        sampling_steps = int(batch_x2.shape[0] * sampling_rate)
        indices = torch.randperm(batch_x2.shape[0])[:sampling_steps]
        batch_x2 = batch_x2[indices]
        batch_y2 = batch_y2[indices]
        batch_x = torch.cat([batch_x, batch_x2], dim=0)
        batch_y = torch.cat([batch_y, batch_y2], dim=0)
        return batch_x, batch_y

    def concat_aug(self, x, y, xy):
        x2, y2 = (
            xy[:, :self.params.seq_len, :],
            xy[:, -self.params.seq_overlap - self.params.pred_len:, :])
        x = torch.cat([x, x2], dim=0)
        y = torch.cat([y, y2], dim=0)
        return x, y

    def train_wavemask(self, batch_x, batch_y, aug_data):
        if isinstance(self.params.rates, str):
            self.params.rates = ast.literal_eval(self.params.rates)

        def put2device(x, y):
            return x.float().to(self.device), y.float().to(self.device)

        if self.params.aug_type == 5:
            aug_data = aug_data.float().to(self.device)
        else:
            aug_data = None
        if self.params.aug_type:
            batch_x, batch_y = put2device(batch_x, batch_y)
            aug = augmentation()
            if self.params.aug_type == 1:
                xy = aug.freq_mask(
                    batch_x, batch_y[:, -self.params.pred_len:, :],
                    rate=self.params.aug_rate, dim=1)
                batch_x, batch_y = self.concat_aug(batch_x, batch_y, xy)
            elif self.params.aug_type == 2:
                xy = aug.freq_mix(
                    batch_x, batch_y[:, -self.params.pred_len:, :],
                    rate=self.params.aug_rate, dim=1)
                batch_x, batch_y = self.concat_aug(batch_x, batch_y, xy)
            elif self.params.aug_type == 3:
                xy = aug.wave_mask(batch_x,
                                   batch_y[:, -self.params.pred_len:, :],
                                   rates=self.params.rates,
                                   wavelet=self.params.wavelet,
                                   level=self.params.level, dim=1)
                batch_x, batch_y = self.augment_and_sample(
                    xy, batch_x, batch_y, self.params.sampling_rate)
            elif self.params.aug_type == 4:
                xy = aug.wave_mix(
                    batch_x, batch_y[:, -self.params.pred_len:, :],
                    rates=self.params.rates, wavelet=self.params.wavelet,
                    level=self.params.level, dim=1)
                batch_x, batch_y = self.augment_and_sample(
                    xy, batch_x, batch_y, self.params.sampling_rate)
            elif self.params.aug_type == 5:
                weighted_xy = aug.emd_aug(aug_data)
                weighted_x, weighted_y = (
                    weighted_xy[:, :self.params.seq_len, :],
                    weighted_xy[:,
                    -self.params.seq_overlap - self.params.pred_len:, :])
                batch_x, batch_y = aug.mix_aug(
                    weighted_x, weighted_y, lambd=self.params.aug_rate)

        batch_x, batch_y = put2device(batch_x, batch_y)
        return self.model(batch_x), batch_x, batch_y

    def train_cycle(self):
        """
        Training routine for CycleNet.
        """
        # ---LOAD DATA---
        _, train_loader, _, val_loader = self.get_training_data()

        # ---TRAINING SETTINGS---
        scheduler, early_stopping = self.set_training_args()

        # ---RUN EPOCHS---
        avg_train_loss = None
        start_time = time.time()
        print(f"\n{'=' * 46}\nTraining of {self.model.__class__.__name__} "
              f"successfully started.\n{'=' * 46}\n")
        for epoch in range(self.params.epochs):
            train_loss, val_loss = [], []
            epoch_time = time.time()

            # ---TRAINING---
            self.model.train()
            for _, (train_batch_x, train_batch_y, train_mark_x, train_mark_y,
                    train_batch_cycle) in enumerate(train_loader):
                self.optimizer.zero_grad()  # avoid gradient accumulation
                train_batch_x, train_batch_y = map(lambda x: x.float().to(
                    self.device), [train_batch_x, train_batch_y])
                train_batch_cycle = train_batch_cycle.int().to(self.device)
                # decoder input
                dec_inp = torch.zeros_like(
                    train_batch_y[:, -self.params.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [train_batch_y[:, :self.params.seq_overlap, :],
                     dec_inp], dim=1).float().to(self.device)

                if any(substr in self.model.__class__.__name__ for substr in
                       {'CycleNet'}):
                    outputs = self.model(train_batch_x, train_batch_cycle)
                else:
                    raise ValueError(f"Unsupported model class: "
                                     f"{self.model.__class__.__name__}")
                if outputs is None:
                    # stop training of model could not be applied
                    break
                outputs, train_batch_y = [x[:, -self.params.pred_len:,
                                          self.variate_dim:].to(self.device)
                                          for x in [outputs, train_batch_y]]
                if torch.isnan(outputs).any():
                    print(f"Outputs NaN: {torch.isnan(outputs).any()}")
                    continue
                # ---CALC LOSS---
                loss = self.loss_criterion(outputs, train_batch_y)
                train_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()

            # ---VALIDATION---
            self.model.eval()
            with torch.no_grad():
                for _, (val_batch_x, val_batch_y, val_mark_x, val_mark_y,
                        val_batch_cycle) in enumerate(val_loader):
                    val_batch_x, val_batch_y = map(
                        lambda x: x.float().to(self.device),
                        [val_batch_x, val_batch_y])

                    val_batch_cycle = val_batch_cycle.int().to(self.device)
                    # decoder input
                    dec_inp = torch.zeros_like(
                        val_batch_y[:, -self.params.pred_len:, :]).float()
                    dec_inp = torch.cat(
                        [val_batch_y[:, :self.params.seq_overlap, :],
                         dec_inp], dim=1).float().to(self.device)

                    if any(substr in self.model.__class__.__name__ for substr
                           in {'CycleNet'}):
                        val_outputs = self.model(val_batch_x, val_batch_cycle)

                    val_outputs, val_batch_y = [x[:, -self.params.pred_len:,
                                                self.variate_dim:].to(
                        self.device) for x in [val_outputs, val_batch_y]]
                    val_loss_item = self.loss_criterion(val_outputs,
                                                        val_batch_y).item()
                    val_loss.append(val_loss_item)
            avg_val_loss = np.average(val_loss)

            # ---EARLY STOPPING---
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("No improvement-training was stopped early.")
                break

            # ---SCHEDULER STEP---
            lr_before = self.optimizer.param_groups[0]['lr']
            scheduler.step(avg_val_loss)
            lr_after = self.optimizer.param_groups[0]['lr']
            if lr_before != lr_after:
                print(f"Learning rate changed from {lr_before} to {lr_after}")

            # ---EPOCH SUMMARY---
            print(f"Epoch: {epoch + 1} | Time: {time.time() - epoch_time}")
            avg_train_loss = np.average(train_loss)
            print("Train loss was: ", avg_train_loss)

        # ---TRAINING SUMMARY---
        total_time = time.time() - start_time
        print(f"Total Time: {total_time} | Final aver. loss: {avg_train_loss}")

        # --- SET TRAINED MODEL---
        self.model_trained = self.model

    def test_cycle(self):
        """
        Test the trained model using test data. Logging with MLFlow.
        :return pred_values: is a list of numpy arrays with predicted sequences
        :return true_values: contains corresponding ground truth
        :return marks_x, marks_y: timestamps of the input and output sequences,
        resp.
        """
        if self.model_trained is None:
            raise ValueError("The model is in an invalid state.")

        # ---LOAD DATA---
        test_set, test_loader = self.provide_series(
            params=self.params, flag="test", data_dict=self.data_dict)
        print("Test data loading completed!\n Data Set: ", test_set)

        # --- TEST INIT---
        pred_values, true_values, marks_x, marks_y = [], [], [], []
        self.model = self.model_trained
        print("Model testing successfully started")

        # ---TEST LOOP---
        self.model.eval()
        with torch.no_grad():
            for (test_batch_x, test_batch_y, test_mark_x, test_mark_y,
                 test_batch_cycle) in test_loader:
                test_batch_x, test_batch_y = map(lambda x: x.float().to(
                    self.device), [test_batch_x, test_batch_y])
                test_batch_cycle = test_batch_cycle.int().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(
                    test_batch_y[:, -self.params.pred_len:, :]).float()
                dec_inp = torch.cat(
                    [test_batch_y[:, :self.params.seq_overlap, :], dec_inp],
                    dim=1).float().to(self.device)

                self.model.to(self.device)
                if any(substr in self.model.__class__.__name__ for substr
                       in {'CycleNet'}):
                    outputs = self.model(test_batch_x, test_batch_cycle)

                outputs, test_batch_y = [x[:, -self.params.pred_len:,
                                         self.variate_dim:]
                                         for x in [outputs, test_batch_y]]

                # ---CREATE OUTPUT (PREDICTION&TRUTH)---
                # detach tensor from computational graph and move it to cpu
                pred_values += [outputs.detach().cpu().numpy()]
                true_values += [test_batch_y.detach().cpu().numpy()]
                marks_x += [test_mark_x.numpy()]
                marks_y += [test_mark_y.numpy()]
        return pred_values, true_values, marks_x, marks_y
