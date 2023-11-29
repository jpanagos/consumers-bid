import numpy as np
import scipy.linalg
import torch
import os

def load_model(model, model_path, opt, optimizer=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict_ = checkpoint["state_dict"]
    state_dict = {}

    for k in state_dict_:
        if k.startswith("module") and not k.startswith("module_list"):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    for k in state_dict:
        if k in model_state_dict:
            if (state_dict[k].shape != model_state_dict[k].shape) or (
                opt.reset_hm
                and k.startswith("hm")
                and (state_dict[k].shape[0] in [80, 1])
            ):
                if opt.reuse_hm:
                    if state_dict[k].shape[0] < state_dict[k].shape[0]:
                        model_state_dict[k][: state_dict[k].shape[0]] = state_dict[k]
                    else:
                        model_state_dict[k] = state_dict[k][
                            : model_state_dict[k].shape[0]
                        ]
                    state_dict[k] = model_state_dict[k]
                else:
                    state_dict[k] = model_state_dict[k]
    for k in model_state_dict:
        if not (k in state_dict):
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict, strict=False)

    # resume optimizer parameters
    if optimizer is not None and opt.resume:
        if "optimizer" in checkpoint:
            start_epoch = checkpoint["epoch"]
            start_lr = opt.lr
            for step in opt.lr_step:
                if start_epoch >= step:
                    start_lr *= 0.1
            for param_group in optimizer.param_groups:
                param_group["lr"] = start_lr
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model


class LSTMModule(torch.nn.Module):
    def __init__(self, lstm_hidden, lstm_layers):
        super(LSTMModule, self).__init__()
        self.num_hidden = lstm_hidden
        self.num_layers = lstm_layers
        self.lstm = torch.nn.LSTM(11, self.num_hidden, num_layers=self.num_layers)
        self.out1 = torch.nn.Linear(self.num_hidden, 64)
        self.out2 = torch.nn.Linear(64, 4 * 5)

    def forward(self, input_traj):

        input_traj = input_traj.permute(1, 0, 2)
        output, (hn, cn) = self.lstm(input_traj)
        x = self.out1(output[-1])
        x = self.out2(x)
        return x


class RecurrentNet(object):

    """
    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).
    """

    def __init__(self, opt):

        self.model = LSTMModule(opt.lstm_neurons, opt.lstm_layers)
        if opt.lstm_weights != "":
            self.model = load_model(self.model, opt.lstm_weights, opt)
            device = torch.device("cuda" if opt.devices > 0 else "cpu")
            self.model = self.model.to(device)
            self.model.eval()
        self.opt = opt
        self.MAX_dis_fut = 5

    def predict(self, h0, c0, new_features):

        new_features = new_features.permute(1, 0, 2)

        with torch.no_grad():
            output, (hn, cn) = self.model.lstm(new_features, (h0, c0))
            x = self.model.out1(output[-1])
            x = self.model.out2(x)

        x = x.view(self.MAX_dis_fut, -1).cpu().detach().numpy()
        predictions = {}
        for i in range(self.MAX_dis_fut):
            predictions[1 + i] = x[i]

        return hn, cn, predictions

    def gating_distance(
        self, mean, covariance, measurements, only_position=False, metric="maha"):

        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean

        if metric == "gaussian":
            d = measurements[:, 3:-1] - mean[3:-1]
            return np.sqrt(np.sum(d * d, axis=1))
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError("invalid distance metric")

