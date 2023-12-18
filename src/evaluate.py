""" Module contains functions for results visualization """

import argparse
import json
import os
import time
import warnings
from typing import Callable, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from cycler import cycler
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from tqdm import tqdm

### Constants

AVAILABLE_MODELS = ["OneLayerFC", "SimpleCNN", "TwoLayerFC"]
AVAILABLE_METHODS = ["SVRG", "L-SVRG", "SARAH", "PAGE"]
AVAILABLE_DATASETS = ["Mushrooms", "MNIST_binary", "MNIST"]
AVAILABLE_ITERATIONS_TYPE = ["iterations", "gradients", "time"]

### Class for Logger instance


class Logger:
    """Manage log messages"""

    def __init__(self, verbose: bool) -> None:
        self.verbose = verbose

    def log(self, message: str):
        """Log message to console

        Args:
            message (str): message to log
        """
        if self.verbose:
            print(message)


### Problem classes


class Problem:
    def reset(self):
        raise NotImplementedError

    def compute_gradient(self, data_indices: Optional[np.ndarray] = None) -> float:
        raise NotImplementedError

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_batch_indices(self, batch_size: int) -> np.ndarray:
        raise NotImplementedError

    def set_params(self, params: list[np.ndarray]):
        raise NotImplementedError

    def iterate_params_grad(self) -> list[tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError

    def iterate_params(self) -> list[np.ndarray]:
        raise NotImplementedError

    def get_data_size(self) -> int:
        raise NotImplementedError

    def zero_grads(self):
        raise NotImplementedError


class OneLayerFC(Problem):
    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x).float().to(self.device)

    def _to_numpy(self, x: torch.Tensor) -> np.ndarray:
        return x.cpu().detach().numpy()

    def _get_data(
        self, data_indices: Optional[np.ndarray] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if data_indices is None:
            return self.data

        return (self.data[0][data_indices], self.data[1][data_indices])

    def _get_result(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(F.relu(x @ self.params["W"] + self.params["B"]))

    def _get_loss(self, res: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(res, targets)

    def _get_value(self, data_indices: Optional[np.ndarray] = None) -> torch.Tensor:
        data = self._get_data(data_indices)
        for p in self.params.values():
            p.requires_grad_()

        x = data[0].to(self.device)
        y = data[1].to(self.device)

        res = self._get_result(x)

        return self._get_loss(res, y)

    def __init__(
        self,
        data: tuple[torch.Tensor, torch.Tensor],
        seed: float,
        device: torch.device,
        output_dim: int,
    ) -> None:
        self.device = device
        self.data = data
        self.seed = seed

        self.data_size: int = data[0].shape[0]
        self.possible_indices = list(range(self.data_size))
        self.input_dim = self.data[0].shape[1]
        self.output_dim = output_dim

        self.reset()

    def reset(self):
        np.random.seed(self.seed)

        self.params: dict[str, torch.Tensor] = {
            "W": self._to_tensor(np.random.rand(self.input_dim, self.output_dim)),
            "B": self._to_tensor(np.random.rand(self.output_dim)),
        }
        self.param_names = list(self.params.keys())

    def compute_gradient(self, data_indices: Optional[np.ndarray] = None) -> float:
        loss = self._get_value(data_indices)
        loss.backward(retain_graph=True)
        return loss.item()

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        res = self._get_result(self._to_tensor(x_values))
        return np.argmax(self._to_numpy(res), axis=1)

    def get_batch_indices(self, batch_size: int) -> np.ndarray:
        return np.random.choice(self.possible_indices, batch_size, replace=False)  # type: ignore

    def set_params(self, params: list[np.ndarray]):
        for name, new_param in zip(self.param_names, params):
            self.params[name] = self._to_tensor(new_param)

    def iterate_params_grad(self) -> list[tuple[np.ndarray, np.ndarray]]:
        params = []
        for name in self.param_names:
            p = self.params[name]
            grad = p.grad
            params.append((self._to_numpy(p), self._to_numpy(grad)))
        return params

    def iterate_params(self) -> list[np.ndarray]:
        params = []
        for name in self.param_names:
            p = self.params[name].clone()
            params.append(self._to_numpy(p))
        return params

    def get_data_size(self) -> int:
        return self.data_size

    def zero_grads(self):
        for name in self.param_names:
            self.params[name] = self.params[name].detach()


class TwoLayerFC(Problem):
    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x).float().to(self.device)

    def _to_numpy(self, x: torch.Tensor) -> np.ndarray:
        return x.cpu().detach().numpy()

    def _get_data(
        self, data_indices: Optional[np.ndarray] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if data_indices is None:
            return self.data

        return (self.data[0][data_indices], self.data[1][data_indices])

    def _get_result(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(
            F.relu(
                F.relu(x @ self.params["W1"] + self.params["B1"]) @ self.params["W2"]
                + self.params["B2"]
            ),
            dim=1,
        )

    def _get_loss(self, res: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(res, torch.argmax(targets, dim=1))

    def _get_value(self, data_indices: Optional[np.ndarray] = None) -> torch.Tensor:
        data = self._get_data(data_indices)
        for p in self.params.values():
            p.requires_grad_()

        x = data[0].to(self.device)
        y = data[1].to(self.device)

        res = self._get_result(x)

        return self._get_loss(res, y)

    def __init__(
        self,
        data: tuple[torch.Tensor, torch.Tensor],
        seed: float,
        device: torch.device,
        output_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        self.device = device
        self.data = data
        self.seed = seed

        self.data_size: int = data[0].shape[0]
        self.possible_indices = list(range(self.data_size))
        self.input_dim = self.data[0].shape[1]
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.reset()

    def reset(self):
        np.random.seed(self.seed)

        self.params: dict[str, torch.Tensor] = {
            "W1": self._to_tensor(np.random.rand(self.input_dim, self.hidden_dim)),
            "B1": self._to_tensor(np.random.rand(self.hidden_dim)),
            "W2": self._to_tensor(np.random.rand(self.hidden_dim, self.output_dim)),
            "B2": self._to_tensor(np.random.rand(self.output_dim)),
        }
        self.param_names = list(self.params.keys())

    def compute_gradient(self, data_indices: Optional[np.ndarray] = None) -> float:
        loss = self._get_value(data_indices)
        loss.backward(retain_graph=True)
        return loss.item()

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        res = self._get_result(self._to_tensor(x_values))
        return np.argmax(self._to_numpy(res), axis=1)

    def get_batch_indices(self, batch_size: int) -> np.ndarray:
        return np.random.choice(self.possible_indices, batch_size, replace=False)  # type: ignore

    def set_params(self, params: list[np.ndarray]):
        for name, new_param in zip(self.param_names, params):
            self.params[name] = self._to_tensor(new_param)

    def iterate_params_grad(self) -> list[tuple[np.ndarray, np.ndarray]]:
        params = []
        for name in self.param_names:
            p = self.params[name]
            grad = p.grad
            params.append((self._to_numpy(p), self._to_numpy(grad)))
        return params

    def iterate_params(self) -> list[np.ndarray]:
        params = []
        for name in self.param_names:
            p = self.params[name].clone()
            params.append(self._to_numpy(p))
        return params

    def get_data_size(self) -> int:
        return self.data_size

    def zero_grads(self):
        for name in self.param_names:
            self.params[name] = self.params[name].detach()


class SimpleCNN(Problem):
    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        return torch.tensor(x).float().to(self.device)

    def _to_numpy(self, x: torch.Tensor) -> np.ndarray:
        return x.cpu().detach().numpy()

    def _get_data(
        self, data_indices: Optional[np.ndarray] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if data_indices is None:
            return self.data

        return (self.data[0][data_indices], self.data[1][data_indices])

    def _get_result(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = F.relu(F.conv2d(x, self.params["K"])).view(-1, self.conv_out)
        return F.softmax(F.relu(feature_map @ self.params["W"] + self.params["B"]))

    def _get_loss(self, res: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(res, targets)

    def _get_value(self, data_indices: Optional[np.ndarray] = None) -> torch.Tensor:
        data = self._get_data(data_indices)
        for p in self.params.values():
            p.requires_grad_()

        x = data[0].to(self.device)
        y = data[1].to(self.device)

        res = self._get_result(x)

        return self._get_loss(res, y)

    def __init__(
        self,
        data: tuple[torch.Tensor, torch.Tensor],
        seed: float,
        device: torch.device,
        output_dim: int,
        kernel_size: int = 3,
    ) -> None:
        self.device = device
        self.seed = seed
        self.kernel_size = kernel_size

        self.data_reshape = 28
        self.data = (
            data[0].reshape(-1, 1, self.data_reshape, self.data_reshape),
            data[1],
        )

        self.data_size: int = data[0].shape[0]
        self.possible_indices = list(range(self.data_size))
        self.input_dim = self.data[0].shape[1]
        self.output_dim = output_dim
        self.conv_out = (self.data_reshape - self.kernel_size + 1) ** 2

        self.reset()

    def reset(self):
        np.random.seed(self.seed)

        self.params: dict[str, torch.Tensor] = {
            "K": self._to_tensor(
                np.random.rand(1, 1, self.kernel_size, self.kernel_size)
            ),
            "W": self._to_tensor(np.random.rand(self.conv_out, self.output_dim)),
            "B": self._to_tensor(np.random.rand(self.output_dim)),
        }
        self.param_names = list(self.params.keys())

    def compute_gradient(self, data_indices: Optional[np.ndarray] = None) -> float:
        loss = self._get_value(data_indices)
        loss.backward(retain_graph=True)
        return loss.item()

    def predict(self, x_values: torch.Tensor) -> np.ndarray:
        res = self._get_result(
            x_values.reshape(-1, 1, self.data_reshape, self.data_reshape).to(self.device)
        )
        return np.argmax(self._to_numpy(res), axis=1)

    def get_batch_indices(self, batch_size: int) -> np.ndarray:
        return np.random.choice(self.possible_indices, batch_size, replace=False)  # type: ignore

    def set_params(self, params: list[np.ndarray]):
        for name, new_param in zip(self.param_names, params):
            self.params[name] = self._to_tensor(new_param)

    def iterate_params_grad(self) -> list[tuple[np.ndarray, np.ndarray]]:
        params = []
        for name in self.param_names:
            p = self.params[name]
            grad = p.grad
            params.append((self._to_numpy(p), self._to_numpy(grad)))
        return params

    def iterate_params(self) -> list[np.ndarray]:
        params = []
        for name in self.param_names:
            p = self.params[name].clone()
            params.append(self._to_numpy(p))
        return params

    def get_data_size(self) -> int:
        return self.data_size

    def zero_grads(self):
        for name in self.param_names:
            self.params[name] = self.params[name].detach()


### Method classes


class Method:
    def update(self, f: Problem, iteration: int) -> int:
        raise NotImplementedError


IterationType = Literal["iterations"] | Literal["gradients"] | Literal["time"]


class Checker:
    def _get_accuracy(self) -> float:
        predictions = self.f.predict(self.test_values)
        return (predictions == self.test_labels).sum() / self.test_labels.shape[0]

    def __init__(
        self,
        name: str,
        f: Problem,
        optimizer: Method,
        test_data: tuple[torch.Tensor, torch.Tensor],
        verbose: bool = True,
    ) -> None:
        self.name = name
        self.f = f
        self.optimizer = optimizer
        self.test_values = test_data[0]
        self.verbose = verbose

        self.test_labels = torch.argmax(test_data[1], axis=1).cpu().numpy()

        self.get_iterations_check: dict[IterationType, Callable[[], float]] = {
            "iterations": lambda: self.total_pass_iterations,
            "gradients": lambda: self.total_used_gradients,
            "time": lambda: self.total_passed_time,
        }

        self.get_iterations_update: dict[IterationType, Callable[[int, float], float]] = {
            "iterations": lambda gradients, passed_time: 1,
            "gradients": lambda gradients, passed_time: gradients,
            "time": lambda gradients, passed_time: passed_time,
        }

        self.reset()

    def calculate_loss(self) -> float:
        loss = self.f.compute_gradient()
        self.f.zero_grads()
        return loss

    def predict(self, x_values: np.ndarray) -> np.ndarray:
        return self.f.predict(x_values)

    def get_name(self) -> str:
        return self.name

    def get_pass_iterations_logs(self) -> list[int]:
        return list(range(self.total_pass_iterations + 1))

    def get_used_gradients_logs(self) -> list[int]:
        return self.used_gradients_logs

    def get_time_logs(self) -> list[float]:
        return self.time_logs

    def get_accuracy_logs(self) -> list[float]:
        return self.accuracy_logs

    def get_loss_logs(self) -> list[float]:
        return self.loss_logs

    def reset(self):
        self.loss_logs = []
        self.time_logs = []
        self.accuracy_logs = []
        self.used_gradients_logs = []

        self.used_gradients = 0
        self.passed_time = 0

        self.total_used_gradients = 0
        self.total_pass_iterations = 0
        self.total_passed_time = 0

    def start(
        self,
        max_iterations: float,
        iteration_type: IterationType = "iterations",
        early_stop_accuracy: float = 0.99,
    ):
        self.reset()

        if self.verbose:
            progress_bar = tqdm(
                total=max_iterations, desc=self.name, leave=True, position=0
            )

        iterations_check = self.get_iterations_check[iteration_type]
        iterations_update = self.get_iterations_update[iteration_type]

        # logging
        self.time_logs.append(self.total_passed_time)
        self.loss_logs.append(self.calculate_loss())
        self.used_gradients_logs.append(self.total_used_gradients)
        self.accuracy_logs.append(self._get_accuracy())

        while iterations_check() < max_iterations:
            k = self.total_pass_iterations

            start_time = time.time()
            used_gradients = self.optimizer.update(self.f, k)
            finish_time = time.time()
            elapsed_time = finish_time - start_time
            loss = self.calculate_loss()

            self.total_pass_iterations += 1
            self.total_passed_time += elapsed_time
            self.total_used_gradients += used_gradients

            # logging
            accuracy = self._get_accuracy()
            self.time_logs.append(self.total_passed_time)
            self.loss_logs.append(loss)
            self.used_gradients_logs.append(self.total_used_gradients)
            self.accuracy_logs.append(accuracy)

            if self.verbose:
                info_dict = {
                    "time": self.passed_time,
                    "loss": loss,
                    "accuracy": accuracy,
                }
                progress_bar.set_postfix(info_dict)  # type: ignore
                progress_bar.update(iterations_update(used_gradients, elapsed_time))  # type: ignore

            if np.isnan(loss):  # diverges
                break

            if accuracy >= early_stop_accuracy:
                break

        if self.verbose:
            progress_bar.close()  # type: ignore


class SGD(Method):
    def __init__(
        self,
        step_size: Callable[[int], float],
        batch_size: int,
    ) -> None:
        self.step_size = step_size
        self.batch_size = batch_size

    def update(self, f: Problem, iteration: int) -> int:
        learning_rate = self.step_size(iteration)

        batch_indices = f.get_batch_indices(self.batch_size)
        f.compute_gradient(batch_indices)

        new_params = []
        for p, grad_p in f.iterate_params_grad():
            new_p = p - learning_rate * grad_p
            new_params.append(new_p.copy())
        f.zero_grads()
        f.set_params(new_params)

        return self.batch_size


class SVRG(Method):
    def __init__(
        self,
        step_size: Callable[[int], float],
        epoch_length: int,
        batch_size: int,
    ) -> None:
        self.step_size = step_size
        self.epoch_length = epoch_length
        self.batch_size = batch_size

    def update(self, f: Problem, iteration: int) -> int:
        learning_rate = self.step_size(iteration)

        tilde_params = f.iterate_params()
        f.compute_gradient()
        full_grads = [g for (_, g) in f.iterate_params_grad()]
        f.zero_grads()

        inside_params = tilde_params.copy()
        aggregated_params = tilde_params.copy()

        for _ in range(self.epoch_length):
            batch_indices = f.get_batch_indices(self.batch_size)

            f.set_params(inside_params)
            f.compute_gradient(batch_indices)
            inside_grads = [g for (_, g) in f.iterate_params_grad()]
            f.zero_grads()

            f.set_params(tilde_params)
            f.compute_gradient(batch_indices)
            tilde_grads = [g for (_, g) in f.iterate_params_grad()]
            f.zero_grads()

            for p_idx in range(len(tilde_grads)):
                g_k = inside_grads[p_idx] - tilde_grads[p_idx] + full_grads[p_idx]
                inside_params[p_idx] = inside_params[p_idx] - learning_rate * g_k
                aggregated_params[p_idx] += inside_params[p_idx]

        final_params = [
            aggregated_p / self.epoch_length for aggregated_p in aggregated_params
        ]
        f.set_params(final_params)

        return f.get_data_size() + 2 * self.epoch_length * self.batch_size


class LSVRG(Method):
    def __init__(
        self,
        step_size: Callable[[int], float],
        probability: float,
        batch_size: int,
    ) -> None:
        self.step_size = step_size
        self.batch_size = batch_size
        self.probability = probability

        self.has_params_tilde: bool = False
        self.params_tilde: list[np.ndarray] = [np.zeros(2)]
        self.full_grads_tilde: list[np.ndarray] = [np.zeros(2)]

    def update(self, f: Problem, iteration: int) -> int:
        learning_rate = self.step_size(iteration)

        if not self.has_params_tilde:
            self.has_params_tilde = True
            f.compute_gradient()
            self.params_tilde, self.full_grads_tilde = zip(*f.iterate_params_grad())
            f.zero_grads()

        initial_params = f.iterate_params()
        batch_indices = f.get_batch_indices(self.batch_size)

        f.compute_gradient(batch_indices)
        grads = [g for (_, g) in f.iterate_params_grad()]
        f.zero_grads()

        f.set_params(self.params_tilde)
        f.compute_gradient(batch_indices)
        grads_tilde = [g for (_, g) in f.iterate_params_grad()]
        f.zero_grads()

        total_grads = [
            g - g_tilde + full_grads_tilde
            for (g, g_tilde, full_grads_tilde) in zip(
                grads, grads_tilde, self.full_grads_tilde
            )
        ]

        final_params = [
            init_p - learning_rate * g for (init_p, g) in zip(initial_params, total_grads)
        ]

        used_gradients = 2 * self.batch_size

        if np.random.random() < self.probability:
            f.set_params(initial_params)
            f.compute_gradient()
            self.params_tilde, self.full_grads_tilde = zip(*f.iterate_params_grad())
            f.zero_grads()

            used_gradients += f.get_data_size()

        f.set_params(final_params)

        return used_gradients


class SARAH(Method):
    def __init__(
        self,
        step_size: Callable[[int], float],
        epoch_length: int,
        batch_size: int,
    ) -> None:
        self.step_size = step_size
        self.epoch_length = epoch_length
        self.batch_size = batch_size

    def update(self, f: Problem, iteration: int) -> int:
        learning_rate = self.step_size(iteration)

        initial_params = f.iterate_params()
        param_list = [initial_params.copy()]

        f.compute_gradient()
        vs = [g for (_, g) in f.iterate_params_grad()]
        f.zero_grads()

        param_list.append([p - learning_rate * v for (p, v) in zip(param_list[0], vs)])

        stop_idx = np.random.randint(self.epoch_length)

        for _ in range(stop_idx - 1):
            batch_indices = f.get_batch_indices(self.batch_size)

            f.set_params(param_list[-1])
            f.compute_gradient(batch_indices)
            prev_grads = [g for (_, g) in f.iterate_params_grad()]
            f.zero_grads()

            f.set_params(param_list[-2])
            f.compute_gradient(batch_indices)
            prev_prev_grads = [g for (_, g) in f.iterate_params_grad()]
            f.zero_grads()

            vs = [
                p_g - pp_g + v for (p_g, pp_g, v) in zip(prev_grads, prev_prev_grads, vs)
            ]

            param_list.append(
                [p - learning_rate * v for (p, v) in zip(param_list[-1], vs)]
            )

        f.set_params(param_list[stop_idx])

        return f.get_data_size() + 2 * (stop_idx + 1) * self.batch_size


class PAGE(Method):
    def __init__(
        self,
        step_size: Callable[[int], float],
        probability: float,
        primary_batch_size: int,
        secondary_batch_size: int,
    ) -> None:
        self.step_size = step_size
        self.probability = probability

        self.primary_batch_size = primary_batch_size
        self.secondary_batch_size = secondary_batch_size

        self.have_gs_computed: bool = False
        self.gs: list[np.ndarray] = [np.zeros(2)]

    def update(self, f: Problem, iteration: int) -> int:
        learning_rate = self.step_size(iteration)

        if not self.have_gs_computed:
            self.have_gs_computed = True
            batch_indices = f.get_batch_indices(self.primary_batch_size)
            f.compute_gradient(batch_indices)
            self.gs = [g for (_, g) in f.iterate_params_grad()]
            f.zero_grads()

        initial_params = f.iterate_params()
        next_params = [
            init_p - learning_rate * g for (init_p, g) in zip(initial_params, self.gs)
        ]
        f.set_params(next_params)

        if np.random.random() < self.probability:
            batch_indices = f.get_batch_indices(self.primary_batch_size)
            f.compute_gradient(batch_indices)
            self.gs = [g for (_, g) in f.iterate_params_grad()]
            f.zero_grads()

            return self.primary_batch_size

        batch_indices = f.get_batch_indices(self.secondary_batch_size)

        f.compute_gradient(batch_indices)
        current_grads = [g for (_, g) in f.iterate_params_grad()]
        f.zero_grads()

        f.set_params(initial_params)
        f.compute_gradient(batch_indices)
        prev_grads = [g for (_, g) in f.iterate_params_grad()]
        f.zero_grads()

        self.gs = [
            g + curr_g - prev_g
            for (g, curr_g, prev_g) in zip(self.gs, current_grads, prev_grads)
        ]

        return 2 * self.secondary_batch_size


### Data loading functions


def load_configs(path: str, logger: Logger) -> dict:
    logger.log(f"Rrading configs from '{path}'...")

    with open(path) as f:
        configs_dict = json.load(f)
    logger.log("Success!")
    return configs_dict


def load_MNIST_binary() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv("./data/interim/mnist_binary.csv", index_col=None)

    y, X = df.iloc[:, 0].to_numpy(), df.iloc[:, 1:].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    return X_train, y_train, X_test, y_test


def load_MNIST() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.read_csv("./data/raw/mnist.csv", index_col=None)

    y, X = df.iloc[:, 0].to_numpy(), df.iloc[:, 1:].to_numpy()
    X = X / 255

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    return X_train, y_train, X_test, y_test


def load_mushrooms() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = load_svmlight_file("./data/raw/mushrooms.txt")
    X, y = data[0].toarray(), data[1]
    y = 2 * y - 3

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, y_train, X_test, y_test


def load_dataset(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if dataset == "MNIST_binary":
        return load_MNIST_binary()
    if dataset == "MNIST":
        return load_MNIST()
    return load_mushrooms()


### Plot functions

PREDEFINED_COLORS = [
    "#ffa500",
    "#c83cbc",
    "#1c1c84",
    "#ff0000",
    "#08a4a7",
    "#008000",
]


AxesValues = (
    Literal["loss"]
    | Literal["accuracy"]
    | Literal["iterations"]
    | Literal["time"]
    | Literal["used_gradients"]
    | Literal["log_used_gradients"]
)
PREDEFINED_AXES_VALUES: dict[AxesValues, dict] = {
    "loss": {
        "name": "Loss",
        "label": "Loss, log scale",
        "get_data": lambda m: m.get_loss_logs(),
        "log": True,
    },
    "accuracy": {
        "name": "Accuracy",
        "label": "Accuracy",
        "get_data": lambda m: m.get_accuracy_logs(),
        "log": False,
    },
    "iterations": {
        "name": "Iterations",
        "label": "Iterations",
        "get_data": lambda m: m.get_pass_iterations_logs(),
        "log": False,
    },
    "time": {
        "name": "Time",
        "label": "Running time, seconds",
        "get_data": lambda m: m.get_time_logs(),
        "log": False,
    },
    "used_gradients": {
        "name": "Used gradients",
        "label": "Used gradients / n",
        "get_data": lambda m: np.array(m.get_used_gradients_logs()) / m.f.get_data_size(),
        "log": False,
    },
    "log_used_gradients": {
        "name": "Used gradients",
        "label": "Used gradients / n, log scale",
        "get_data": lambda m: np.array(m.get_used_gradients_logs()) / m.f.get_data_size(),
        "log": True,
    },
}


PlotDescription = tuple[AxesValues, AxesValues]


def draw_method_plots(
    methods: list[Checker],
    title: str,
    plots: list[PlotDescription],
    save_path: str,
    ylim=None,
    plot_width: float = 12,  # 16
    plot_height: float = 7,  # 7
    use_rainbow: bool = False,
    use_common_legend: bool = True,
):
    num_plots = len(plots)
    if use_rainbow:
        num_colors = len(methods)
        cm = plt.get_cmap("gist_rainbow")
        colors = [cm(1.0 * i / num_colors) for i in range(num_colors)]
    else:
        colors = PREDEFINED_COLORS

    style_cycler = cycler(linestyle=["-", "--", ":", "-."]) * cycler(color=colors)

    fig, axs = plt.subplots(num_plots, 1, figsize=(plot_width, plot_height * num_plots))
    fig.suptitle(title, fontsize=14)
    axs_list = [axs] if num_plots == 1 else list(axs.flat)

    for ax in axs_list:
        ax.grid()
        ax.set_prop_cycle(style_cycler)
        if ylim is not None:
            ax.set_ylim(top=ylim)

    for ax, (p1, p2) in zip(axs_list, plots):
        p1_data = PREDEFINED_AXES_VALUES[p1]
        p2_data = PREDEFINED_AXES_VALUES[p2]

        ax.set_title(f"{p2_data['name']} over {p1_data['name']}")
        ax.set(xlabel=p1_data["label"], ylabel=p2_data["label"])

        if p1_data["log"]:
            ax.set_xscale("log")
        if p2_data["log"]:
            ax.set_yscale("log")

        for method in methods:
            label = method.name

            x_values = p1_data["get_data"](method)
            y_values = p2_data["get_data"](method)

            ax.plot(x_values, y_values, label=label)
            ax.scatter(x_values[-1], y_values[-1], s=15)

    if use_common_legend:
        lines_labels = [axs_list[0].get_legend_handles_labels()]
        lines, labels = [sum(x, []) for x in zip(*lines_labels)]
        fig.legend(
            lines,
            labels,
            scatterpoints=1,
            markerscale=3,
            loc="outside lower center",
            ncol=min(5, len(methods)),
            bbox_to_anchor=(0.5, -0.05),
        )
    else:
        if len(methods) > 1:
            for ax in axs_list:
                # ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1))
                ax.legend()

    plt.tight_layout()
    plt.subplots_adjust(
        top=1 - 0.1 / (num_plots**0.5), bottom=0.12 / (num_plots**2), hspace=0.15
    )
    plt.savefig(save_path)


### Utilities functions


def setup_problem(model: str) -> type[Problem]:
    if model == "OneLayerFC":
        return OneLayerFC
    if model == "SimpleCNN":
        return SimpleCNN
    return TwoLayerFC


def setup_methods(
    algorithms: list[str],
    problem: type[Problem],
    configs_dict: dict,
    train_values: torch.Tensor,
    train_targets: torch.Tensor,
    test_values: torch.Tensor,
    test_targets: torch.Tensor,
    random_seed: int,
    device: torch.device,
    num_labels: int,
    data_size: int,
    logger: Logger,
) -> list[Checker]:
    logger.log("Setting up methods...")
    methods = []

    for m in algorithms:
        m_configs = configs_dict[m]

        if m == "SVRG":
            method = SVRG(
                lambda _: m_configs["step_size"],
                m_configs["epoch_length"],
                data_size // m_configs["batches_number"],
            )
        elif m == "L-SVRG":
            method = LSVRG(
                lambda _: m_configs["step_size"],
                m_configs["probability"],
                data_size // m_configs["batches_number"],
            )

        elif m == "SARAH":
            method = SARAH(
                lambda _: m_configs["step_size"],
                m_configs["epoch_length"],
                m_configs["batch_size"],
            )

        else:
            method = PAGE(
                lambda _: m_configs["step_size"],
                m_configs["probability"],
                m_configs["batch_size_primary"],
                m_configs["batch_size_secondary"],
            )

        methods.append(
            Checker(
                m,
                problem((train_values, train_targets), random_seed, device, num_labels),
                method,
                (test_values, test_targets),
                verbose=logger.verbose,
            ),
        )

    logger.log("Success!")
    return methods


def start_methods(
    methods: list[Checker],
    accuracy_threshold: float,
    max_iterations: float,
    iterations_type: IterationType,
):
    for method in methods:
        _ = method.start(
            max_iterations=max_iterations,
            iteration_type=iterations_type,
            early_stop_accuracy=accuracy_threshold,
        )


def construct_and_save_plots(
    methods: list[Checker], save_path: str, dataset: str, model: str, logger: Logger
):
    plot_pairs = [
        ("used_gradients", "loss"),
        ("used_gradients", "accuracy"),
        ("iterations", "loss"),
        ("iterations", "accuracy"),
    ]
    title = f"{dataset} | {model}"
    names = "_".join([m.name for m in methods])
    dt = int(time.time())
    for plot_pair in plot_pairs:
        name1 = PREDEFINED_AXES_VALUES[plot_pair[0]]["name"]
        name2 = PREDEFINED_AXES_VALUES[plot_pair[1]]["name"]
        file_name = f"{names}_{name1}over{name2}{dt}.svg"
        logger.log(f"Plotting {name1} over {name2}...")
        draw_method_plots(
            methods,
            plots=[plot_pair],
            title=title,
            save_path=os.path.join(save_path, file_name),
            use_common_legend=False,
        )
        logger.log(f"Successfully saved to {os.path.join(save_path,file_name)}")


def labels_to_vectors(labels: np.ndarray, num_labels: int) -> np.ndarray:
    vectorized = []
    local_map_dict = {label: i for i, label in enumerate(np.unique(labels))}
    for label in labels:
        zeroes = np.zeros(num_labels)
        zeroes[local_map_dict[label]] = 1.0
        vectorized.append(zeroes)
    return np.array(vectorized)


def evaluate():
    """Evaluate methods"""

    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate methods")

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        dest="model",
        choices=AVAILABLE_MODELS,
        default=AVAILABLE_MODELS[0],
        help=f"model to train (default: {AVAILABLE_MODELS[0]})",
    )
    parser.add_argument(
        "-s",
        "--save-path",
        type=str,
        dest="save_path",
        default="./data/figures",
        help="relative path to save generated plots (default: ./data/figures)",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        dest="dataset",
        choices=AVAILABLE_DATASETS,
        default=AVAILABLE_DATASETS[0],
        help=f"dataset to use (default: {AVAILABLE_DATASETS[0]})",
    )
    parser.add_argument(
        "-a",
        "--algorithms",
        type=str,
        nargs="+",
        dest="algorithms",
        default=AVAILABLE_METHODS[:2],
        help=f"algorithms to compare \
            (default: {AVAILABLE_METHODS[:2]})",
    )
    parser.add_argument(
        "-l",
        "--load-configs-path",
        type=str,
        dest="load_configs_path",
        default="./data/interim/configs.json",
        help="path to methods configuration (default: ./data/interim/configs.json)",
    )
    parser.add_argument(
        "-t",
        "--accuracy-threshold",
        type=float,
        dest="accuracy_threshold",
        default=0.99,
        help="accuracy threshold for early stopping (default: 0.99)",
    )
    parser.add_argument(
        "-x",
        "--max-iterations",
        type=float,
        dest="max_iterations",
        default=1e6,
        help="maximum number of iterations (default: 1e6)",
    )
    parser.add_argument(
        "-y",
        "--iterations-type",
        type=str,
        dest="iterations_type",
        default="gradients",
        help="type of iterations (default: 1e6)",
    )
    parser.add_argument(
        "-r",
        "--random-seed",
        type=int,
        dest="random_seed",
        default=42,
        help="random seed (default: 42)",
    )
    parser.add_argument(
        "-c",
        "--cuda",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="use CUDA if available (default: True)",
    )
    parser.add_argument(
        "-i",
        "--ignore-warnings",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="ignore warnings (default: True)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="print information (default: True)",
    )

    namespace = parser.parse_args()
    (
        model,
        save_path,
        dataset,
        algorithms,
        load_configs_path,
        accuracy_threshold,
        max_iterations,
        iterations_type,
        random_seed,
        cuda,
        ignore_warnings,
        verbose,
    ) = (
        namespace.model,
        namespace.save_path,
        namespace.dataset,
        namespace.algorithms,
        namespace.load_configs_path,
        namespace.accuracy_threshold,
        namespace.max_iterations,
        namespace.iterations_type,
        namespace.random_seed,
        namespace.cuda,
        namespace.ignore_warnings,
        namespace.verbose,
    )
    verbose: bool = bool(verbose)
    cuda: bool = bool(cuda)
    ignore_warnings: bool = bool(ignore_warnings)

    if iterations_type not in AVAILABLE_ITERATIONS_TYPE:
        print(f"ERROR! Unfamiliar iteration type '{iterations_type}'!")
        raise RuntimeError()

    device = torch.device("cuda" if (cuda and torch.cuda.is_available()) else "cpu")

    if ignore_warnings:
        warnings.filterwarnings("ignore")

    # Set up logger
    logger = Logger(verbose)

    if dataset == "Mushrooms" and model == "SimpleCNN":
        print("ERROR! Cannot use SimpleCNN model with Mushrooms dataset!")
        raise RuntimeError()

    for method in algorithms:
        if method not in AVAILABLE_METHODS:
            print(f"ERROR! '{method}' is not available")
            raise RuntimeError()

    print(algorithms)
    # Load configs
    configs_dict = load_configs(load_configs_path, logger)

    # Load dataset
    logger.log(f"Loading dataset '{dataset}'...")
    X_train, Y_train, X_test, Y_test = load_dataset(dataset)
    logger.log("Success!")

    # Convert to tensors
    logger.log("Converting data to tensors...")
    data_size = Y_train.shape[0]
    num_labels = len(np.unique(np.concatenate([Y_train, Y_test])))
    train_values = torch.tensor(X_train).float()
    train_targets = torch.tensor(labels_to_vectors(Y_train, num_labels)).float()
    test_values = torch.tensor(X_test).float()
    test_targets = torch.tensor(labels_to_vectors(Y_test, num_labels)).float()
    logger.log("Success!")

    # Setup problem
    problem = setup_problem(model)

    # Setup methods
    methods = setup_methods(
        algorithms,
        problem,
        configs_dict,
        train_values,
        train_targets,
        test_values,
        test_targets,
        random_seed,
        device,
        num_labels,
        data_size,
        logger,
    )

    # Start methods
    start_methods(
        methods,
        accuracy_threshold,
        max_iterations,
        iterations_type,
    )

    # # Construct and save plots
    construct_and_save_plots(methods, save_path, dataset, model, logger)

    logger.log("Done!")


if __name__ == "__main__":
    evaluate()
