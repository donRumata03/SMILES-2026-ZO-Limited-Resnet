"""
zo_optimizer.py — Zero-order optimizer skeleton (student-implemented).

Students: Implement your gradient-free optimization logic inside
``ZeroOrderOptimizer``. The skeleton uses a 2-point central-difference
estimator as a starting point — you are expected to replace or extend it.

Key design points
-----------------
* **Layer selection** is entirely your responsibility. Set ``self.layer_names``
  to the list of parameter names you want to optimize. You can change this list
  at any time — even between ``.step()`` calls — to implement curriculum or
  progressive-layer strategies.
* **Compute budget** is enforced by ``validate.py``: ``.step()`` is called
  exactly ``n_batches`` times. Each call may invoke the model as many times as
  your estimator requires, but be mindful that more evaluations per step leave
  fewer steps in the total budget.
* **No gradients** are computed anywhere in this file. All updates must be
  derived from scalar loss values obtained by calling ``loss_fn()``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import yaml


def _load_config() -> dict:
    with Path("zo_config.yaml").open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ZeroOrderOptimizer:
    """Gradient-free optimizer for fine-tuning a subset of model parameters.

    The optimizer maintains a list of *active* parameter names
    (``self.layer_names``). On each ``.step()`` call it perturbs only those
    parameters, estimates a pseudo-gradient from forward-pass loss values, and
    applies an update. All other parameters remain strictly frozen.

    Args:
        model:            The ``nn.Module`` to optimize.
        lr:               Step size / learning rate.
        eps:              Perturbation magnitude for the finite-difference
                          estimator.
        perturbation_mode: Distribution used to sample the perturbation
                          direction. ``"gaussian"`` draws from N(0, I);
                          ``"uniform"`` draws from U(-1, 1) and normalises.

    Student task:
        1. Set ``self.layer_names`` to the parameter names you want to tune.
           Inspect available names with ``[n for n, _ in model.named_parameters()]``.
        2. Replace or extend ``_estimate_grad`` with a better estimator.
        3. Replace or extend ``_update_params`` with a better update rule.
        4. Optionally change ``self.layer_names`` inside ``.step()`` to
           implement dynamic layer selection strategies.

    Example — tune only the final linear layer::

        optimizer = ZeroOrderOptimizer(model)
        optimizer.layer_names = ["fc.weight", "fc.bias"]
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        eps: float = 1e-3,
        perturbation_mode: str = "gaussian",
    ) -> None:
        self.model = model
        self.lr = lr
        self.eps = eps

        if perturbation_mode not in ("gaussian", "uniform"):
            raise ValueError(
                f"perturbation_mode must be 'gaussian' or 'uniform', "
                f"got '{perturbation_mode}'"
            )
        self.perturbation_mode = perturbation_mode
        self.config = _load_config()
        self.use_lora = bool(self.config["use_lora"])
        self.lora_rank = int(self.config["lora_rank"])
        self.layer_names: list[str] = ["fc.weight"] if self.use_lora else ["fc.bias"]

        self.bias_lr = 0.05
        self.bias_eps = 0.01
        self.lora_lr = 0.02
        self.lora_eps = 0.01
        self.rng = torch.Generator(device="cpu")
        self.rng.manual_seed(42)

        self.weight0 = self.model.fc.weight.detach().cpu().clone()
        self.a = torch.zeros(self.weight0.shape[0], self.lora_rank, dtype=self.weight0.dtype)
        self.b = torch.randn(
            self.lora_rank,
            self.weight0.shape[1],
            generator=self.rng,
            dtype=self.weight0.dtype,
        ) / (self.weight0.shape[1] ** 0.5)

    # ------------------------------------------------------------------
    # Internal helpers — students may modify these.
    # ------------------------------------------------------------------

    def _active_params(self) -> dict[str, nn.Parameter]:
        """Return a mapping from name → parameter for all active layer names.

        Only parameters whose names appear in ``self.layer_names`` are
        returned. Parameters not in this mapping are never modified.

        Returns:
            Dict mapping parameter name to its ``nn.Parameter`` tensor.

        Raises:
            KeyError: If a name in ``self.layer_names`` does not exist in the
                      model.
        """
        named = dict(self.model.named_parameters())
        missing = [n for n in self.layer_names if n not in named]
        if missing:
            raise KeyError(
                f"The following layer names were not found in the model: "
                f"{missing}. Use [n for n, _ in model.named_parameters()] "
                f"to inspect valid names."
            )
        return {n: named[n] for n in self.layer_names}

    def _sample_direction(self, param: torch.Tensor) -> torch.Tensor:
        """Sample a random unit-norm perturbation vector of the same shape as ``param``.

        Args:
            param: The parameter tensor whose shape determines the output shape.

        Returns:
            A tensor of the same shape as ``param``, normalised to unit L2 norm.
        """
        u = torch.randint(
            0,
            2,
            param.shape,
            generator=self.rng,
            device="cpu",
            dtype=torch.int64,
        ).to(dtype=param.dtype)
        return u.mul_(2.0).sub_(1.0).to(device=param.device)

    def _materialize_lora_weight(self, a_value: torch.Tensor | None = None) -> None:
        if a_value is None:
            a_value = self.a
        weight = self.weight0.to(device=self.model.fc.weight.device, dtype=self.model.fc.weight.dtype)
        a_value = a_value.to(device=self.model.fc.weight.device, dtype=self.model.fc.weight.dtype)
        b_value = self.b.to(device=self.model.fc.weight.device, dtype=self.model.fc.weight.dtype)
        with torch.no_grad():
            self.model.fc.weight.copy_(weight + a_value @ b_value)

    def _estimate_grad(
        self,
        loss_fn: Callable[[], float],
        params: dict[str, nn.Parameter],
    ) -> dict[str, torch.Tensor]:
        """Estimate a pseudo-gradient for each active parameter.

        Skeleton: 2-point central-difference estimator.
        For each active parameter ``p`` independently:
            1. Sample a random unit vector ``u`` of the same shape as ``p``.
            2. Evaluate  f_plus  = loss_fn() with ``p ← p + eps * u``
            3. Evaluate  f_minus = loss_fn() with ``p ← p - eps * u``
            4. Restore ``p`` to its original value.
            5. Pseudo-gradient ← ``(f_plus - f_minus) / (2 * eps) * u``

        This is an unbiased estimator of the directional derivative along ``u``
        scaled back to parameter space.

        Args:
            loss_fn: Callable that evaluates the objective on the current batch
                     and returns a scalar ``float``. May be called multiple
                     times; each call must use the *same* batch.
            params:  Dict of active parameter name → tensor (from
                     ``_active_params``).

        Returns:
            Dict mapping each parameter name to its estimated pseudo-gradient
            tensor (same shape as the parameter).

        Student task:
            Replace this with a more efficient or accurate estimator:
        """
        # ------------------------------------------------------------------
        # STUDENT: Replace or extend the gradient estimation below.
        # ------------------------------------------------------------------
        grads: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            if self.use_lora:
                delta_a = self._sample_direction(self.a)
                a_plus = self.a + self.lora_eps * delta_a
                a_minus = self.a - self.lora_eps * delta_a
                self._materialize_lora_weight(a_plus)
                f_plus = loss_fn()
                self._materialize_lora_weight(a_minus)
                f_minus = loss_fn()
                self._materialize_lora_weight(self.a)
                grads["fc.weight"] = ((f_plus - f_minus) / (2.0 * self.lora_eps)) * delta_a
                return grads

            for name, param in params.items():
                u = self._sample_direction(param)

                # f(x + eps * u)
                param.data.add_(self.bias_eps * u)
                f_plus = loss_fn()

                # f(x - eps * u)  — restore then subtract
                param.data.sub_(2.0 * self.bias_eps * u)
                f_minus = loss_fn()

                # Restore original value
                param.data.add_(self.bias_eps * u)

                grad_estimate = ((f_plus - f_minus) / (2.0 * self.bias_eps)) * u
                grads[name] = grad_estimate

        return grads
        # ------------------------------------------------------------------

    def _update_params(
        self,
        params: dict[str, nn.Parameter],
        grads: dict[str, torch.Tensor],
    ) -> None:
        """Apply the estimated pseudo-gradients to the active parameters.

        Skeleton: vanilla gradient *descent* step (minimising the loss).
            ``p ← p - lr * grad``

        Args:
            params: Dict of active parameter name → tensor.
            grads:  Dict of pseudo-gradient name → tensor (same keys as
                    ``params``).

        Student task:
            Replace with a more sophisticated update rule, e.g.:
              - Momentum: accumulate an exponential moving average of gradients.
              - Adam-style: maintain first and second moment estimates.
              - Clipped update: ``p ← p - lr * clip(grad, max_norm)``.
        """
        # ------------------------------------------------------------------
        # STUDENT: Replace or extend the parameter update below.
        # ------------------------------------------------------------------
        with torch.no_grad():
            if self.use_lora:
                self.a.sub_(self.lora_lr * grads["fc.weight"].cpu())
                self._materialize_lora_weight(self.a)
                return

            for name, param in params.items():
                param.data.sub_(self.bias_lr * grads[name])
        # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, loss_fn: Callable[[], float]) -> float:
        """Perform one zero-order optimisation step.

        Calls ``loss_fn`` one or more times to estimate pseudo-gradients for
        the currently active parameters (``self.layer_names``), then applies
        an update. Parameters *not* in ``self.layer_names`` are never touched.

        Args:
            loss_fn: A callable that takes no arguments and returns a scalar
                     ``float`` representing the loss on the current mini-batch.
                     ``validate.py`` guarantees that every call to ``loss_fn``
                     within a single ``.step()`` invocation uses the *same*
                     fixed batch of data.

        Returns:
            The loss value at the *start* of the step (before any update),
            obtained from the first call to ``loss_fn()``.

        Note:
            ``validate.py`` calls ``.step()`` exactly ``n_batches`` times.
            Each forward pass inside ``loss_fn`` counts toward your compute
            budget, so prefer estimators that minimise the number of calls.
        """
        params = self._active_params()

        # Record the loss before any perturbation.
        with torch.no_grad():
            if self.use_lora:
                self._materialize_lora_weight(self.a)
            loss_before = loss_fn()

        grads = self._estimate_grad(loss_fn, params)
        self._update_params(params, grads)

        return float(loss_before)
