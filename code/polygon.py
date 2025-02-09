from typing import Optional

import torch
from torch import Tensor, Size


def smartmatmul(a: Tensor, b: Tensor, force_a_vector=False, force_b_vector=False):
    """Assumes input tensors are matrices or diagonal matrices represented as vectors unless force_X_vector is set"""
    if force_a_vector:
        raise NotImplementedError
    if a.dim() > 1 and b.dim() > 1:
        return a @ b
    if a.dim() > 1 and force_b_vector:
        return (a @ b.unsqueeze(1)).squeeze()  # can be about 3x faster than (a * b).sum(dim=1)
    if a.dim() > 1 or b.dim() > 1:  # but not both...
        return a * b
    return a * b


class Polygon:
    def __init__(self, *, parent: Optional['Polygon'],
                 lower_weight: Optional[Tensor] = None,
                 lower_bias: Optional[Tensor] = None,
                 upper_weight: Optional[Tensor] = None,
                 upper_bias: Optional[Tensor] = None,
                 lower_bound: Optional[Tensor] = None,
                 upper_bound: Optional[Tensor] = None,
                 output_shape: Optional[Size]):
        self.parent = parent
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        # Weights are always 2D matrices. For efficiency's sake, we can, in the case of diagonal matrices, use
        #  elementwise multiplication with a vector instead of matrix multiplication with a (sparse) diagonal matrix,
        #  leading to a 15x potential speedup.
        # matrix * vector
        # matrix @ torch.diag(vector).to_sparse_coo()
        self.lower_weight = lower_weight if lower_weight is not None else None
        self.upper_weight = upper_weight if upper_weight is not None else None

        if lower_bias is None and self.lower_weight is not None:
            lower_bias = torch.zeros(self.lower_weight.size(0), dtype=self.lower_weight.dtype)
        if upper_bias is None and self.upper_weight is not None:
            upper_bias = torch.zeros(self.upper_weight.size(0), dtype=self.upper_weight.dtype)
        self.lower_bias = lower_bias
        self.upper_bias = upper_bias

        self.output_shape = output_shape

    # @profile # kernprof -v -l code/test.py
    def backsubstitute(self, allow_cache=False):
        # Weights might be sparse matrices, as we often have sparse transformations (e.g. for ReLU it is
        #   a diagonal matrix) and operations are significantly faster (50ms instead of 10+ seconds)
        lower_weight: Tensor = self.lower_weight.clone()
        lower_bias: Tensor = self.lower_bias.clone()
        upper_weight: Tensor = self.upper_weight.clone()
        upper_bias: Tensor = self.upper_bias.clone()

        parent = self.parent
        # Try and go as deep as possible, so back to Input()
        while parent.parent is not None:
            # Split decision through ReLU-like switches; using torch.where is not feasible on larger matrices
            # Do bias first because weight is updated
            zero = torch.zeros_like(lower_weight)
            lwmin = torch.min(lower_weight, zero)
            lwmax = torch.max(lower_weight, zero)
            lower_bias = lower_bias + smartmatmul(lwmax, parent.lower_bias, force_b_vector=True) + smartmatmul(
                lwmin, parent.upper_bias, force_b_vector=True)
            lower_weight = smartmatmul(lwmax, parent.lower_weight) + smartmatmul(lwmin, parent.upper_weight)

            zero = torch.zeros_like(upper_weight)
            uwmin = torch.min(upper_weight, zero)
            uwmax = torch.max(upper_weight, zero)
            upper_bias = upper_bias + smartmatmul(uwmin, parent.lower_bias, force_b_vector=True) + smartmatmul(
                uwmax, parent.upper_bias, force_b_vector=True)
            upper_weight = smartmatmul(uwmin, parent.lower_weight) + smartmatmul(uwmax, parent.upper_weight)
            parent = parent.parent  # GRANDPA :D

        # "Caching" of deep dependencies
        if allow_cache:
            self.parent = parent
            self.lower_weight = lower_weight
            self.upper_weight = upper_weight
            self.lower_bias = lower_bias
            self.upper_bias = upper_bias

        zero = torch.zeros_like(lower_weight)
        lwmin = torch.min(lower_weight, zero)
        lwmax = torch.max(lower_weight, zero)
        self.lower_bound = smartmatmul(lwmax, parent.lower_bound, force_b_vector=True) + smartmatmul(
            lwmin, parent.upper_bound, force_b_vector=True) + lower_bias

        zero = torch.zeros_like(upper_weight)
        uwmin = torch.min(upper_weight, zero)
        uwmax = torch.max(upper_weight, zero)
        self.upper_bound = smartmatmul(uwmin, parent.lower_bound, force_b_vector=True) + smartmatmul(
            uwmax, parent.upper_bound, force_b_vector=True) + upper_bias

    def __str__(self):
        d = dict(upper_weight=self.upper_weight, lower_weight=self.lower_weight,
                 upper_bias=self.upper_bias, lower_bias=self.lower_bias,
                 lower_bound=self.lower_bound, upper_bound=self.upper_bound,
                 output_shape=self.output_shape)
        ds = ', '.join([f"{key}={value if not hasattr(value, 'shape') else value.shape}" for key, value in d.items()
                        if value is not None])
        return f'{type(self).__name__}(parent={type(self.parent).__name__ if self.parent is not None else None}, {ds})'

    def __repr__(self):
        return str(self)
