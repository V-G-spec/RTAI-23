import torch
from torch import Tensor

from utils.conv import get_hw_after_conv, torch_conv_layer_to_affine, flat
from polygon import Polygon


class BaseLayerTransformer(torch.nn.Module):
    dtype = None

    def __init__(self, layer, input_shape):
        super().__init__()
        self._layer = (layer, )  # hacky way to hide layer from PyTorch's parameter lookup
        self.input_shape = input_shape
        if self.layer is not None:
            for param in self.layer.parameters():
                param.requires_grad = False

    @property
    def layer(self):
        return self._layer[0]

    def forward(self, *args, **kwargs) -> Polygon:
        raise NotImplementedError


class LayerTransformer(BaseLayerTransformer):
    _layer_type = None

    def forward(self, poly: Polygon) -> Polygon:
        raise NotImplementedError


class Input(BaseLayerTransformer):
    def __init__(self):
        super().__init__(layer=None, input_shape=None)

    def forward(self, x: Tensor, eps: float) -> Polygon:
        lower_bound = (x - eps).clamp(min=0, max=1).flatten()
        upper_bound = (x + eps).clamp(min=0, max=1).flatten()

        return Polygon(
            parent=None,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            output_shape=x.shape,
        )


class LinearTransformer(LayerTransformer):
    _layer_type = torch.nn.Linear

    @property
    def weights(self):
        return self.layer.weight.data.detach().clone()

    @property
    def bias(self):
        return self.layer.bias.data.detach().clone() if self.layer.bias is not None else None

    def forward(self, poly: Polygon) -> Polygon:
        assert self.input_shape == torch.Size([self.layer.in_features]), "Shape mismatch"

        return Polygon(
            parent=poly,
            lower_weight=self.weights, lower_bias=self.bias,
            upper_weight=self.weights, upper_bias=self.bias,
            output_shape=torch.Size([self.layer.out_features]))


class IdentityTransformer(LayerTransformer):
    _layer_type = torch.nn.Identity

    def forward(self, poly: Polygon) -> Polygon:
        return poly  # TODO verify if we want to copy it or not


class FlattenTransformer(LayerTransformer):
    _layer_type = torch.nn.Flatten

    def __init__(self, layer, input_shape):
        super().__init__(layer, input_shape)
        self.output_shape = flat(input_shape)

    def forward(self, poly: Polygon) -> Polygon:
        # Since these are identity matrices, we keep them as sparse matrices to decrease calc from O(n**2) to O(n)
        return Polygon(parent=poly,
                       lower_weight=torch.ones(self.output_shape),
                       upper_weight=torch.ones(self.output_shape),
                       output_shape=torch.Size([self.output_shape]))


class Conv2DTransformer(LinearTransformer):
    _layer_type = torch.nn.Conv2d

    def __init__(self, layer, input_shape):
        super().__init__(layer=None, input_shape=input_shape)
        self.output_shape = None
        self._conv_layer = layer

    def forward(self, poly: Polygon) -> Polygon:
        assert poly.output_shape == self.input_shape, "Shape mismatch"
        h, w = get_hw_after_conv(self._conv_layer, poly.output_shape)
        self._layer = (torch_conv_layer_to_affine(self._conv_layer, poly.output_shape), )  # hacky thing
        __input_shape = self.input_shape
        self.input_shape = flat(self.input_shape)
        output = super().forward(poly)
        self.input_shape = __input_shape
        output.output_shape = torch.Size([self._conv_layer.out_channels, h, w])
        return output


class LeakyReLUTransformer(LayerTransformer):
    _layer_type = torch.nn.LeakyReLU

    def __init__(self, layer, input_shape):
        super().__init__(layer, input_shape)
        self.layer_slope = layer.negative_slope if hasattr(layer, "negative_slope") else None
        self.alpha = torch.nn.Parameter(torch.ones(flat(self.input_shape)))

        self._init_with_heuristic = False  # have the weights been initialized with the heuristic or just set to 1

    def forward(self, poly: Polygon) -> Polygon:
        assert poly.output_shape == self.input_shape, "Shape mismatch"
        poly.backsubstitute()

        # Init weights if needed
        if not self._init_with_heuristic:
            # tunable_slope is either 1. y >= layer_slope or 2. y >= x
            condition: Tensor = poly.upper_bound <= -poly.lower_bound
            with torch.no_grad():
                self.alpha.data[condition] = self.layer_slope
            self._init_with_heuristic = True
        # Clamp them
        with torch.no_grad():
            low = min(self.layer_slope, 1)
            high = max(self.layer_slope, 1)
            self.alpha.data.clamp_(low, high)

        # TODO probably buggy

        args = (flat(poly.output_shape),)
        kwargs = {'dtype': self.dtype} if self.dtype is not None else {}
        lower_weight = torch.ones(flat(poly.output_shape), dtype=self.dtype)
        upper_weight = torch.ones(flat(poly.output_shape), dtype=self.dtype)
        lower_bias = torch.zeros(*args, **kwargs)
        upper_bias = torch.zeros(*args, **kwargs)

        # strictly negative: upper_bounds[-1] <= 0
        lower_weight[poly.upper_bound <= 0] = self.layer_slope
        upper_weight[poly.upper_bound <= 0] = self.layer_slope

        # strictly positive: lower_bounds[-1] >= 0
        # depend on single previous neuron with identity, so don't change from eye

        tunable_slope, constant_slope = lower_weight, upper_weight
        constant_bias = upper_bias
        if self.layer_slope > 1:
            # Swap
            tunable_slope, constant_slope = constant_slope, tunable_slope
            constant_bias = lower_bias
        crossing: Tensor = (poly.lower_bound < 0) & (0 < poly.upper_bound)  # noqa
        if torch.any(crossing):
            # tunable_slope is either 1. y >= layer_slope or 2. y >= x
            tunable_slope[crossing] = self.alpha[crossing]

            # constant_slope is always y <= lambda * (x - l) = -(lambda * l) * 1 + (lambda) * x
            # lambda values are identical for both relaxations
            lambdas = (poly.upper_bound - poly.lower_bound * self.layer_slope) / (
                    poly.upper_bound - poly.lower_bound)
            constant_slope[crossing] = lambdas[crossing]
            # add constant bias term in form y - m * x = b  # TODO seems like it has to be negated for some reason
            constant_bias[crossing] = -(lambdas * poly.lower_bound - self.layer_slope * poly.lower_bound)[crossing]

        # TODO simplify
        if self.layer_slope > 1:
            return Polygon(
                parent=poly,
                lower_weight=constant_slope, lower_bias=constant_bias,
                upper_weight=tunable_slope, upper_bias=upper_bias,
                output_shape=poly.output_shape)
        return Polygon(
            parent=poly,
            lower_weight=tunable_slope, lower_bias=lower_bias,
            upper_weight=constant_slope, upper_bias=constant_bias,
            output_shape=poly.output_shape)


class ReLUTransformer(LeakyReLUTransformer):
    _layer_type = torch.nn.ReLU

    def __init__(self, layer, input_shape):
        super().__init__(layer, input_shape)
        self.layer_slope = 0


class Output(LayerTransformer):
    def __init__(self, true_label: int, input_shape):
        super().__init__(layer=None, input_shape=input_shape)
        self.true_label = true_label

        args = (flat(self.input_shape),)
        kwargs = {'dtype': self.dtype} if self.dtype is not None else {}
        weights = -1 * torch.eye(*args, **kwargs)
        weights[:, self.true_label] = 1
        self.weights = torch.cat([weights[:self.true_label], weights[self.true_label + 1:]])  # could be made sparse

    def forward(self, poly: Polygon) -> Polygon:
        assert len(poly.output_shape) == 1, "Output has invalid dimensions"
        assert poly.output_shape == self.input_shape, "Shape mismatch"

        return Polygon(
            parent=poly,
            lower_weight=self.weights, upper_weight=self.weights,
            output_shape=torch.Size([poly.output_shape[0] - 1]))


# _transformers = [FlattenTransformer, LinearTransformer, Conv2DTransformer, ]
_transformers = [FlattenTransformer, LinearTransformer, Conv2DTransformer, ReLUTransformer, LeakyReLUTransformer]


def get_transformer(layer: torch.nn.Module) -> LayerTransformer:
    for _t_class in _transformers:
        if isinstance(layer, _t_class._layer_type):
            return _t_class
    else:
        raise NotImplementedError(layer)
