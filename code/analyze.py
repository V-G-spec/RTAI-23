import torch
from layers import Input, get_transformer, Output, BaseLayerTransformer

DOUBLE_PRECISION = False
REGULARIZE = True


def analyze(net: torch.nn.Module,
            inputs: torch.Tensor, eps: float, true_label: int) -> bool:
    assert isinstance(net, torch.nn.Sequential)
    if DOUBLE_PRECISION:
        net = net.double()
        inputs = inputs.double()
        BaseLayerTransformer.dtype = torch.double

    assert len(inputs.shape) == 1 + 2  # Channels x Data; no batch dimension
    inputs.requires_grad = True

    transformers = [Input()]
    poly = transformers[0].forward(inputs.clone(), eps=eps)
    for layer in net:
        transformer = get_transformer(layer)(
            layer=layer, input_shape=poly.output_shape)
        transformers.append(transformer)
        poly = transformers[-1].forward(poly)
    transformers.append(Output(true_label, input_shape=poly.output_shape))
    poly = transformers[-1].forward(poly)
    poly.backsubstitute()

    if torch.all(poly.lower_bound > 0):
        return True

    transformers = torch.nn.Sequential(*transformers)
    if len([x for x in transformers.parameters()]) > 0:
        optimizer = torch.optim.Adam(transformers.parameters(), lr=0.5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, cooldown=5,
                                                               threshold=0.02, threshold_mode="abs", min_lr=0.1)
        while True:
            optimizer.zero_grad()
            loss = -1 * torch.min(poly.lower_bound)
            if REGULARIZE:
                loss = loss + 0.01 * -1 * torch.mean(poly.lower_bound)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            poly = transformers[0].forward(inputs, eps=eps)
            for transformer in transformers[1:]:
                poly = transformer.forward(poly)
            poly.backsubstitute()
            if torch.all(poly.lower_bound > 0):
                return True
    return False
