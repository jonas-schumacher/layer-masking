import torch


class MaskedLayer(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.Tensor,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bias=False,  # no need to use a bias
        )
        self.mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.linear(x, self.weight * self.mask, self.bias)
        return x


class CustomModel(torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask: torch.Tensor,
    ) -> None:
        super(CustomModel, self).__init__()

        self.masked_layer = MaskedLayer(
            in_features=in_features,
            out_features=out_features,
            mask=mask,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.masked_layer(x)
        return x
