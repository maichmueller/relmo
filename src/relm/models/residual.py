from torch.nn import Module


class ResidualModule(Module):
    def __init__(
        self,
        module: Module,
        truncated_dim: int | None = None,
        truncate_right: bool | None = None,
    ) -> None:
        """
        A residual module that adds the input to the output of the given module.

        If truncated_dim is specified, the input is truncated to that size
        in the last dimension before being added to the output of the module. If truncate_right is True,
        the rightmost dimensions are truncated; otherwise, the leftmost dimensions
        are truncated.
        """
        super().__init__()
        self.module = module
        self.truncated_dim = truncated_dim
        self.truncate_right = truncate_right

    def forward(self, x):
        if self.truncated_dim is None or x.size(-1) == self.truncated_dim:
            return x + self.module(x)

        if self.truncate_right:
            return x[..., : self.truncated_dim] + self.module(x)
        else:
            return x[..., -self.truncated_dim :] + self.module(x)
