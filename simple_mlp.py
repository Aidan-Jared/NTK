import equinox as eqx
import jax
from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray


class MLP(eqx.Module):
    layers: list
    def __init__(
            self,
            key: PRNGKeyArray,
            input_dim: Int,
            output_dim: Int
    ):
        key1, key2 = jax.random.split(key)
        self.layers = [
            eqx.nn.Linear(input_dim, output_dim, key=key1),
            jax.nn.sigmoid,
            ]
    def __call__(
            self,
            x: Float[Array, " batch input_dim"]
    ) -> Float[Array, "batch output_dim"]:
        for layer in self.layers:
            x = layer(x)
        return x