import equinox as eqx
import jax
import jax.numpy as jnp
import optax
import torch
import torchvision
from jaxtyping import Array, Float, Int, PyTree
from jax_meta.utils.losses import cross_entropy

from cnn import CNN

SEED = 42
BATCH_SIZE = 32

key = jax.random.PRNGKey(SEED)
torch.manual_seed(SEED)


def NTK(
        model: CNN,
        x: Array
):
    @eqx.filter_jit
    def get_jacobian(x):
        j = eqx.filter_jacrev(lambda m: m(x))(model)
        j_flat, _ = jax.flatten_util.ravel_pytree(j)
        return j_flat

    
    jacobians = jax.vmap(get_jacobian)(x)

    test = jnp.dot(jacobians, jacobians.T)

    jnp.linalg.eigh(test)

    return test

def eNTK(
        model: CNN,
        x: Array,
        y: Array
):
    @eqx.filter_jit
    def get_loss_grad(x,y):
        def loss(x, y):
            pred_y = model(x)
            return cross_entropy(pred_y, y)
        err, grad = eqx.filter_value_and_grad(loss)(x,y)
        grad_flat, _ = jax.flatten_util.ravel_pytree(grad)
        return err, grad_flat
    err, loss_grads = jax.vmap(get_loss_grad)(x, y)

    return jnp.dot(loss_grads, loss_grads.T)


if __name__ == "__main__":
    key, subkey = jax.random.split(key, 2)
    model = CNN(subkey)

    normalize_data = torchvision.transforms.Compose(
        [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    train_dataset= torchvision.datasets.MNIST(
        "MNIST",
        train= True,
        download=True,
        transform=normalize_data,
    )

    test_dataset= torchvision.datasets.MNIST(
        "MNIST",
        train= False,
        download=True,
        transform=normalize_data,
    )

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # group NTK by losses jnp.ix_()

    for x, y in trainloader:
        eNTK(model, x.numpy(), y.numpy())