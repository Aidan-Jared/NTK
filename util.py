import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from jax_meta.utils.losses import cross_entropy
from jaxtyping import Array, Float, Int, PyTree, PRNGKeyArray

def make_data(key:PRNGKeyArray, n_samples = 500, mean = jnp.array([1, 1]), sigma = jnp.array([1, 1])):
    return mean + sigma * jax.random.normal(key, shape = (n_samples,2))

def build_binary_dataset(
          key: PRNGKeyArray,
          n_samples: Int = 500,
          mean0 : Array = jnp.array([0, 0]),
          mean1: Array = jnp.array([5, 4]),
          sigma0: Array = jnp.array([1, 2]),
          sigma1: Array = jnp.array([1, 3])
):
    key1, key2 = jax.random.split(key)
    y0 = jnp.zeros([n_samples,1]) 
    y1 = jnp.ones([n_samples,1]) 
    X0 = jnp.hstack([
         make_data(key1, n_samples, mean=mean0, sigma=sigma0), 
         y0
        ])
    X1 = jnp.hstack([
        make_data(key1, n_samples, mean=mean1, sigma=sigma1),
        y1
    ])

    return jnp.vstack([X0, X1])

def NTK(
        model: PyTree,
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
        model: PyTree,
        x: Array,
        y: Array
) -> tuple[Array, Array]:
    @eqx.filter_jit
    def get_loss_grad(x,y):
        def loss(x, y):
            pred_y = model(x)
            return cross_entropy(pred_y, y)
        err, grad = eqx.filter_value_and_grad(loss)(x,y)
        grad_flat, _ = jax.flatten_util.ravel_pytree(grad)
        return err, grad_flat
    err, grad_flat = jax.vmap(get_loss_grad)(x, y)

    return jnp.dot(grad_flat, grad_flat.T), err

def loss(
          model: PyTree,
          x: Array, 
          y: Array
          ):
            pred_y = jax.vmap(model)(x)
            ce = cross_entropy(pred_y, y)
            return jnp.mean(ce)