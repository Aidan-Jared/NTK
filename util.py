import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from jax_meta.utils.losses import cross_entropy
from jax_meta.utils.metrics import accuracy
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
    X = jnp.vstack([make_data(key1, n_samples, mean=mean0, sigma=sigma0), 
                   make_data(key2, n_samples, mean=mean1, sigma=sigma1)])
    y = jnp.vstack([y0, y1])

    key3, key = jax.random.split(key1)
    perm = jax.random.permutation(key3, n_samples)

    X = X[perm]
    y = y[perm]

    return X, y

def build_xor_data(key, centers = jnp.array([
            [-1, -1],
            [1, -1],
            [-1, 1],
            [1, 1]
        ]), noise = .3, n_samples = 100):
    cluster_sample_n = n_samples // 4
    cluster_labels = jnp.array([0, 1, 1, 0])
    X = []
    y = []
    for i in range(4):
        subkey1, key = jax.random.split(key)

        cluster_sample = centers[i] + jax.random.normal(
            subkey1, (cluster_sample_n,2)
        ) * noise
        X.append(cluster_sample)
        y.append(jnp.full(cluster_sample_n, cluster_labels[i]))

    X = jnp.vstack(X)
    y = jnp.concat(y)

    subkey, key = jax.random.split(key)
    perm = jax.random.permutation(subkey, n_samples)
    X = X[perm]
    y = y[perm]
    return X, y


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
          ) -> tuple[Float, Float]:
            pred_y = jax.vmap(model)(x)
            ce = cross_entropy(pred_y, y)
            acc = accuracy(pred_y, y)
            return (jnp.mean(ce), acc)