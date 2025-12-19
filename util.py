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

def build_xor_data(
          key : PRNGKeyArray, 
          centers: Array = jnp.array([
            [-1, -1],
            [1, -1],
            [-1, 1],
            [1, 1]
        ]), 
        noise: Float = .5, 
        n_samples: Int = 100) -> tuple[Array, Array]:
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

    subkey1, subkey2 = jax.random.split(key)
    # noise_idx = jax.random.choice(subkey1, a= n_samples, shape = (int(n_samples * noise),), replace=False)
    # y = y.at[noise_idx].set(1 - y[noise_idx])

    perm = jax.random.permutation(subkey2, n_samples)
    X = X[perm]
    y = y[perm]
    return X, y

def random_poision(
          key:PRNGKeyArray, 
          y:Array, 
          alpha:Float
          ) -> Array:
    n = int(y.shape[0] * alpha)
    poision_idx = jax.random.permutation(key, y.shape[0])[:n]
    y = y.at[poision_idx].set(jnp.invert(y[poision_idx].astype(bool)).astype(int))
    return y

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
) -> Array:
    @eqx.filter_jit
    def get_loss_grad(x,y):
        def loss(model, x, y):
            return cross_entropy(model(x), y)
        grad = eqx.filter_jacrev(lambda m: loss(m, x, y))(model)
        grad_flat, _ = jax.flatten_util.ravel_pytree(grad)
        return grad_flat
    grad_flat = jax.vmap(get_loss_grad)(x, y)

    return jnp.dot(grad_flat, grad_flat.T)

def trNTK(
        model: PyTree,
        x_i: Array,
        x_j: Array,
        key : PRNGKeyArray,
        project_dim: Int = 100
):
    
    N = 2
    params, static = eqx.partition(model, eqx.is_array)
    @eqx.filter_jit
    def get_loss_grad(x,yidx):
        def loss(params, x, yidx):
            model = eqx.combine(params, static)
            pred_y = model(x)
            return pred_y[yidx]
        grad = jax.grad(loss)(params, x, yidx)
        grad_flat, _ = jax.flatten_util.ravel_pytree(grad)
        return grad_flat
    
    yidx = jnp.arange(N)

    grad_class = jax.vmap(get_loss_grad, (None, 0))
    grads_class = jax.vmap(grad_class, in_axes=(0, None))
    grads_i = grads_class(x_i, yidx)
    grads_j = grads_class(x_j, yidx)
    
    proj_matrix = jax.random.normal(key, (grads_i.shape[2], project_dim), dtype=jnp.float32)
    proj_grads_i = (grads_i @ proj_matrix)
    proj_grads_j = (grads_j @ proj_matrix)

    kernal = jnp.einsum("ick, jck->ij", proj_grads_i, proj_grads_j)

    norms_i = jnp.sqrt(jnp.einsum("ick,ick->i", proj_grads_i, proj_grads_i))
    norms_j = jnp.sqrt(jnp.einsum("ick,ick->i", proj_grads_j, proj_grads_j))

    kernal = kernal / jnp.outer(norms_i, norms_j)

    return kernal


def loss(
          model: PyTree,
          x: Array, 
          y: Array
          ) -> tuple[Float, Float]:
            pred_y = jax.vmap(model)(x)
            ce = cross_entropy(pred_y, y)
            acc = accuracy(pred_y, y)
            return (jnp.mean(ce), acc)