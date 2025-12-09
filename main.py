import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from torch import manual_seed, tensor
from jaxtyping import Array, Float, Int, PyTree

from util import build_binary_dataset, NTK, eNTK, loss

import tqdm as tqdm

SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4

key = jax.random.PRNGKey(SEED)
manual_seed(SEED)


eqx.filter_jit
def train(
        model: PyTree,
        trainloader: DataLoader,
        testloader: DataLoader,
        optim: optax.GradientTransformation,
        opt_state:PyTree,
        NTK_steps: Int
):
    step = 0
    NTKs = []
    errs = []
    train_losses = []
    test_losses = []

    @eqx.filter_jit
    def make_step(
        model: PyTree,
        opt_state: PyTree,
        x: Float[Array, " batch in_features"],
        y: Float[Array, " batch 1"]
    ):
        loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model,  opt_state, loss_value

    for (x, y) in trainloader:
        step += 1
        x = x.numpy()
        y = y.numpy()
        if step % NTK_steps == 0:
            NTK_m, err = eNTK(model, x, y)
            NTKs.append(NTK_m)
            errs.append(err)
        model, opt_state, train_loss = make_step(model, opt_state, x, y)
        train_losses.append(train_loss)
    
    for (x,y) in testloader:
        x = x.numpy()
        y = y.numpy()
        test_loss = eqx.filter_jit(loss)(model, x, y)
        test_losses.append(test_loss)
    return model, opt_state, NTKs, errs, jnp.mean(jnp.array(train_losses)), jnp.mean(jnp.array(test_losses))
 

def main():
    key1, key2, key3 = jax.random.split(key, 3)
    data = build_binary_dataset(key1, n_samples=1000)
    data = jax.random.permutation(key2, data)
    
    split = int(data.shape[0] * .8)
    train_data = data[:split]
    test_data = data[split:]

    train_data = TensorDataset(tensor(np.array(train_data[:,0:2])), tensor(np.array(train_data[:,2], dtype=np.int8)))
    test_data = TensorDataset(tensor(np.array(test_data[:,0:2])), tensor(np.array(test_data[:,2], dtype=np.int8)))
    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = eqx.nn.MLP(in_size=2, out_size=2, width_size=2, depth=1, key=key3, final_activation=jax.nn.sigmoid)
    optim = optax.sgd(LR)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    NTKs = []
    errs = []

    for epoch in tqdm.tqdm(range(EPOCHS)):
        model, opt_state, NTKr, err, train_loss, test_loss = train(model, trainloader, testloader, optim, opt_state, 5)
        NTKs.extend(NTKr)
        errs.extend(err)
        print(f"{epoch}: train loss = {train_loss.item()} test loss = {test_loss.item()}")




if __name__ == "__main__":
    main()
