import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from torch import manual_seed, tensor
from jaxtyping import Array, Float, Int, PyTree

from util import build_binary_dataset, build_xor_data, NTK, eNTK, loss

import tqdm as tqdm

SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
LR = .1

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
    train_acc = []
    test_losses = []
    test_acc = []

    @eqx.filter_jit
    def make_step(
        model: PyTree,
        opt_state: PyTree,
        x: Float[Array, " batch in_features"],
        y: Float[Array, " batch 1"]
    ):
        (loss_value, acc), grads = eqx.filter_value_and_grad(loss, has_aux=True)(model, x, y)
        updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model,  opt_state, loss_value, acc

    for (x, y) in trainloader:
        step += 1
        x = x.numpy()
        y = y.numpy()
        if step % NTK_steps == 0:
            NTK_m, err = eNTK(model, x, y)
            NTKs.append(NTK_m)
            errs.append(err)
        model, opt_state, train_loss, acc = make_step(model, opt_state, x, y)
        train_losses.append(train_loss)
        train_acc.append(acc)
    
    for (x,y) in testloader:
        x = x.numpy()
        y = y.numpy()
        test_loss, acc = eqx.filter_jit(loss)(model, x, y)
        test_losses.append(test_loss)
        test_acc.append(acc)
    
    return model, opt_state, NTKs, errs, train_losses, test_losses, train_acc, test_acc
 

def main():
    key1, key2  = jax.random.split(key, 2)
    # X, y = build_binary_dataset(key1, n_samples=1000)
    X, y = build_xor_data(key1, n_samples=1000)
    
    split = int(X.shape[0] * .8)
    train_data = X[:split]
    train_target = y[:split]
    test_data = X[split:]
    test_target = y[split:]

    train_data = TensorDataset(tensor(np.array(train_data)), tensor(np.array(train_target)))
    test_data = TensorDataset(tensor(np.array(test_data)), tensor(np.array(test_target)))
    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = eqx.nn.MLP(in_size=2, out_size=2, width_size=2, depth=2, key=key2, activation=jax.nn.tanh, final_activation=jax.nn.sigmoid)
    optim = optax.sgd(LR)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    NTKs = []
    errs = []
    train_losses = []
    train_acces = []
    test_losses = []
    test_acces = []

    for epoch in tqdm.tqdm(range(EPOCHS)):
        model, opt_state, NTKr, err, train_loss, test_loss, train_acc, test_acc = train(model, trainloader, testloader, optim, opt_state, 5)
        NTKs.extend(NTKr)
        errs.extend(err)
        train_losses.extend(train_loss)
        train_acces.extend(train_acc)
        test_losses.extend(test_loss)
        test_acces.extend(test_acc)

        if epoch % 10 ==0:
            train_loss = sum(train_losses) / len(train_losses)
            train_acc = sum(train_acces) / len(train_acces)
            test_loss = sum(test_losses) / len(test_losses)
            test_acc = sum(test_acces) / len(test_acces)
            print(f"{epoch}: train loss = {train_loss}, train acc = {train_acc}, test loss = {test_loss}, test acc = {test_acc}")
            train_losses = []
            train_acces = []
            test_losses = []
            test_acces = []




if __name__ == "__main__":
    main()
