import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from torch import manual_seed, tensor
from jaxtyping import Array, Float, Int, PyTree

from cnn import MLP

from util import build_binary_dataset, build_xor_data, NTK, eNTK, trNTK, loss

import tqdm as tqdm

import polars as pl

SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
LR = .01

key = jax.random.PRNGKey(SEED)
manual_seed(SEED)

@eqx.filter_jit
def FGSM(
        model: PyTree,
        x: Array,
        y: Array
) ->tuple[Array, Array]:
    epsilon = .03

    def loss_fn(x):
        y_pred = jax.vmap(model)(x)
        one_hot = jax.nn.one_hot(y, num_classes=2)
        loss = -jnp.sum(one_hot * jax.nn.log_softmax(y_pred))
        return loss

    grad = eqx.filter_grad(loss_fn)(x)

    x_adv = x + epsilon * jnp.sign(grad)

    pred_clean = jnp.argmax(jax.vmap(model)(x), axis=1)

    pred_adv = jnp.argmax(jax.vmap(model)(x_adv), axis=1)
    is_adv = pred_clean != pred_adv
    
    return x_adv, is_adv


# @eqx.filter_jit
def train(
        model: PyTree,
        trainloader: DataLoader,
        testloader: DataLoader,
        optim: optax.GradientTransformation,
        opt_state:PyTree,
        NTK_steps: Int
):
    step = 0
    NTK_test = []
    NTK_adv = []
    errs = []
    train_losses = []
    train_acc = []
    test_losses = []
    test_acc = []

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
        model, opt_state, train_loss, acc = make_step(model, opt_state, x, y)
        train_losses.append(train_loss.item())
        train_acc.append(acc.item())
    
    for (x,y) in testloader:
        x = x.numpy()
        y = y.numpy()
        test_loss, acc = eqx.filter_jit(loss)(model, x, y)
        test_losses.append(test_loss.item())
        test_acc.append(acc.item())
        x_adv, is_adv = FGSM(model, x, y)
        if jnp.sum(is_adv) > 0:
            trNTK_jit = eqx.filter_jit(trNTK)
            kernal_adv = trNTK_jit(model, x, x_adv, jax.random.PRNGKey(42))
            kernal = trNTK_jit(model, x, x, jax.random.PRNGKey(42))
            NTK_adv.append(kernal_adv)
            NTK_test.append(kernal)
        # NTK_m= eNTK(model, x, y)
    return model, opt_state, (NTK_adv, NTK_test), errs, train_losses, test_losses, train_acc, test_acc
 

def main():
    key1, key2  = jax.random.split(key, 2)
    # X, y = build_binary_dataset(key1, n_samples=1000)
    X, y = build_xor_data(key1, n_samples=2000)
    
    split = int(X.shape[0] * .8)
    train_data = X[:split]
    train_target = y[:split]
    test_data = X[split:]
    test_target = y[split:]

    train_data = TensorDataset(tensor(np.array(train_data)), tensor(np.array(train_target)))
    test_data = TensorDataset(tensor(np.array(test_data)), tensor(np.array(test_target)))
    trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = MLP(key2)


    optim = optax.sgd(LR)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    errs = []
    train_losses = []
    train_acces = []
    test_losses = []
    test_acces = []
    results = pl.DataFrame()

    for epoch in tqdm.tqdm(range(EPOCHS)):
        model, opt_state, NTKr, err, train_loss, test_loss, train_acc, test_acc = train(model, trainloader, testloader, optim, opt_state, 5)
        errs.extend(err)
        train_losses.extend(train_loss)
        train_acces.extend(train_acc)
        test_losses.extend(test_loss)
        test_acces.extend(test_acc)

        res = pl.from_dict(
            {
                "epoch" : epoch,
                "train_loss" : np.mean(train_loss).item(),
                "train_acces" : np.mean(train_acces).item(),
                "test_losses" : np.mean(test_losses).item(),
                "test_acces" : np.mean(test_acces).item(),
                "NTK_adv" : str(NTKr[0]),
                "NTK_test" : str(NTKr[1])
            }
        )

        results = pl.concat([results, res])

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

    results.write_parquet("NTK_Data.parquet")


if __name__ == "__main__":
    main()
