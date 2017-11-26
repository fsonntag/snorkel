from __future__ import print_function
import torch
import numpy as np
import scipy.sparse
import timeit


def plsa(count, n_dim, n_iter, learning_rate=1e-3):
    n_terms = count.shape[0]
    nnz = count._nnz()

    count = count.to_dense()
    count = torch.autograd.Variable(count)

    L = np.random.rand(n_terms, n_dim)
    L /= L.sum(axis=1, keepdims=True)
    L = torch.FloatTensor(L)
    L = torch.autograd.Variable(L, requires_grad=True)

    for t in range(n_iter):
        # Forward pass: compute predicted y using operations on Variables; these
        # are exactly the same operations we used to compute the forward pass using
        # Tensors, but we do not need to keep references to intermediate values since
        # we are not implementing the backward pass by hand.
        count_pred = L.mm(L.t())

        # Compute and print loss using operations on Variables.
        # Now loss is a Variable of shape (1,) and loss.data is a Tensor of shape
        # (1,); loss.data[0] is a scalar value holding the loss.
        loss = (count_pred - count).pow(2).sum()
        if t % 100 == 0:
            print(t, loss.data[0] / n_terms ** 2)

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Variables with requires_grad=True.
        # After this call w1.grad and w2.grad will be Variables holding the gradient
        # of the loss with respect to w1 and w2 respectively.
        loss.backward()

        # Update weights using gradient descent; w1.data and w2.data are Tensors,
        # w1.grad and w2.grad are Variables and w1.grad.data and w2.grad.data are
        # Tensors.
        L.data -= learning_rate * L.grad.data

        # Manually zero the gradients after updating weights
        L.grad.data.zero_()

    return L


def plsa_sparse(count, n_dim, n_iter, M=None, learning_rate=1e-3, batch_size=100, negatives=10):
    n_terms = count.shape[0]
    nnz = count._nnz()

    batch_size = min(batch_size, nnz + negatives)
    negatives = min(negatives, batch_size)

    if M is None:
        M = np.random.rand(n_terms, n_dim)
        M /= M.sum(axis=1, keepdims=True)
        M = torch.FloatTensor(M)
        M = torch.autograd.Variable(M, requires_grad=True)

    for t in range(n_iter):
        # Gather mini-batch
        batch = torch.LongTensor(np.random.choice(nnz, batch_size - negatives, replace=False))
        i = torch.cat((count._indices()[0][batch], torch.LongTensor(np.random.randint(n_terms, size=negatives))))
        j = torch.cat((count._indices()[1][batch], torch.LongTensor(np.random.randint(n_terms, size=negatives))))
        L = M[i][:]
        R = M[j][:]
        C = torch.cat((torch.FloatTensor(count._values()[batch]), torch.FloatTensor(negatives).zero_()))
        C = torch.autograd.Variable(C, requires_grad=False)

        # Forward pass: compute predicted y using operations on Variables; these
        # are exactly the same operations we used to compute the forward pass using
        # Tensors, but we do not need to keep references to intermediate values since
        # we are not implementing the backward pass by hand.
        C_pred = (L * R).sum(1)

        # Compute and print loss using operations on Variables.
        # Now loss is a Variable of shape (1,) and loss.data is a Tensor of shape
        # (1,); loss.data[0] is a scalar value holding the loss.
        loss = (C_pred - C).pow(2).sum()
        if t % 100 == 0:
            print(t, loss.data[0] / batch_size)
            # print((M.data.mm(M.data.t()) - count.to_dense()).pow(2).sum() / n_terms ** 2)
            # print()

        # Use autograd to compute the backward pass. This call will compute the
        # gradient of loss with respect to all Variables with requires_grad=True.
        # After this call w1.grad and w2.grad will be Variables holding the gradient
        # of the loss with respect to w1 and w2 respectively.
        loss.backward()

        # Update weights using gradient descent; w1.data and w2.data are Tensors,
        # w1.grad and w2.grad are Variables and w1.grad.data and w2.grad.data are
        # Tensors.
        M.data -= learning_rate * M.grad.data

        # Manually zero the gradients after updating weights
        M.grad.data.zero_()

    return M


def init_matrix(n_terms, sparsity):
    m = int(n_terms ** 2 * sparsity)

    ind = torch.LongTensor(2, m)
    ind[0] = torch.LongTensor(np.random.choice(n_terms, m))
    ind[1] = torch.LongTensor(np.random.choice(n_terms, m))
    v = torch.FloatTensor(np.ones(m, np.int64))

    count = torch.sparse.FloatTensor(ind, v, torch.Size([n_terms, n_terms]))
    count = count.coalesce()
    return count


def main():
    n_terms, sparsity = 1000, 0.01

    count = init_matrix(n_terms, sparsity)

    M = None
    for lr in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        M = plsa_sparse(count, 5, 1000, M, lr)


if __name__ == "__main__":
    main()
