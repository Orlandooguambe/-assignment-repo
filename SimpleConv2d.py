#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SimpleConv2d — scratch CNN 2D em NumPy (um arquivo só)

Inclui:
- Conv2d (NCHW), MaxPool2D, AveragePool2D, Flatten
- Inicializadores (Simple/Xavier/He) e otimizadores (SGD/AdaGrad)
- Ativações (ReLU, Softmax+CrossEntropy combinado)
- Classe Scratch2dCNNClassifier (Conv->ReLU->Pool->Flatten->FC->ReLU->FC->Softmax)
- Funções de tamanho de saída 1D/2D (Problems 2 e 3)
- Testes de arrays pequenos (Problem 2)
- Treino breve no MNIST (Problem 7) com fallback TF->sklearn
- Cálculo dos parâmetros (Problem 10) e explicações (Problem 11)
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # silencia INFO/Warnings do TF, se usado

import numpy as np
import matplotlib.pyplot as plt

# ===========================
# Utilidades: tamanhos de saída
# ===========================

def conv1d_out_len(N_in, F, P=0, S=1):
    """Problem 2 (1D): Nout = (N_in + 2P - F)//S + 1"""
    return (N_in + 2*P - F)//S + 1

def conv2d_out_hw(Nh_in, Nw_in, Fh, Fw, Ph=0, Pw=0, Sh=1, Sw=1):
    """Problem 3 (2D): Nh_out = (Nh_in + 2Ph - Fh)//Sh + 1, idem para W"""
    Nh_out = (Nh_in + 2*Ph - Fh)//Sh + 1
    Nw_out = (Nw_in + 2*Pw - Fw)//Sw + 1
    return Nh_out, Nw_out


# ===========================
# Inicializadores
# ===========================

class SimpleInitializer:
    """Gaussiano simples com desvio padrão sigma"""
    def __init__(self, sigma=0.01):
        self.sigma = float(sigma)
    def W(self, *shape):
        return self.sigma * np.random.randn(*shape)
    def B(self, *shape):
        return np.zeros(shape, dtype=float)

class XavierInitializer:
    """Xavier/Glorot: std = 1/sqrt(n_in)"""
    def __init__(self):
        pass
    def W(self, *shape):
        # shape típico FC: (n_in, n_out), Conv2d: (C_out, C_in, Kh, Kw)
        if len(shape) == 2:
            n_in = shape[0]
        else:
            # conv2d
            C_out, C_in, Kh, Kw = shape
            n_in = C_in * Kh * Kw
        std = 1.0 / np.sqrt(n_in)
        return std * np.random.randn(*shape)
    def B(self, *shape):
        return np.zeros(shape, dtype=float)

class HeInitializer:
    """He: std = sqrt(2/n_in) (adequado para ReLU)"""
    def __init__(self):
        pass
    def W(self, *shape):
        if len(shape) == 2:
            n_in = shape[0]
        else:
            C_out, C_in, Kh, Kw = shape
            n_in = C_in * Kh * Kw
        std = np.sqrt(2.0 / n_in)
        return std * np.random.randn(*shape)
    def B(self, *shape):
        return np.zeros(shape, dtype=float)


# ===========================
# Otimizadores
# ===========================

class SGD:
    """Gradiente descendente simples"""
    def __init__(self, lr=0.01):
        self.lr = float(lr)
    def update(self, layer):
        # layer deve expor dW, dB, W, B
        layer.W -= self.lr * layer.dW
        if layer.B is not None:
            layer.B -= self.lr * layer.dB
        return layer

class AdaGrad:
    """AdaGrad básico"""
    def __init__(self, lr=0.01, eps=1e-8):
        self.lr = float(lr)
        self.eps = float(eps)
        self.cache_W = None
        self.cache_B = None
    def update(self, layer):
        if self.cache_W is None:
            self.cache_W = np.zeros_like(layer.W)
            self.cache_B = np.zeros_like(layer.B) if layer.B is not None else None
        self.cache_W += layer.dW * layer.dW
        layer.W -= self.lr * layer.dW / (np.sqrt(self.cache_W) + self.eps)
        if layer.B is not None:
            self.cache_B += layer.dB * layer.dB
            layer.B -= self.lr * layer.dB / (np.sqrt(self.cache_B) + self.eps)
        return layer


# ===========================
# Ativações
# ===========================

class ReLU:
    def __init__(self):
        self.mask = None
    def forward(self, X):
        self.mask = (X > 0).astype(float)
        return X * self.mask
    def backward(self, dA):
        return dA * self.mask

class SoftmaxWithCE:
    """Softmax + Cross-Entropy combinados (estável numericamente)"""
    def __init__(self):
        self.Z = None  # probs
        self.Y = None  # one-hot alvo
    def forward(self, A, Y_onehot):
        # A: (N, C), Y_onehot: (N, C)
        A_shift = A - A.max(axis=1, keepdims=True)
        expA = np.exp(A_shift)
        Z = expA / expA.sum(axis=1, keepdims=True)
        self.Z = Z
        self.Y = Y_onehot
        # CE média (ou soma). Aqui usaremos média por batch.
        # adicionar eps para estabilidade
        eps = 1e-7
        ce = -np.sum(Y_onehot * np.log(Z + eps)) / A.shape[0]
        return Z, ce
    def backward(self):
        # dA = (Z - Y)/N
        N = self.Y.shape[0]
        return (self.Z - self.Y) / N


# ===========================
# Camadas
# ===========================

class Conv2d:
    """
    Convolução 2D (NCHW):
    W: (C_out, C_in, Kh, Kw)
    B: (C_out,)
    Stride (Sh, Sw), sem padding (para padding, pré-aplique np.pad no input)
    """
    def __init__(self, C_in, C_out, Kh, Kw, initializer, optimizer, Sh=1, Sw=1):
        self.C_in = C_in
        self.C_out = C_out
        self.Kh = Kh
        self.Kw = Kw
        self.Sh = Sh
        self.Sw = Sw
        self.W = initializer.W(C_out, C_in, Kh, Kw)
        self.B = initializer.B(C_out)
        self.optimizer = optimizer
        # caches p/ backward
        self.X = None
        self.dW = None
        self.dB = None

    def forward(self, X):
        # X: (N, C_in, H, W)
        self.X = X
        N, C, H, W = X.shape
        assert C == self.C_in
        H_out, W_out = conv2d_out_hw(H, W, self.Kh, self.Kw, 0, 0, self.Sh, self.Sw)
        A = np.zeros((N, self.C_out, H_out, W_out), dtype=float)
        # Conv "na mão"
        for n in range(N):
            for m in range(self.C_out):
                for i in range(H_out):
                    hs = i * self.Sh
                    for j in range(W_out):
                        ws = j * self.Sw
                        # região: X[n, :, hs:hs+Kh, ws:ws+Kw]
                        A[n, m, i, j] = np.sum(
                            self.X[n, :, hs:hs+self.Kh, ws:ws+self.Kw] * self.W[m, :, :, :]
                        ) + self.B[m]
        return A

    def backward(self, dA):
        # dA: (N, C_out, H_out, W_out)
        N, _, H_out, W_out = dA.shape
        _, _, H, W = self.X.shape
        self.dW = np.zeros_like(self.W)
        self.dB = np.zeros_like(self.B)
        dX = np.zeros_like(self.X)

        # Gradientes
        for n in range(N):
            for m in range(self.C_out):
                for i in range(H_out):
                    hs = i * self.Sh
                    for j in range(W_out):
                        ws = j * self.Sw
                        grad = dA[n, m, i, j]
                        # dW acumula X * grad
                        self.dW[m] += self.X[n, :, hs:hs+self.Kh, ws:ws+self.Kw] * grad
                        # dB acumula grad
                        self.dB[m] += grad
                        # dX acumula W * grad
                        dX[n, :, hs:hs+self.Kh, ws:ws+self.Kw] += self.W[m] * grad

        # atualiza
        self.optimizer.update(self)
        return dX


class MaxPool2D:
    """
    MaxPool 2D (NCHW) com janela (Ph, Pw) e strides (Sh, Sw) (por padrão iguais)
    """
    def __init__(self, Ph=2, Pw=2, Sh=None, Sw=None):
        self.Ph = Ph
        self.Pw = Pw
        self.Sh = Sh if Sh is not None else Ph
        self.Sw = Sw if Sw is not None else Pw
        self.X = None
        self.max_idx = None  # guarda índices do máximo por janela

    def forward(self, X):
        # X: (N, C, H, W)
        self.X = X
        N, C, H, W = X.shape
        H_out, W_out = conv2d_out_hw(H, W, self.Ph, self.Pw, 0, 0, self.Sh, self.Sw)
        A = np.zeros((N, C, H_out, W_out), dtype=float)
        # guardamos a máscara dos máximos para backward
        self.max_idx = np.zeros((N, C, H_out, W_out, 2), dtype=int)  # (pi, pj)

        for n in range(N):
            for k in range(C):
                for i in range(H_out):
                    hs = i * self.Sh
                    for j in range(W_out):
                        ws = j * self.Sw
                        window = X[n, k, hs:hs+self.Ph, ws:ws+self.Pw]
                        pi, pj = np.unravel_index(np.argmax(window), window.shape)
                        A[n, k, i, j] = window[pi, pj]
                        self.max_idx[n, k, i, j] = (hs + pi, ws + pj)
        return A

    def backward(self, dA):
        # dA: (N, C, H_out, W_out)
        N, C, H, W = self.X.shape
        dX = np.zeros_like(self.X)
        H_out, W_out = dA.shape[2], dA.shape[3]
        for n in range(N):
            for k in range(C):
                for i in range(H_out):
                    for j in range(W_out):
                        pi, pj = self.max_idx[n, k, i, j]
                        dX[n, k, pi, pj] += dA[n, k, i, j]
        return dX


class AveragePool2D:
    """Average pooling 2D (NCHW)"""
    def __init__(self, Ph=2, Pw=2, Sh=None, Sw=None):
        self.Ph = Ph
        self.Pw = Pw
        self.Sh = Sh if Sh is not None else Ph
        self.Sw = Sw if Sw is not None else Pw
        self.X = None

    def forward(self, X):
        self.X = X
        N, C, H, W = X.shape
        H_out, W_out = conv2d_out_hw(H, W, self.Ph, self.Pw, 0, 0, self.Sh, self.Sw)
        A = np.zeros((N, C, H_out, W_out), dtype=float)
        for n in range(N):
            for k in range(C):
                for i in range(H_out):
                    hs = i * self.Sh
                    for j in range(W_out):
                        ws = j * self.Sw
                        window = X[n, k, hs:hs+self.Ph, ws:ws+self.Pw]
                        A[n, k, i, j] = np.mean(window)
        return A

    def backward(self, dA):
        N, C, H, W = self.X.shape
        dX = np.zeros_like(self.X)
        H_out, W_out = dA.shape[2], dA.shape[3]
        scale = 1.0 / (self.Ph * self.Pw)
        for n in range(N):
            for k in range(C):
                for i in range(H_out):
                    hs = i * self.Sh
                    for j in range(W_out):
                        ws = j * self.Sw
                        dX[n, k, hs:hs+self.Ph, ws:ws+self.Pw] += dA[n, k, i, j] * scale
        return dX


class Flatten:
    """NCHW -> N x (C*H*W)"""
    def __init__(self):
        self.shape_in = None
    def forward(self, X):
        self.shape_in = X.shape
        N = X.shape[0]
        return X.reshape(N, -1)
    def backward(self, dA):
        return dA.reshape(self.shape_in)


class FC:
    """Camada totalmente conectada"""
    def __init__(self, n_in, n_out, initializer, optimizer, use_bias=True):
        self.W = initializer.W(n_in, n_out)
        self.B = initializer.B(n_out) if use_bias else None
        self.optimizer = optimizer
        self.X = None
        self.dW = None
        self.dB = None

    def forward(self, X):
        self.X = X  # (N, n_in)
        A = X @ self.W  # (N, n_out)
        if self.B is not None:
            A = A + self.B
        return A

    def backward(self, dA):
        # dA: (N, n_out)
        self.dW = self.X.T @ dA  # (n_in, n_out)
        if self.B is not None:
            self.dB = dA.sum(axis=0)
        dX = dA @ self.W.T  # (N, n_in)
        self.optimizer.update(self)
        return dX


# ===========================
# Classificador CNN 2D simples (MNIST)
# ===========================

class Scratch2dCNNClassifier:
    """
    Arquitetura bem simples:
    Conv2d(C_in=1,C_out=8, 3x3) -> ReLU -> MaxPool(2x2)
    Flatten -> FC(8*13*13 -> 64) -> ReLU -> FC(64 -> 10) -> SoftmaxCE
    (Entrada: NCHW = (N,1,28,28))
    """
    def __init__(self, lr=0.01, use_adagrad=True, seed=0):
        np.random.seed(seed)
        opt = AdaGrad(lr=lr) if use_adagrad else SGD(lr=lr)
        init_conv = HeInitializer()   # bom para ReLU
        init_fc   = XavierInitializer()
        self.conv = Conv2d(1, 8, 3, 3, initializer=init_conv, optimizer=opt, Sh=1, Sw=1)
        self.relu1 = ReLU()
        self.pool = MaxPool2D(2, 2)  # 28->(28-2)/2+1? Com stride 2 sem padding => 28->14
        self.flat = Flatten()
        self.fc1 = FC(8*13*13, 64, initializer=init_fc, optimizer=opt)  # ver cálculo abaixo
        self.relu2 = ReLU()
        self.fc2 = FC(64, 10, initializer=init_fc, optimizer=opt)
        self.sm = SoftmaxWithCE()

    def _forward(self, X):
        # Conv: H_out,W_out = (28-3)//1+1 = 26; depois MaxPool 2x2 stride 2 -> 13x13
        A = self.conv.forward(X)          # (N,8,26,26)
        Z = self.relu1.forward(A)
        P = self.pool.forward(Z)          # (N,8,13,13)
        F = self.flat.forward(P)          # (N, 8*13*13)
        A1 = self.fc1.forward(F)          # (N,64)
        Z1 = self.relu2.forward(A1)
        A2 = self.fc2.forward(Z1)         # (N,10)
        return A, Z, P, F, A1, Z1, A2

    def _backward(self, caches, Y_onehot):
        A, Z, P, F, A1, Z1, A2 = caches
        # Softmax+CE
        probs, ce = self.sm.forward(A2, Y_onehot)
        dA2 = self.sm.backward()          # (N,10)
        dZ1 = self.fc2.backward(dA2)      # (N,64)
        dA1 = self.relu2.backward(dZ1)    # (N,64)
        dF  = self.fc1.backward(dA1)      # (N,8*13*13)
        dP  = self.flat.backward(dF)      # (N,8,13,13)
        dZ  = self.pool.backward(dP)      # (N,8,26,26)
        dA  = self.relu1.backward(dZ)     # (N,8,26,26)
        _   = self.conv.backward(dA)      # (N,1,28,28) grad p/ camadas anteriores (não usado)
        return ce, probs

    def fit(self, X, y_onehot, X_val=None, y_val_onehot=None, epochs=3, batch_size=64, verbose=True):
        N = X.shape[0]
        hist = {"train_ce": [], "train_acc": [], "val_ce": [], "val_acc": []}
        idx_all = np.arange(N)
        for ep in range(1, epochs+1):
            np.random.shuffle(idx_all)
            ce_sum, correct = 0.0, 0
            nb = 0
            for start in range(0, N, batch_size):
                nb += 1
                end = min(start+batch_size, N)
                bidx = idx_all[start:end]
                Xb = X[bidx]
                Yb = y_onehot[bidx]
                caches = self._forward(Xb)
                ce, probs = self._backward(caches, Yb)
                ce_sum += ce
                correct += (np.argmax(probs, axis=1) == np.argmax(Yb, axis=1)).sum()

            train_ce = ce_sum / nb
            train_acc = correct / N
            hist["train_ce"].append(train_ce)
            hist["train_acc"].append(train_acc)

            if X_val is not None and y_val_onehot is not None:
                val_probs = self.predict_proba(X_val, batch_size=batch_size)
                eps = 1e-7
                val_ce = -np.sum(y_val_onehot * np.log(val_probs + eps)) / X_val.shape[0]
                val_acc = (np.argmax(val_probs, axis=1) == np.argmax(y_val_onehot, axis=1)).mean()
                hist["val_ce"].append(val_ce)
                hist["val_acc"].append(val_acc)
                if verbose:
                    print(f"Epoch {ep:02d}/{epochs} | train CE {train_ce:.3f} acc {train_acc:.3f} | "
                          f"val CE {val_ce:.3f} acc {val_acc:.3f}", flush=True)
            else:
                if verbose:
                    print(f"Epoch {ep:02d}/{epochs} | train CE {train_ce:.3f} acc {train_acc:.3f}", flush=True)
        return hist

    def predict_proba(self, X, batch_size=256):
        probs_all = []
        for start in range(0, X.shape[0], batch_size):
            end = min(start+batch_size, X.shape[0])
            _, _, _, _, _, _, A2 = self._forward(X[start:end])
            # apenas forward de softmax (sem gravar) para probabilidade:
            A_shift = A2 - A2.max(axis=1, keepdims=True)
            expA = np.exp(A_shift)
            probs = expA / expA.sum(axis=1, keepdims=True)
            probs_all.append(probs)
        return np.vstack(probs_all)

    def predict(self, X, batch_size=256):
        return np.argmax(self.predict_proba(X, batch_size=batch_size), axis=1)


# ===========================
# Helpers: dados MNIST (TF -> sklearn fallback), one-hot, normalização
# ===========================

def load_mnist_2d_nchw(limit_train=None, limit_val=None, limit_test=None, seed=0):
    """
    Retorna X_train, y_train_onehot, X_val, y_val_onehot, X_test, y_test_onehot (N,1,28,28)
    """
    rng = np.random.default_rng(seed)
    X_train = y_train = X_test = y_test = None

    # 1) Tente tensorflow.keras
    try:
        from tensorflow.keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    except Exception:
        # 2) Fallback: sklearn openml
        try:
            from sklearn.datasets import fetch_openml
            mn = fetch_openml('mnist_784', version=1, as_frame=False)
            X = mn.data.reshape(-1, 28, 28)
            y = mn.target.astype(np.int64)
            # split 60k/10k (padrão MNIST)
            X_train, y_train = X[:60000], y[:60000]
            X_test, y_test = X[60000:], y[60000:]
        except Exception as e:
            raise RuntimeError(f"Não foi possível carregar MNIST: {e}")

    # normaliza [0,1] e reshape para NCHW
    X_train = X_train.astype(np.float32) / 255.0
    X_test  = X_test.astype(np.float32) / 255.0
    # split val 20% do treino
    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=seed, stratify=y_train)

    # aplica limites (para rodar mais rápido)
    if limit_train: 
        idx = rng.permutation(X_tr.shape[0])[:limit_train]
        X_tr, y_tr = X_tr[idx], y_tr[idx]
    if limit_val:
        idx = rng.permutation(X_val.shape[0])[:limit_val]
        X_val, y_val = X_val[idx], y_val[idx]
    if limit_test:
        idx = rng.permutation(X_test.shape[0])[:limit_test]
        X_test, y_test = X_test[idx], y_test[idx]

    # NCHW
    X_tr  = X_tr.reshape(-1, 1, 28, 28)
    X_val = X_val.reshape(-1, 1, 28, 28)
    X_test= X_test.reshape(-1, 1, 28, 28)

    # one-hot
    def one_hot(y, num_classes=10):
        oh = np.zeros((y.shape[0], num_classes), dtype=float)
        oh[np.arange(y.shape[0]), y] = 1.0
        return oh

    y_tr_oh  = one_hot(y_tr)
    y_val_oh = one_hot(y_val)
    y_te_oh  = one_hot(y_test)
    return X_tr, y_tr_oh, X_val, y_val_oh, X_test, y_te_oh, y_tr, y_val, y_test


# ===========================
# Testes e execução
# ===========================

def test_problem2_small_arrays():
    # Conforme enunciado do Problem 2 (2D conv forward/backward)
    # x: (1,1,4,4)
    x = np.array([[[[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9,10,11,12],
                    [13,14,15,16]]]]).astype(float)
    # w: (2,1,3,3) — no enunciado listaram (2,3,3) assumindo 1 canal de entrada
    w_kernels = np.array([
        [[0.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0,-1.0, 0.0]],
        [[0.0, 0.0, 0.0],
         [0.0,-1.0, 1.0],
         [0.0, 0.0, 0.0]],
    ]).reshape(2,1,3,3)
    # Construir uma camada Conv2d com pesos fixos
    opt = SGD(lr=0.0)  # não atualiza
    conv = Conv2d(C_in=1, C_out=2, Kh=3, Kw=3, initializer=SimpleInitializer(0.0), optimizer=opt, Sh=1, Sw=1)
    conv.W = w_kernels.copy()
    conv.B = np.zeros(2)

    a = conv.forward(x)
    print("\n== Problem 2: small Conv2d forward ==")
    print(a.astype(int))  # esperado: [[[-4,-4],[-4,-4]], [[1,1],[1,1]]]

    # Um delta arbitrário para testar backward (shape igual a a): (1,2,2,2)
    delta = np.array([[[[-4, -4],
                        [10, 11]],
                       [[ 1, -7],
                        [ 1,-11]]]], dtype=float)
    dX = conv.backward(delta)  # dW/dB também populados
    # Se tiver padding no enunciado, eles somaram certas bordas; aqui, mostramos dB:
    print("\nConv2d dB (soma de deltas por saída):", conv.dB.astype(int))
    # e parte do dX central (apenas para referência)
    print("Conv2d dX (shape):", dX.shape)

def problem10_params_calc():
    """
    Calcula tamanhos e #parâmetros para 3 camadas pedidas:
    1) In: 144x144, Cin=3 | F: 3x3, Cout=6 | stride 1, no padding
       Out: H=(144-3)//1+1=142, W=142
       Params: W=(Cout*Cin*3*3)=6*3*9=162  + B=6  => 168
    2) In: 60x60, Cin=24 | F:3x3, Cout=48 | s=1, no pad
       Out: 58x58
       Params: 48*24*9=10368 + 48 = 10416
    3) In: 20x20, Cin=10 | F:3x3, Cout=20 | stride=2, no pad
       Out: H=(20-3)//2+1 = (17)//2+1 = 8+1=9, W=9
       Params: 20*10*9=1800 + 20 = 1820
    """
    H1, W1 = conv2d_out_hw(144,144,3,3,0,0,1,1)
    H2, W2 = conv2d_out_hw(60,60,3,3,0,0,1,1)
    H3, W3 = conv2d_out_hw(20,20,3,3,0,0,2,2)
    p1 = 6*3*3*3 + 6
    p2 = 48*24*3*3 + 48
    p3 = 20*10*3*3 + 20
    print("\n== Problem 10: output sizes & params ==")
    print(f"1) Out: {H1}x{W1}, params: {p1}")
    print(f"2) Out: {H2}x{W2}, params: {p2}")
    print(f"3) Out: {H3}x{W3}, params: {p3}")

def problem11_explanations():
    print("\n== Problem 11: filter size notes ==")
    print("- Por que 3x3 é tão usado?\n"
          "  * Empilha-se várias camadas 3x3 para obter campo receptivo grande, com menos parâmetros do que usar 7x7 direto.\n"
          "  * Mais não-linearidades (ReLU) entre filtros menores ajudam a expressividade.\n"
          "  * 3x3 mantém custo/memória sob controle e funciona muito bem na prática (VGG popularizou).\n")
    print("- Para que serve 1x1?\n"
          "  * Mistura/‘projeta’ canais sem olhar vizinhança espacial, atuando como um FC por pixel.\n"
          "  * Reduz/expande dimensão de canais (gargalo) e introduz não-linearidade barata entre convs maiores (ex.: Inception/ResNet).")

def quick_plot_history(hist):
    if not hist: 
        return
    epochs = len(hist["train_ce"])
    it = np.arange(1, epochs+1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(it, hist["train_ce"], label="train CE")
    if len(hist["val_ce"]) == epochs:
        plt.plot(it, hist["val_ce"], label="val CE")
    plt.xlabel("epoch"); plt.ylabel("cross-entropy"); plt.legend(); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(it, hist["train_acc"], label="train acc")
    if len(hist["val_acc"]) == epochs:
        plt.plot(it, hist["val_acc"], label="val acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.show(block=True)


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    # Problem 1/2/3 sanity prints
    print("== Problem 2 (1D size) quick check ==")
    print("N_in=4, F=3, P=0, S=1 ->", conv1d_out_len(4,3,0,1))

    test_problem2_small_arrays()
    problem10_params_calc()
    problem11_explanations()

    # Problem 7: treino curto no MNIST (para comprovar funcionamento)
    print("\n== Problem 7: train Scratch2dCNNClassifier on MNIST (short run) ==")
    # Para rodar rápido em notebook/PC modesto, subamostra um pouco:
    X_tr, y_tr_oh, X_val, y_val_oh, X_te, y_te_oh, y_tr, y_val, y_te = load_mnist_2d_nchw(
        limit_train=10000, limit_val=2000, limit_test=2000, seed=0
    )
    clf = Scratch2dCNNClassifier(lr=0.02, use_adagrad=True, seed=0)
    hist = clf.fit(X_tr, y_tr_oh, X_val, y_val_oh, epochs=5, batch_size=64, verbose=True)

    # Avaliação
    val_probs = clf.predict_proba(X_val)
    val_acc = (np.argmax(val_probs, axis=1) == np.argmax(y_val_oh, axis=1)).mean()
    te_probs = clf.predict_proba(X_te)
    te_acc = (np.argmax(te_probs, axis=1) == np.argmax(y_te_oh, axis=1)).mean()
    print(f"Validation accuracy: {val_acc:.3f} | Test accuracy: {te_acc:.3f}")

    quick_plot_history(hist)
