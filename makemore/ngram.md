
# Table of Contents

1.  [Introduction](#orgde38c09)
2.  [Preliminaries](#org25dfa55)
3.  [Directly estimating the probabilities](#orgcc7a24c)
    1.  [Building the lookup tables](#orge27b669)
    2.  [Building the probability distribution over each key](#orgd25319f)
    3.  [Iteratively sampling to generate fake names](#orgfaaf2e1)
4.  [A cheap neural network approach](#org6c02f76)
    1.  [The forward pass](#org5031d1c)
    2.  [Sampling from the model](#orgdac9571)
    3.  [The training loop](#org7caec03)
    4.  [How did our losses change?](#org6c4e525)
    5.  [Extra: how low can the loss get?](#org1b22cdd)



<a id="orgde38c09"></a>

# Introduction

This follows Andrej Karpathy&rsquo;s `makemore` videos.


<a id="org25dfa55"></a>

# Preliminaries

    import math
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import torch
    
    torch.manual_seed(0)
    
    def get_names():
        return open('names.txt', 'r').read().splitlines()
    
    names = get_names()
    names[:5]

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">emma</td>
<td class="org-left">olivia</td>
<td class="org-left">ava</td>
<td class="org-left">isabella</td>
<td class="org-left">sophia</td>
</tr>
</tbody>
</table>


<a id="orgcc7a24c"></a>

# Directly estimating the probabilities


<a id="orge27b669"></a>

## Building the lookup tables

These will allow us to convert the n-character context into an integer and back,
and similarly for the 1-character prediction from the context, which we will use to
put them into a matrix later.

    ngram_n = 2
    
    def augment(n, word):
        return '.'*(n-1) + word + '.'*(n-1)
    
    def split_kv(i, n, word):
        a = augment(n, word)
        key = a[i:i+n-1]
        val = a[i+n-1]
        # print(i, i+n-1, word, a, key, val)
        return (key, val)
    
    def split_kv_index(i, n, word):
        k, v = split_kv(i, n, word)
        return (ktoi[k], vtoi[v])
    
    def all_splits(n, word):
        ks = []
        vs = []
        for i in range(len(word) + 1):
            k, v = split_kv(i, n, word)
            ks.append(k)
            vs.append(v)
        return (ks, vs)
    
    def ngrams_of(n, word):
        out = dict()
        ks, vs = all_splits(n, word)
        for k, v in zip(ks, vs):
            out[k] = v
        return out
    
    all_ngrams = list(ngrams_of(ngram_n, name) for name in names)
    
    def gen_maps(g):
        r = sorted(list(set(g)))
        rtoi = {z:i for i,z in enumerate(r)}
        itor = {i:z for z,i in rtoi.items()}
        nr = len(r)
        return (rtoi, itor, nr)
    
    (ktoi, itok, nk) = gen_maps(k for d in all_ngrams for k in d.keys())
    (vtoi, itov, nv) = gen_maps(k for d in all_ngrams for k in d.values())
    
    print(nk, nv)

    27 27

Now that we have our keys and values, we can build a Torch tensor of frequencies:

    N = torch.zeros((nk, nv), dtype=torch.int32)
    for word in names:
        for i in range(len(word) + 1):
            ik, iv = split_kv_index(i, ngram_n, word)
            N[ik, iv] += 1


<a id="orgd25319f"></a>

## Building the probability distribution over each key

    P = N.float()
    P = P / P.sum(1, keepdims=True)
    P[0]

    tensor([0.0000, 0.1377, 0.0408, 0.0481, 0.0528, 0.0478, 0.0130, 0.0209, 0.0273,
            0.0184, 0.0756, 0.0925, 0.0491, 0.0792, 0.0358, 0.0123, 0.0161, 0.0029,
            0.0512, 0.0642, 0.0408, 0.0024, 0.0117, 0.0096, 0.0042, 0.0167, 0.0290])

Does each row really represent a probability distribution?

    P.sum(1)[0:10]

    tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
            1.0000])

Note that this isn&rsquo;t really necessary: `torch.multinomial` works fine with things
that don&rsquo;t sum to $1$.


<a id="orgfaaf2e1"></a>

## Iteratively sampling to generate fake names

    torch.manual_seed(0)
    gens = []
    for i in range(10):
        out = '.' * (ngram_n - 1)
        n = 0
        z = None
        while True:
            z = P[n]
            new_n = torch.multinomial(P[n], 1, True).data.item()
            n = new_n
            if n == 0:
                break
            out += itov[n]
            n = ktoi[out[1-ngram_n:]]
        gens.append(out[ngram_n-1:])
    print(gens)

    ['bhrirerirco', 'maren', 'f', 'lem', 'a', 'vieiynana', 'sa', 'l', 'merlershonin', 'aytty']


<a id="org6c02f76"></a>

# A cheap neural network approach

One-layer neural network: we take our $N_k$ keys and one-hot them, then feed
them into a $N_k \times N_v$ matrix whose output we softmax to produce a
probability distribution over the values.

First, let&rsquo;s re-seed the random number generator, so we can deterministically re-execute the rest of this buffer without
having to recompute everything higher up.

    import torch.nn.functional as F
    torch.manual_seed(0)

    <torch._C.Generator at 0x7f36a5b65890>

    def training_set():
        xs = []
        ys = []
        for w in names:
            ks, vs = all_splits(ngram_n, w)
            x = list(ktoi[k] for k in ks)
            y = list(vtoi[v] for v in vs)
            xs.extend(x)
            ys.extend(y)
        return (torch.tensor(xs), torch.tensor(ys))
    
    xs, ys = training_set()

We&rsquo;ll only have a single set of weights here, since we only have one layer.

    torch.manual_seed(0)
    
    W = torch.randn((nk, nv), dtype=torch.float32, requires_grad=True, device='cuda') # nk * nv


<a id="org5031d1c"></a>

## The forward pass

    def forward(xs): # N
        xs_enc = F.one_hot(xs, num_classes=nk).float().to('cuda') # N * nk
        logits = xs_enc @ W # N * nv
        probs = logits.softmax(1)
        return probs


<a id="orgdac9571"></a>

## Sampling from the model

    def sample(num_samples):
        torch.manual_seed(0)
        gens = []
        for i in range(num_samples):
            out = '.' * (ngram_n - 1)
            n = 0
            z = None
            while True:
                probs = forward(torch.tensor([n]))
                new_n = torch.multinomial(probs, 1, True).data.item()
                n = new_n
                if n == 0:
                    break
                out += itov[n]
                new_k = out[1-ngram_n:]
                if new_k not in ktoi:
                    break
                n = ktoi[new_k]
            gens.append(out[ngram_n-1:])
        return gens


<a id="org7caec03"></a>

## The training loop

The learning rate schedule is just something random; I experimented with stuff like
$A \exp\left( \lambda \cdot \mathrm{loss_change} \right) $ but they didn&rsquo;t do that well
and I didn&rsquo;t feel like tinkering much further with a poorly reinvented wheel.

    def train(xs, ys):
        W.grad = None
        old_loss = 0.0
        losses = []
        for i in range(501):
            pred_ys = forward(xs)
            loss = -pred_ys[torch.arange(len(xs)), ys].log().mean()
            loss.backward()
            new_loss = loss.data.item()
            loss_change = old_loss - new_loss
            old_loss = new_loss
            losses.append(new_loss)
            if i == 0:
                lr = 100
            elif i < 10:
                lr = 10
            elif i < 25:
                lr = 1
            elif i < 50:
                lr = 0.1
            elif i < 250:
                lr = 0.01
            else:
                lr = 0.001
            lr = float(lr)
            W.data += -lr * W.grad
            if i % 50 == 0:
                samples = sample(5)
                print(f'iter {i}: {lr=:.4f} loss={loss.data.item():.4f}, {samples=}')
        return losses
    
    losses = train(xs, ys)

    iter 0: lr=100.0000 loss=3.6698, samples=['m', 'pcdfuyxhksoraosbjeansxvfsfxxhhpphmkngrtzeljarrkhzxkchfeeqceu', 'spspcmmrcvgggeawfanxbhanvfeyshodpuansinvtgeeppx', 'rhuxoryaytdnbna', 'kcjtsybsegxannvfavtzu']
    iter 50: lr=0.0100 loss=2.5651, samples=['mami', 'auyni', 'soraleyalann', 'mi', 'fxxheeli']
    iter 100: lr=0.0100 loss=2.5555, samples=['mami', 'auyni', 'soraleyaladsxli', 'fxxheeli', 'kn']
    iter 150: lr=0.0100 loss=2.5424, samples=['mami', 'auyni', 'soraleyaladsxli', 'fxxheeli', 'kn']
    iter 200: lr=0.0100 loss=2.5315, samples=['m', 'mi', 'auyni', 'soraleyaladsxli', 'fxxheeli']
    iter 250: lr=0.0010 loss=2.5242, samples=['m', 'me', 'auyni', 'soraleyaladsxli', 'fxxhe']
    iter 300: lr=0.0010 loss=2.5234, samples=['m', 'me', 'auyni', 'soraleyaladsxli', 'fxxhe']
    iter 350: lr=0.0010 loss=2.5222, samples=['m', 'me', 'auyni', 'soraleyaladsxli', 'fxxhe']
    iter 400: lr=0.0010 loss=2.5206, samples=['m', 'me', 'auyni', 'soraleyaladsxli', 'fxxhe']
    iter 450: lr=0.0010 loss=2.5188, samples=['m', 'me', 'auyni', 'soraleyaladsxli', 'fxxheeli']
    iter 500: lr=0.0010 loss=2.5169, samples=['m', 'me', 'auyni', 'soraleyaladsxli', 'fxxheeli']


<a id="org6c4e525"></a>

## How did our losses change?

    plt.plot(losses)

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<tbody>
<tr>
<td class="org-left">&lt;matplotlib.lines.Line2D</td>
<td class="org-left">at</td>
<td class="org-left">0x7f35ac544820&gt;</td>
</tr>
</tbody>
</table>

![img](./.ob-jupyter/aae51076ffd4ccca2f65a6a3101521a7a51f3eff.png)


<a id="org1b22cdd"></a>

## Extra: how low can the loss get?

It&rsquo;s the loss of the frequency-based model on the training set:

    expected_loss = -P[xs,ys].log().mean()
    print(expected_loss)

    tensor(2.4540)

