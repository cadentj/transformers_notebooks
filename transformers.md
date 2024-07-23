---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.2
  kernelspec:
    display_name: Python 3
    name: python3
---

```python id="eo1EwQhfQJ0S"
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtyping import TensorType
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = "cuda:0"
```

# Transformers

In 2017, Google Brain released a paper titled *Attention is All you Need*, where they departed from the traditional Seq2Seq architectures used in NLP and introduced a new kind of architecture: the transformer. As the title of their paper suggests, the transformer is based on attention mechanisms, and does not use any kind of recurrent states as in RNNs. This meant that a transformer could be trained significantly faster than a Seq2Seq model using RNNs, as sequences were no longer processed one token at a time, but rather all at once. It also significantly reduced the complexity of the models being worked with. As opposed to having complex Seq2Seq variations that can quickly become intractable, the transformer is composed of relatively straightforward matrix multiplications all the way through.


This notebook focuses on GPT-2 style transformers, introduced by OpenAI in their paper *Language Models are Unsupervised Multitask Learners*. The original transformer architecture used an encoder-decoder structure for sequence to sequence tasks like translation. GPT-2 and most modern transformers use only the decoder component to excel at language modeling and text generation. In this first section, we'll focus on text generation at a high level.


Let's start with tokenization. A model's "vocabulary" isn't simply a list of words like we've done previously. Transformers use subword tokenization instead of full words. We can think of tokens as individual, meaningful components of language, with parts like prefixes root words (or parts of root words), suffixes, and the like. Learning the atomic structure of language allows our model to generalize onto never-before-seen words and form more a nuanced representation of language.

**How do we define tokens?**

We need to define a standard way of splitting up language into a series of substrings, where each substring is a member of our **vocabulary** set.

Could we use a dictionary, and have our vocabulary be the set of all words in the dictionary? No, because this couldn't handle arbitrary text (e.g. URLs, punctuation, etc). We need a more general way of splitting up language.

Could we just use the 256 ASCII characters? This fixes the previous problem, but it loses structure of language - some sequences of characters are more meaningful than others. For example, "language" is a lot more meaningful than "hjksdfiu". We want "language" to be a single token, but not "hjksdfiu" - this is a more efficient use of our vocab.

What actually happens? The most common strategy is called **Byte-Pair encodings**.

We begin with the 256 ASCII characters as our tokens, and then find the most common pair of tokens, and merge that into a new token. Note that we do have a space character as one of our 256 tokens, and merges using space are very common. For instance, here are the five first merges for the tokenizer used by GPT-2 (you'll be able to verify this below).

```
" t"
" a"
"he"
"in"
"re"
```

Note - you might see the character `Ġ` in front of some tokens. This is a special token that indicates that the token begins with a space. Tokens with a leading space and those without a leading space are different.

We'll use the `AutoModelForCausalLM` and `AutoTokenizer` libraries to load in a transformer model and its tokenizer from Huggingface. (Hugging Face is an open source ML platform for sharing pretrained models, datasets, and much more!)

```python
repo_id = "openai-community/gpt2"

model = AutoModelForCausalLM.from_pretrained(repo_id)
tokenizer = AutoTokenizer.from_pretrained(repo_id)
```

Try printing the tokenizer's `vocab` attribute.

```python
tokenizer.vocab
```

The column on the right is the individual token's "order" in the vocabulary. Sorting the vocabulary will give us a better idea of what's going on.

```python
vocab_list = list(tokenizer.vocab.items())
sorted_vocab = sorted(vocab_list, key=lambda n: n[1])

print(sorted_vocab[:10])
print(sorted_vocab[250:260])
```

As you get to the end of the vocabulary, you'll be producing some pretty weird-looking esoteric tokens (because you'll already have exhausted all of the short frequently-occurring ones):

```python
print(sorted_vocab[-5:])
```

There are a few funky and frustrating things about tokenization, which causes it to behave differently than you might expect. 

Let's define a short method to convert lists of ids into tokens to get an idea of what's going on.

```python
def to_str_tokens(string: str) -> list[str]:
    """
    Turn a string into a list of tokens.

    Args:
        string (str): The string to tokenize.

    Returns:
        List[str]: A list of stringified tokens.
    """

    # The .encode method turns a string
    # into a list of integer ids, .decode 
    # does the opposite
    tokens = tokenizer.encode(string)
    
    str_tokens = []
    for token in tokens:

        str_tokens.append(
            tokenizer.decode(token)
        )

    return str_tokens
```

Whether a word begins with a capital or space matters!

```python
print(to_str_tokens("Ralph"))
print(to_str_tokens(" Ralph"))
print(to_str_tokens(" ralph"))
print(to_str_tokens("ralph"))
```

#### Arithmetic is a mess.

Length is inconsistent, common numbers bundle together.

```python
print(to_str_tokens("56873+3184623=123456789-1000000000"))
```

## Text generation

Now that we understand the basic ideas here, let's go through the entire process of text generation, from our original string to a new token which we can append to our string and plug back into the model. 

We start with a sequence of length $N$ tokens (words or subwords). This sequence is fed into the model which predicts probabilities for the $N+1$th word. We use this to construct a new sequence of length $N+1$, feed this new sequence in the model, and so on! We say a transformers is autoregressive, because it only predicts future words based on past data.

To mark the beginning of a sequence, we'll use a (BOS) token. 

* It provides context that this is the start of a sequence, which can help the model generate more appropriate text.
* It can act as a "rest position" for attention heads (more on this later, when we discuss attention).

In GPT-2, the End of Sequence (EOS), Beginning of Sequence (BOS) and Padding (PAD) tokens are all the same, `<|endoftext|>` with index `50256`. This is because GPT-2 is an autoregressive model that only processes text left ot right, so it has no need to distinguish between BOS and EOS tokens.

#### **Step 1:** Convert text to tokens

The sequence gets tokenized, so it has shape `[batch, seq_len]`. Here, the batch dimension is just one (because we only have one sequence).


```python
reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = tokenizer.encode(reference_text, return_tensors="pt")

print(tokens)
print(tokens.shape)
```

#### **Step 2:** Map tokens to logits

From our input of shape `[batch, seq_len]`, we get output of shape `[batch, seq_len, vocab_size]`. The `[i, j, :]`-th element of our output is a vector of logits representing our prediction for the `j+1`-th token in the `i`-th sequence.

If you haven't encountered the term "logits" before, here's a quick refresher.

Given an arbitrary vector $x$, we can turn it into a probability distribution via the **softmax** function: $x_i \to \frac{e^{x_i}}{\sum e^{x_j}}$. The exponential makes everything positive; the normalization makes it add to one.

The model's output is the vector $x$ (one for each prediction it makes). We call this vector a logit because it represents a probability distribution, and it is related to the actual probabilities via the softmax function.

```python
logits = model(tokens).logits
```

#### **Step 3:** Convert the logits to a distribution with a softmax

This doesn't change the shape, it is still `[batch, seq_len, vocab_size]`.

```python
probs = logits.softmax(dim=-1)
print(probs.shape)
```

#### **Step 3.5:** What is the most likely next token at each position?

```python
# Batch decode convert a list of lists of token ids into a list of strings by calling decode.
top_token = probs.argmax(dim=-1)[0]
most_likely_next_tokens = tokenizer.batch_decode(top_token)

print(list(zip(to_str_tokens(reference_text), most_likely_next_tokens)))
```

We can see that, in a few cases (particularly near the end of the sequence), the model accurately predicts the next token in the sequence. We might guess that `"take over the world"` is a common phrase that the model has seen in training, which is why the model can predict it.

#### **Step 4:** Map distribution to a token

```python
# Index to the last token position of the softmaxed logits.
# Then take the argmax along the last dimension.
next_token = probs[0, -1].argmax(dim=-1)

# Decode the most probable token into a string.
next_char = tokenizer.decode(next_token)

print(f"'{next_char}'")
```

Note that we're indexing `logits[0, -1]`. This is because logits have shape `[1, sequence_length, vocab_size]`, so this indexing returns the vector of length `vocab_size` representing the model's prediction for what token follows the **last** token in the input sequence.

In this case, we can see that the model predicts the token `' I'`.

### **Step 5:** Add this to the end of the input, re-run

There are more efficient ways to do this (e.g. where we cache some of the values each time we run our input, so we don't have to do as much calculation each time we generate a new value), but this doesn't matter conceptually right now.

```python
print(f"Sequence so far: {tokenizer.batch_decode(tokens)[0]}")

def generate(
    tokens: TensorType["batch", "sequence"],
    running_token: TensorType["seq"],
    n_tokens: int = 10
) -> TensorType["batch", "sequence"]:
    """
    Generate a sequence of n_tokens by appending tokens to the input sequence.

    Args:
        tokens (Tensor): The input sequence.
        running_token (Tensor): The running token.
        n_tokens (int): The number of tokens to generate.

    Returns:
        (Tensor): The generated sequence.
    """

    # We want to iteratively generate the next token and append it to the running sequence. 

    for _ in range(n_tokens):
        
        # Define new input sequence, by appending the previously generated token
        # You can use [None, None] to add two new dimensions to the tensor. 
        # This is equivalent to unsqueeze(0).unsqueeze(0), and it makes 
        # the 0d running compatible with the running sequence of tokens shape (batch, seq)
        
        # <COGINST>
        tokens = torch.cat([tokens, running_token[None, None]], dim=-1)
        # </COGINST>

        # Pass our new sequence through the model, to get new output
        # Get the predicted token at the end of our sequence by taking
        # the most probable token from the last position of the logits.

        # Logits will be shape (batch, seq, vocab_size)
        # The argmax should be taken over the last dimension of the last token
        # e.g. tensor[:,-1,:]

        # <COGINST>
        logits = model(tokens).logits
        running_token = logits[0, -1].argmax(dim=-1)
        # </COGINST>

        # Decode and print the result
        next_char = tokenizer.decode(running_token)

        print(f"{tokens.shape[-1]}th char = {next_char}")

    return tokens

_ = generate(tokens, next_token, 10)
```

Now that we have a high level understanding of transformer behavior, let's build one from scratch to understand the role of each component!

<!-- #region -->
**Embedding**

At the start of a transformer, we have an embedding $W_E$ that consists of a lookup of integer tokens $t$ to vectors.


**Residual Stream**

Through the heart of the transformer, we have the residual stream. The residual stream is the sum of all previous outputs of layers of the model, is the input to each new layer. It has shape `[batch, seq_len, d_model]` (where `d_model` is the length of a single embedding vector).

The initial value of the residual stream is denoted $x_0$ in the diagram, and $x_i$ are later values of the residual stream (after more attention and MLP layers have been applied to the residual stream).

The residual stream is *really* fundamental. It's the central object of the transformer. It's how model remembers things, moves information between layers for composition, and it's the medium used to store the information that attention moves between positions.

**Transformer layers**
Then we have a series of `n_layers`, each consisting of an attention and mlp component. In this notebook, we'll construct an attention only transformer.
<!-- #endregion -->

Image - TODO?


## Attention


Attention layers play a crucial role in transformer architectures, enabling tokens to access information from distant positions in the sequence. Here, we will be using a _scaled dot product attention_. As the name suggests, we will compute our attention scores by computing dot products.

This attention function takes in three inputs: the _queries_ $Q$, the _keys_ $K$, and the _values_ $V$. The names are not completely arbitrary here: we can think of the keys and values as key-value pairs in a Python dictionary.

For a given sequence, $Q$, $K$, and $V$ will each be a $(seq, d_{head})$ matrix where $seq$ is the length of the token sequence.

For instance, the queries will be represented by the matrix

\begin{align}
&\:\begin{matrix}\xleftarrow{\hspace{0.5em}} & d_k & \xrightarrow{\hspace{0.5em}}\end{matrix} \\
Q =\;\, &\begin{bmatrix}\leftarrow & \vec{q}_1 & \rightarrow \\ \leftarrow & \vec{q}_2 & \rightarrow \\ & \vdots & \\ \leftarrow & \vec{q}_t & \rightarrow\end{bmatrix}\;\;\begin{matrix}\Big\uparrow \\ t \\ \Big\downarrow\end{matrix}
\end{align}

We will compute attention scores from $Q$ and $K$. In particular, we will matrix multiply $Q$ and the transpose of $K$ to calculate the dot product between each query vector and each key vector

This gives us a $(seq_{q}, seq_{k})$ matrix $E$, where the $(i, j)^\text{th}$ element in $E$ is the attention score between the $i^\text{th}$ query vector and the $j^\text{th}$ key vector. We’ll call $E$ the attention scores matrix.

\begin{align}
&\begin{bmatrix}
    \;\;\; \uparrow \;\; & \quad \uparrow \; & & \;\, \uparrow \\ 
    \;\;\; \vec{k}_1 \;\; & \quad \vec{k}_2 \; & \;\; \cdots & \;\;\;\, \vec{k}_T \;\; \\ 
    \;\;\; \downarrow \;\; & \quad \downarrow \; & & \;\, \downarrow
\end{bmatrix} \\
E = QK^\intercal =
\begin{bmatrix}
    \leftarrow & \vec{q}_1 & \rightarrow \vphantom{\vec{q}_1\cdot\vec{k}_t} \\
    \leftarrow & \vec{q}_2 & \rightarrow \vphantom{\vec{q}_2\cdot\vec{k}_t} \\
    & \vdots & \\
    \leftarrow & \vec{q}_t & \rightarrow \vphantom{\vec{q}_T\cdot\vec{k}_t}
\end{bmatrix}
&\begin{bmatrix}
    \vec{q}_1\cdot\vec{k}_1 & \vec{q}_1\cdot\vec{k}_2 & \cdots & \vec{q}_1\cdot\vec{k}_T \\
    \vec{q}_2\cdot\vec{k}_1 & \vec{q}_2\cdot\vec{k}_2 & \cdots & \vec{q}_2\cdot\vec{k}_T \\
    \vdots & \vdots & \ddots & \vdots \\
    \vec{q}_t\cdot\vec{k}_1 & \vec{q}_t\cdot\vec{k}_2 & \cdots & \vec{q}_t\cdot\vec{k}_T \\
\end{bmatrix}=
\begin{bmatrix}
    e_{1,1} & e_{1,2} & \cdots & e_{1,T} \vphantom{\vec{q}_1\cdot\vec{k}_T} \\
    e_{2,1} & e_{2,2} & \cdots & e_{2,T} \vphantom{\vec{q}_2\cdot\vec{k}_T} \\
    \vdots & \vdots & \ddots & \vdots \\
    e_{t,1} & e_{t,2} & \cdots & e_{t,T} \vphantom{\vec{q}_T\cdot\vec{k}_T} \\
\end{bmatrix}
\end{align}


<!-- #region -->
Next, we will _scale_ each of the attention scores by a factor of $\frac{1}{\sqrt{d_k}}$. This is done to push the dot products into a reasonable range for the subsequent softmax - too negative of a dot product will results in a near-zero gradient and thus the model will train less effectively.

\begin{equation}
E' = \frac{1}{\sqrt{d_k}}E
\end{equation}


Finally we take the softmax of $E'$ to get our attention weights, where the sum in softmax is done over the columns of $E'$

\begin{equation}
\alpha = \operatorname{softmax}(E') =
\begin{bmatrix}
    \operatorname{softmax}\begin{pmatrix}e'_{1,1} & e'_{1,2} & \cdots & e'_{1,T}\end{pmatrix} \\
    \operatorname{softmax}\begin{pmatrix}e'_{2,1} & e'_{2,2} & \cdots & e'_{2,T}\end{pmatrix} \\
    \vdots \\
    \operatorname{softmax}\begin{pmatrix}e'_{t,1} & e'_{t,2} & \cdots & e'_{t,T}\end{pmatrix}
\end{bmatrix}
\end{equation}
<!-- #endregion -->

The resulting $\alpha$ is our attention pattern. The attention pattern is shape $(seq_{q}, seq_{k})$ matrix $E$. 

The $(i, j)^\text{th}$ element in the attention pattern is a probability weighting how much information to copy between $i$ and $j$

Let's compute some attention scores to get a feel for whats going on. Write the function `compute_attn_pattern` which accepts a weight matrices $W_Q$ and $W_K$, an input $x$, and an attention mask, then returns an attention pattern.

```python
def get_mask(size: int) -> TensorType["size", "size"]:
    """
    Return an upper triangular mask of (size, size).

    Args:
        size (int): The size of the mask.
    """

    # Create a Tensor of ones shape (size, size).
    # Make it a 'mask' by setting the dtype to torch.bool.
    # This allows us to use it as a mask: 
    # e.g. tensor[mask]
    # which only returns the elements of tensor where mask is True.

    # <COGINST>
    ones_like = torch.ones(
        size, 
        size, 
        dtype=torch.bool, 
        device=DEVICE
    )
    # </COGINST>
    
    # Create an upper triangular mask by taking the transpose of the lower triangular mask.

    # <COGINST>
    mask = torch.triu(ones_like).T
    # </COGINST>

    return mask

def compute_attn_weights(
    W_Q: TensorType["d_model", "d_head"],
    W_K: TensorType["d_model", "d_head"],
    x: TensorType["seq", "d_model"],
    mask: Optional[TensorType["d_head", "d_head"]] = None
) -> TensorType["seq", "seq"]:  
    """
    Compute the masked attention pattern between W_Q and W_K
    for a given input tensor x.

    Args:
        W_Q (Tensor): The query weight matrix.
        W_K (Tensor): The key weight matrix.
        x (Tensor): The input tensor.
        mask (Tensor, optional): The mask to apply to the attention pattern.
    
    Returns:
        A: The attention pattern.
    """

    # Compute Q, K using W_Q, W_K
    # The matrices Q, K will be shape (seq, d_head)

    # <COGINST>
    Q = x @ W_Q 
    K = x @ W_K
    # </COGINST>

    # Then, compute the attention scores A, QK^T
    # A will be shape (seq, seq). 

    # Scale the attention scores by 1/sqrt(d_head) to push the dot product into a reasonable range for softmax.
    # During training, if the dot products are too large, the softmax will saturate (all values will be close to 0 or 1) 
    # and the gradients will be very small, making learning difficult.

    # <COGINST>
    A = Q @ K.transpose(-2, -1)
    # </COGINST>

    # Apply the mask to A. One helpful Torch function here
    # is torch.masked_fill(mask, value), which will replace all values in A
    # where mask is 'True' with a value.

    if mask is not None:
        A = A.masked_fill(mask == 0, -1e9)

    # Finally, apply the softmax function to A, and return the result.
    # Applying the softmax computes the attention weights from the attention scores
    # where each element (i,j) is a probability weighing how much information to copy between i and j

    # <COGINST>
    A = F.softmax(A, dim = -1)
    # </COGINST>
    
    return A
```

`utils.py` provides a couple functions to load in weights and activations from a pretrained model, GPT-2.

`get_split_l0_heads()`
- This function loads and splits the weights $W_Q, W_K, W_V$ in layer 0. Transformers in practice typically have their heads concatenated for more efficient computation. The GPT-2 implementation from Hugging Face has $W_Q, W_K, W_V$ combined into a single weight matrix shape `[d_model, (d_model * 3)]`. We split that weight matrix into $W_Q, W_K, W_V$, then split those matrices further into `[d_model, n_heads, d_head]`.

`get_pre_attn()`
- This function accepts a prompt and returns the hidden state before attention in the 0th layer. This is our input `x`.

### A note on multi headed attention

We've only discussed single headed attention until now. In single headed attention, a single set of attention scores is calculated for each token in the sequence. $Q$, $K$, and $V$ will each be a $(seq, d_{head})$ matrix.

Attention can easily scale to using multiple heads. In multi headed attention, the weight matrices for $Q$, $K$, and $V$ are shape `[d_model, (n_heads * d_head)]`. Each head independently computes its own set of attention scores, allowing the model to focus on different parts of the sequence simultaneously. The outputs of these heads are then concatenated and linearly transformed to produce the final output.

`get_split_l0_heads` will accept an argument for `head_idx`. Under the hood, `get_split_l0_heads` will split the head weights and return the weight for the respective head index. Write the function `visualize_all_heads()` which iterates through all heads in GPT2 and computes the attention pattern.

```python
from utils import get_pre_attn, get_split_l0_heads
import circuitsvis as cv

def visualize_all_heads(prompt: str, n_heads: int) -> None:

    """
    Visualize the attention patterns for all heads in the model for a given prompt.

    Args:
        prompt (str): The input prompt.
        n_heads (int): The number of heads to visualize.

    Returns:
        None
    """

    # Begin by loading in the string tokens and pre-attention weights with `get_pre_attn`.

    # <COGINST>
    string_tokens, pre_attn = get_pre_attn(prompt=prompt)
    # </COGINST>

    heads = []

    for head in range(n_heads):

        # Get the weight matrices for the query and key for the head.
        # These will be shape (d_model, d_head).

        # <COGINST>
        W_Q, W_K = get_split_l0_heads(head_idx=head)
        # </COGINST>

        # Compute the attention mask for the sequence length.

        # <COGINST>
        mask = get_mask(len(string_tokens))
        # </COGINST>

        # Compute the attention weights for the head and append to the list of heads.

        # <COGINST>
        A = compute_attn_weights(W_Q, W_K, pre_attn, mask=mask)

        heads.append(A)
        # </COGINST>

    heads = torch.stack(heads, dim=0)

    # CircuitsVis is a helpful library for visualizing Transformer related things.
    
    display(
        cv.attention.attention_patterns(
            tokens = string_tokens, 
            attention = heads
        )
    )

prompt = "When Mary and John went to the store, John gave a drink to"
n_heads = 12

visualize_all_heads(
    prompt = prompt,
    n_heads = n_heads
)
```

We notice that there are three basic patterns which repeat quite frequently:

* `prev_token_heads`, which attend mainly to the previous token (e.g. head `0.7`)
* `current_token_heads`, which attend mainly to the current token (e.g. head `0.1`)
* `first_token_heads`, which attend mainly to the first token (e.g. heads `0.0`, although these are a bit less clear-cut than the other two)

The `prev_token_heads` and `current_token_heads` are perhaps unsurprising, because words that are close together in a sequence probably have a lot more mutual information than words that are far apart (i.e. we could get quite far using bigram or trigram prediction).

The `first_token_heads` are a bit more surprising. The basic intuition here is that the first token in a sequence is often used as a resting or null position for heads that only sometimes activate (since our attention probabilities always have to add up to 1).


Awesome! Now let's put it all together by writing our own single headed attention layer.

And, as we did before, we will use our attention weights as the coefficients in weighted sum of the value vectors - the rows of $V$ - and we can accomplish this with yet another matrix multiplication:

\begin{align}
&\begin{bmatrix}
    \xleftarrow{\hspace{6em}} & \vec{v}_1 & \xrightarrow{\hspace{6em}} \\
    \xleftarrow{\hspace{6em}} & \vec{v}_2 & \xrightarrow{\hspace{6em}} \\
    & \vdots &  \\
    \xleftarrow{\hspace{6em}} & \vec{v}_T & \xrightarrow{\hspace{6em}}
\end{bmatrix} \\
C = \alpha V =
\begin{bmatrix}
    \alpha_{1,1} & \alpha_{1,2} & \cdots & \alpha_{1,T} \vphantom{\sum\limits_{i=1}^T} \\
    \alpha_{2,1} & \alpha_{2,2} & \cdots & \alpha_{2,T} \vphantom{\sum\limits_{i=1}^T} \\
    \vdots & \vdots & \ddots & \vdots \\
    \alpha_{t,1} & \alpha_{t,2} & \cdots & \alpha_{t,T} \vphantom{\sum\limits_{i=1}^T} \\
\end{bmatrix}&
\begin{bmatrix}
    \sum\limits_{i=1}^T \alpha_{1,i}(\vec{v}_i)_1 & \sum\limits_{i=1}^T \alpha_{1,i}(\vec{v}_i)_2 & \cdots & \sum\limits_{i=1}^T \alpha_{1,i}(\vec{v}_i)_{d_v} \\
    \sum\limits_{i=1}^T \alpha_{2,i}(\vec{v}_i)_1 & \sum\limits_{i=1}^T \alpha_{2,i}(\vec{v}_i)_2 & \cdots & \sum\limits_{i=1}^T \alpha_{2,i}(\vec{v}_i)_{d_v} \\
    \vdots & \vdots & \ddots & \vdots \\
    \sum\limits_{i=1}^T \alpha_{t,i}(\vec{v}_i)_1 & \sum\limits_{i=1}^T \alpha_{t,i}(\vec{v}_i)_2 & \cdots & \sum\limits_{i=1}^T \alpha_{t,i}(\vec{v}_i)_{d_v}
\end{bmatrix} =
\begin{bmatrix}
    \leftarrow & \vec{c}_1 & \rightarrow \vphantom{\sum\limits_{i=1}^T} \\
    \leftarrow & \vec{c}_2 & \rightarrow \vphantom{\sum\limits_{i=1}^T} \\
    & \vdots &  \\
    \leftarrow & \vec{c}_t & \rightarrow \vphantom{\sum\limits_{i=1}^T}
\end{bmatrix}
\end{align}

This will give us a matrix $C$ that has shape-$(t, d_v)$, where each row is a context vector in the same sense as before.
Because of we weight the value vectors by attention weights from the keys, it often makes sense to have $K$ and $V$ be the same;
if we were to draw a parallel to our previous Seq2Seq model's attention mechanism, both $K$ and $V$ would have been $H^e$, while $Q$ would be $\vec{h}{}^d_t$.

All in all, we can write our attention as the function

\begin{equation}
C = \operatorname{Attention}(Q, K, V) = \operatorname{softmax}\bigg(\frac{QK^\intercal}{\sqrt{d_k}}\bigg)V.
\end{equation}


Most of what we wrote before can be repurposed into the forward method of our SingleHeadAttention module. Instead of taking $W_Q$ and $W_K$ as inputs, we want to tweak `compute_attn_pattern` to use submodules `q_layer` and `k_layer` instead.

```python id="sgqYZmyO-muh"
class SingleHeadAttention(nn.Module):
    def __init__(self, d_model: int):
        """
        Here we will assume that the input dimensions are same as the
        output dims.
        """
        super().__init__()

        # Create Q, K, V weight matrices for the attention head.

        # <COGINST>
        self.q_layer = torch.nn.Linear(d_model, d_model)
        self.k_layer = torch.nn.Linear(d_model, d_model)
        self.v_layer = torch.nn.Linear(d_model, d_model)
        # </COGINST>

    def forward(
        self,
        x: TensorType["seq", "d_model"], 
        mask: TensorType["seq", "seq"] = None, 
        return_weights: bool = False
    ) -> TensorType["seq", "d_model"]:
        """
        Assume x is shape (seq, d_model)
        
        W_Q, W_K, and W_V are weights for projecting into queries,
        keys, and values, respectively. Here these will have shape
        (d_model, d_head), yielding d dimensional vectors for each input.

        This function should return a (seq) dimensional attention weight
        for each input -- i.e., an attention matrix with shape (seq, seq).

        Derive Q, K, V matrices, then self attention weights. These should
        be used to compute the final representations (seq, d_model); optionally
        return the weights matrix if `return_weights=True`.

        Args:
            x (Tensor): The input tensor.
            mask (Tensor): The mask tensor.
            return_weights (bool): Whether to return the attention weights.
        """
        
        # Compute the attention scores for the input tensor x. 
        
        # <COGINST>
        Q = self.q_layer(x)
        K = self.k_layer(x)
        V = self.v_layer(x)

        A = Q @ K.transpose(-2, -1)

        if mask is not None:
            A = A.masked_fill(mask == 0, -1e9)
            
        weights = F.softmax(A, dim=-1)
        # </COGINST>

        # Compute the output by multiplying the attention weights by the value matrix V.
        # Include a check to return the weights if `return_weights=True`.

        # <COGINST>
        if return_weights:
          return weights, weights @ V

        out = weights @ V
        # </COGINST>

        return out
        

```

## Layer Norm


The layer norm is a simple normalization function applied at the start of each layer. It converts each input vector to have mean zero and variance 1, then applies an elementwise scaling and translation.

Let's implement our own LayerNorm function. The [PyTorch docs](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html) provide a quick summary of the parameters from the original paper.


\begin{equation}
y = \frac{x - \mathbb{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} \cdot \gamma + \beta
\end{equation}


Some helpful functions might be [`.mean()`](https://pytorch.org/docs/stable/generated/torch.mean.html) and [`.var()`](https://pytorch.org/docs/stable/generated/torch.var.html).

```python
class LayerNorm(nn.Module):

    def __init__(self, d_model: int, eps=1e-6):
        super().__init__()
        
        # Create attributes for .eps, .w, and .b.
        # These correspond to the epsilon value for numerical stability,
        # the weight vector, and the bias vector, respectively.

        # Use nn.Parameter to ensure that these are trainable parameters.

        # <COGINST>
        self.eps = eps

        self.w = nn.Parameter(torch.ones(d_model))
        self.b = nn.Parameter(torch.zeros(d_model))
        # </COGINST>

    def forward(
        self, 
        residual: TensorType["seq", "d_model"]
    ) -> TensorType["seq", "d_model"]:
        
        # Compute the mean and standard deviation of the residual tensor.
        # Use .mean and .std along the d_model dimension.

        # <COGINST>
        residual_mean = residual.mean(dim=-1, keepdim=True)
        residual_std = (residual.var(dim=-1, keepdim=True, unbiased=False) + self.eps).sqrt()
        # </COGINST>

        # Normalize the residual tensor using the mean and standard deviation.
        # Then apply the learned weight and bias.

        # <COGINST>
        residual = (residual - residual_mean) / residual_std
        residual = residual * self.w + self.b
        # </COGINST>

        return residual
```

Let's test out LayerNorm to see if its doing the normalization we want. Create a random input tensor shape `[batch_size, seq_length, d_model]`. Instantiate a LayerNorm object and normalize the tensor.

```python
# Create input data
batch_size, d_model = 1000, 64
input_data = torch.randn(batch_size, d_model) * 5 + 2

# Initialize and apply LayerNorm
layer_norm = LayerNorm(d_model)
output_data = layer_norm(input_data)

def plot_distribution(ax, data, title):
    ax.hist(data.flatten().detach().numpy(), bins=50, density=True)
    ax.set_title(title)
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 0.5)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 3))
plot_distribution(ax1, input_data, "Input Data Distribution")
plot_distribution(ax2, output_data, "Output Data Distribution after LayerNorm")
```

Let's bundle everything together into a simple transformer layer. Self-attention refers to the fact that our input sequence `src` attends to itself via attention!

```python
class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_dim: int):
        super(SelfAttentionLayer, self).__init__()

        # The self-attention layer should contain a SingleHeadAttention module
        # followed by a LayerNorm module.

        # <COGINST>
        self.self_attn = SingleHeadAttention(embed_dim)
        self.norm1 = LayerNorm(embed_dim)
        # </COGINST>

    def forward(
        self, 
        src: TensorType["seq", "d_model"], 
        mask: TensorType["seq", "seq"] = None
    ):
        """
        Args:
            src (Tensor): The input tensor.
            mask (Tensor): The mask tensor.
        
        Returns:
            Tensor: The output tensor.
        """

        # Simply pass the input through the SingleHeadAttention layer,
        # followed by a LayerNorm module.
        src = src + self.norm1(
            self.self_attn(src, mask=mask)
        )

        return src
```

## Positional Encoding


Attention operates over all pairs of positions. This means it's invariant with regards to position - the attention calculation from token 5 to token 1 and token 5 to token 2 are the same by default. We don't want this - nearby tokens should be more relevant. 

`PositionalEncoding` class is a crucial component in transformer-based models. It adds positional information to input embeddings, allowing the model to understand the sequence order of inputs.


We want to insert some unique positional information. Our implementation of  `PositionalEncoding` class uses sine and cosine functions of different frequencies to generate these unique positional embeddings for each position in the input sequence. These embeddings are then added to the input embeddings to provide positional context.

The encoding for a position `pos` and dimension `i` is given by:

\begin{align}
PE_{(pos, 2i)} &= \sin(pos / 100^{2i/d_{model}}) \\
PE_{(pos, 2i+1)} &= \cos(pos / 100^{2i/d_{model}})
\end{align}

where $d_{model}$ is the dimensionality of the model.



```python
def positional_encoding(position: TensorType["seq"], d_model: int) -> TensorType["seq", "d_model"]:
    """
    Generate positional encoding for the input sequence.

    Positional encoding is used in transformer models to provide the model with 
    information about the relative or absolute position of the tokens in the sequence.

    Args:
        position (Tensor): Tensor representing the position of the tokens in the sequence.
        d_model (int): The dimension of the encoding vector.

    Returns:
        (Tensor): A tensor of shape (N, d_model) containing the positional encodings.
    """ 

    # Initialize a position tensor shape (seq, d_model) with zeros.
    # Create a tensor of positions (0, 1, 2, ..., N-1) and reshape it to (N, 1)

    # <COGINST>
    pe = torch.zeros(position.size(0), d_model)
    position = torch.arange(0, position.size(0), dtype=torch.float).unsqueeze(1)
    # </COGINST>

    # Compute the scaling term for the sine and cosine functions.

    # <COGINST>
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(100.0) / d_model))
    # </COGINST>

    # Set every even index to sin and every odd index to cos.
    # Slicing may be helpful here.

    # <COGINST>
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    # </COGINST>
    
    return pe

# Plot for positions 0-99 and dimensions 0, 1, 2, 3
positions = torch.arange(100).float()
d_model = 512
```

We can plot the sine and cosine functions for different dimensions.

```python
plt.figure(figsize=(12, 6))

pe = positional_encoding(positions, d_model)

for i in [0,100,200]:
    plt.plot(positions.numpy(), pe[:, i].numpy(), label=f'Dimension {i}')

plt.legend()
plt.title("Positional Encodings for Different Dimensions")
plt.xlabel("Position")
plt.ylabel("Encoding Value")
plt.show()
```

Lets see how different positions `[0, 10, 25, 50]` are represented across a model dimension of 512.

The x-axis represents the dimensions of the encoding vector, while the y-axis shows the encoding value, ranging from -1 to 1. Each line represents a different position's encoding.

As we move from left to right, the frequency of the waves decreases, resulting in the visible pattern change. Lower dimensions (left side) use higher frequency waves, allowing for fine-grained position differentiation. Higher dimensions (right side) use lower frequency waves, capturing broader positional information. This multi-scale representation enables the model to capture both local and global positional relationships in the input sequence.

```python
plt.figure(figsize=(12, 6))
for pos in [0, 10, 25, 50]:
    plt.plot(pe[pos].numpy(), label=f'Position {pos}')

plt.legend()
plt.title("Full Positional Encoding for Different Positions")
plt.xlabel("Dimension")
plt.ylabel("Encoding Value")
plt.show()
```

Now, integrate your positional encoding into the module PositionalEncoding. It should accept an input `x`.

```python
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len=5000):
        super(PositionalEncoding, self).__init__()

        # Create a positional encoding matrix of shape (max_len, d_model)
        # Then create a position tensor of shape (max_len, 1)

        # <COGINST>
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # </COGINST>

        # Finally compute the scaling term for the sine and cosine functions.
        # Set every even index to sin and every odd index to cos.
        # Slicing may be helpful here.

        # <COGINST>
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(100.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # </COGINST>

        # Register the positional encoding as a buffer.
        # A buffer is a tensor that is not considered a model parameter.
        self.register_buffer('pe', pe)

    def forward(self, x: TensorType["seq", "d_model"]) -> TensorType["seq", "d_model"]:
        # Add the positional encoding to the input tensor.

        # <COGINST>
        x = x + self.pe[:x.size(0), :]
        # </COGINST>

        return x
```

<!-- #region id="vuRuwq1Q_Z4x" -->
## Putting It All Together: A Simple Transformer Implementation

The provided code demonstrates how the key components we've discussed - positional encoding, layer normalization, and self-attention layers - come together to form a basic transformer model. Let's break down the `TransformerModel` class:


- `self.embed`: An embedding layer that converts input tokens to dense vectors.
- `self.pos_encoder`: Adds positional information to the embedded tokens.
- `self.layers`: A list of `SelfAttentionLayer` modules, forming the core of the transformer.
- `self.norm`: A final layer normalization applied after all attention layers.
- `self.decoder`: A linear layer that projects the final representations to vocabulary-sized logits.
<!-- #endregion -->

```python id="xEbCUxGivkmA"
class CogModel(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int):
        super(CogModel, self).__init__()

        # Create an embedding layer, positional encoding, and a list of self-attention layers.

        # <COGINST>
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim)
        self.layers = nn.ModuleList([SelfAttentionLayer(embed_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(embed_dim)
        self.decoder = nn.Linear(embed_dim, vocab_size)
        # </COGINST>

    def forward(
        self, 
        src: TensorType["batch", "seq"]
    ) -> TensorType["batch", "seq", "vocab_size"]:
        
        """
        Positional encodings are added to token embeddings at the start.
        Multiple self-attention layers are stacked to process the input deeply.
        Layer normalization is applied both within the attention layers and at the end of the entire stack.
        The model maintains the same dimensionality (embed_dim) throughout, 
        using residual connections (likely within SelfAttentionLayer) to facilitate gradient flow.

        Args:
            src (Tensor): The input tensor.

        Returns:
            (Tensor): The output tensor
        """

        # Create an attention mask to ensure the model only attends to previous tokens (for causal/autoregressive modeling).
        
        # <COGINST>
        mask = torch.triu(torch.ones(
        src.size(1), src.size(1), dtype=torch.bool, device=src.device)).T
        # </COGINST>

        # Embed the input tokens and add positional encodings.
        # Pass the embeddings through each self-attention layer sequentially.

        # <COGINST>
        src = self.embed(src)  
        src = self.pos_encoder(src)
        for layer in self.layers:
            src = layer(src, mask)
        # </COGINST>

        # Apply a final layer normalization.
        # Project the output to vocabulary-sized logits using the decoder.

        # <COGINST>
        src = self.norm(src)
        output = self.decoder(src)
        # </COGINST>

        return output

```

Rather than training the transformer yourself, we've trained it for you on a toy task.

```python
from utils import load_cogmodel

model, tokenizer = load_cogmodel()
```

You can call "echo" following a word to have the model repeat the word, or "upper" to repeat the word in all caps.

Let's play around with the model! Write a generate text method to play around with the model's outputs. Note that the model isn't perfect at performing the sequence tasks it was trained on - if you want, go check out the notebook today's exercises were based on to see if you can do better!

```python
def generate_text(prompt: str, max_len: int = 28) -> str: 
    """
    Generate text from a prompt using the CogModel.
    
    Args:
        prompt (str): The input prompt.
        max_len (int): The maximum length of the generated text.
    
    Returns:
        (str): The generated text.
    """

    
    # Use the tokenizer (just like before) to encode a prompt into a Tensor of token ids.

    # <COGINST>
    prompt_tensor = tokenizer.encode(prompt, return_tensors="pt")
    # </COGINST>

    # For a maximum length of tokens, generate the next token by passing the prompt tensor through the model.
    for _ in range(max_len):

        # Pass the prompt tensor through the model to get the output logits. 
        # If the predicted token is a 0, or padding token, break the loop.
        # Finally, concatenate the predicted token to the prompt tensor.

        # <COGINST>
        output = model(prompt_tensor)
        pred_token = output.argmax(dim=2)[:,-1]

        if pred_token == 0:
            break
        # </COGINST>
        
        prompt_tensor = torch.cat((prompt_tensor, pred_token.unsqueeze(0)), dim=1)

    # Decode the token ids into a string and return the result.

    # <COGINST>
    predicted_sentence = "".join(tokenizer.batch_decode(prompt_tensor[0]))
    # </COGINST>

    return predicted_sentence

result = generate_text('minecraft echo')
print(result)

result = generate_text('minecraft upper')
print(result)
```
