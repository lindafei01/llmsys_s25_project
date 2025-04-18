import numpy as np
from .tensor import tensor, tensor_from_numpy
from .module import Module, Parameter
from .modules_basic import (
    Embedding,
    Dropout,
    LayerNorm1d,
    Linear
)
from .tensor_ops import TensorBackend
from .nn import (
    max,
    softmax,
    dropout,
    GELU,
)
from .tensor_functions import Attn_Softmax, LayerNorm, FlashAttention
from typing import Any, Dict, Optional, Sequence, Tuple

datatype = np.float32


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool=False, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None, use_fused_kernel: bool=False):
        super().__init__()
        """Implements Multi-Head Attention as described in "Attention Is All You Need"

        Args:
            n_embd    : Dimensionality of embeddings and hidden states
            n_head    : Number of heads
            p_dropout : Dropout ratio for dropout layer
            causal    : If True, then apply a causal mask during self-attention
            bias      : If True, then apply a bias in Linear layers
        
        Attributes:
            q_projection   : Linear layer projecting input to Q matrix
            k_projection   : Linear layer projecting input to K matrix
            v_project      : Linear layer projecting input to V matrix
            out_projection : Linear output projection layer
            dropout        : Dropout layer
        """
        self.backend   = backend
        self.n_embd    = n_embd 
        self.n_head    = n_head
        self.causal    = causal
        self.attn_hidden_dim = n_embd // n_head

        # COPY FROM ASSIGN2_4
        # raise NotImplementedError
        self.q_projection = Linear(n_embd, n_embd, bias=bias, backend=self.backend)
        self.k_projection = Linear(n_embd, n_embd, bias=bias, backend=self.backend)
        self.v_projection = Linear(n_embd, n_embd, bias=bias, backend=self.backend)
        self.out_projection = Linear(n_embd, n_embd, bias=bias, backend=self.backend)
        self.dropout = Dropout(p_dropout)

    def create_causal_mask(self, bs, nh, seq_len):
        """
        return a 1x1xTxt triangular causal mask for Q @ K^T (which will get broadcasted to BxHxTxT)
        """
        # mask = -np.finfo(datatype).max * np.triu(np.ones((1, 1, seq_len, seq_len), dtype=datatype), 1) # This should be ok, but may be problematic -> the loss will be NaN in Assignment 3 because the mask will not broadcast correctly in the kernel.
        mask = -np.finfo(datatype).max * np.triu(np.ones((bs, nh, seq_len, seq_len), dtype=datatype), 1) # Correct version for Assignment 3.
        return tensor_from_numpy(mask, backend=self.backend)

    def project_to_query_key_value(self, x):
        """Project x to Q, transpose of K, V for self attention
        
        Args:
            x: embeddings or hidden states (batch_size x seq_len x n_embd)

        Returns:
            Q   : The Query Matrix (batch_size x num_heads x seq_len x attn_hidden_dim)
            K^T : The Key Matrix Transposed (batch_size x num_heads x attn_hidden_dim x seq_len)
            V   : The Value Matrix (batch_size x num_heads x seq_len x attn_hidden_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        
        # COPY FROM ASSIGN2_4
        # raise NotImplementedError
        q = self.q_projection(x).view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3)
        kT = self.k_projection(x).view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 3, 1)
        v = self.v_projection(x).view(batch_size, seq_len, self.n_head, self.attn_hidden_dim).permute(0, 2, 1, 3)
        
        return q, kT, v

    # def self_attention(self, q, kT, v):
    #     """Given q, kT, and v of sizes defined above, return the result of MultiHeadAttention as described in the writeup
    #     softmax((q @ kT) / sqrt(attn_hidden_dim)) @ V.
    #     NOTE: We have added support for Batch Matrix Multiplication with 4 dimensions.
    #     This means given tensors A of shape (a, b, m, n) and B of shape (a, b, n, p), 
    #     A @ B will be of the shape (a, b, m, p). Take a moment to consider why we need it.

    #     Args:
    #         q  : Queries Tensor of shape (batch_size x num_heads x seq_len x attn_hidden_dim)
    #         kT : Keys Tensor of shape (batch_size x num_heads x attn_hidden_dim x seq_len)
    #         v  : Values Tensor of shape (batch_size x num_heads x seq_len x attn_hidden_dim)

    #     Returns:
    #         output : Tensor of shape (batch_size, seq_len, n_embd)
    #     """
    #     batch_size, num_head, queries_len, q_dim = q.shape
    #     _, _, k_dim, _ = kT.shape
    #     _, _, _, v_dim = v.shape
    #     assert q_dim == k_dim == v_dim
    #     result = None
        
    #     if not self.use_fused_kernel:
    #         # COPY FROM ASSIGN2_4
    #         # raise NotImplementedError
    #         scores = q @ kT # (batch_size, num_heads, seq_len, seq_len)
    #         scores = scores / (self.attn_hidden_dim ** 0.5)
    #         if self.causal:
    #             scores = scores + self.create_causal_mask(queries_len)
            
    #         attn = softmax(scores, dim=3)
    #         attn = self.dropout(attn)
    #         result = attn @ v # (batch_size, num_heads, seq_len, attn_hidden_dim)

    #         result = result.contiguous()
    #         result = result.permute(0, 2, 1, 3)
    #         result = result.contiguous()
    #         result = result.view(batch_size, queries_len, self.n_embd)
    #         result = self.out_projection(result)
    #     else:
    #         # BEGIN ASSIGN3_3
    #         # raise NotImplementedError
    #         scores = q @ kT
    #         scores = scores / (self.attn_hidden_dim ** 0.5)
    #         mask = None
    #         if self.causal:
    #             mask = self.create_causal_mask(batch_size, num_head, queries_len)
            
    #         from .tensor_functions import Attn_Softmax
    #         attn = Attn_Softmax.apply(scores, mask if mask is not None else scores.zeros((batch_size, 1, 1, queries_len)))
    #         attn = self.dropout(attn)
    #         result = attn @ v
            
    #         result = result.contiguous()
    #         result = result.permute(0, 2, 1, 3)
    #         result = result.contiguous()
    #         result = result.view(batch_size, queries_len, self.n_embd)
    #         result = self.out_projection(result)
    #         # END ASSIGN3_3

    #     return result

    def self_attention(self, q, kT, v):
        """Given q, kT, and v of sizes defined above, return the result of MultiHeadAttention as described in the writeup
        softmax((q @ kT) / sqrt(attn_hidden_dim)) @ V.
        NOTE: We have added support for Batch Matrix Multiplication with 4 dimensions.
        This means given tensors A of shape (a, b, m, n) and B of shape (a, b, n, p), 
        A @ B will be of the shape (a, b, m, p). Take a moment to consider why we need it.

        Args:
            q  : Queries Tensor of shape (batch_size x num_heads x seq_len x attn_hidden_dim)
            kT : Keys Tensor of shape (batch_size x num_heads x attn_hidden_dim x seq_len)
            v  : Values Tensor of shape (batch_size x num_heads x seq_len x attn_hidden_dim)

        Returns:
            output : Tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, k_dim, _ = kT.shape
        _, _, _, v_dim = v.shape
        assert q_dim == k_dim == v_dim
        result = None
        
        if self.use_flash_attention:
            # Reshape kT back to (batch_size, num_heads, seq_len, head_dim)
            k = kT.permute(0, 1, 3, 2)
            # Use FlashAttention
            result = FlashAttention.apply(q, k, v, self.causal, 1.0 / (self.attn_hidden_dim ** 0.5))
            
            # Reshape the output for the output projection
            result = result.contiguous()
            result = result.permute(0, 2, 1, 3)
            result = result.contiguous()
            result = result.view(batch_size, queries_len, self.n_embd)
            result = self.out_projection(result)
        elif not self.use_fused_kernel:
            # COPY FROM ASSIGN2_4
            # Original implementation
            scores = q @ kT 
            scores = scores / (self.attn_hidden_dim ** 0.5)
            if self.causal:
                scores = scores + self.create_causal_mask(queries_len)
            
            attn = softmax(scores, dim=3)
            attn = self.dropout(attn)
            result = attn @ v

            result = result.contiguous()
            result = result.permute(0, 2, 1, 3)
            result = result.contiguous()
            result = result.view(batch_size, queries_len, self.n_embd)
            result = self.out_projection(result)
        else:
            # ASSIGN3_3 implementation
            scores = q @ kT
            scores = scores / (self.attn_hidden_dim ** 0.5)
            mask = None
            if self.causal:
                mask = self.create_causal_mask(batch_size, num_head, queries_len)
            
            from .tensor_functions import Attn_Softmax
            attn = Attn_Softmax.apply(scores, mask if mask is not None else scores.zeros((batch_size, 1, 1, queries_len)))
            attn = self.dropout(attn)
            result = attn @ v
            
            result = result.contiguous()
            result = result.permute(0, 2, 1, 3)
            result = result.contiguous()
            result = result.view(batch_size, queries_len, self.n_embd)
            result = self.out_projection(result)

        return result

    def forward(self, x):
        """Computes MultiHeadAttention with causal masking if needed. 

        Args:
            x : Tensor of shape (batch_size, seq_len, embedding_dim)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        # COPY FROM ASSIGN2_4
        # raise NotImplementedError
        q, kT, v = self.project_to_query_key_value(x)
        return self.self_attention(q, kT, v)


class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int=256, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """The Feed Forward Module.
        
        Args:
            n_embd     : in_size of first linear layer and out_size of last linear layer
            middle_dim : out_size of first linear layer and in_size of last linear layer
            p_dropout  : Dropout probability
            bias       : If bias should be applied in linear layers
        
        Attributes:
            linear_in  : first linear layer
            linear_out : second linear layer
            dropout    : dropout layer
        """
        # COPY FROM ASSIGN2_4
        # raise NotImplementedError
        self.linear_in  = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout    = Dropout(p_dropout)

    def forward(self, x):
        """A FFN Module in a Pre-LN Transformer with GELU Activation and dropout.

        Args:
            x : Tensor of shape (batch_size x seq_len x n_embd)

        Returns:
            output : Tensor of shape (batch_size x seq_len x n_embd)
        """
        batch_size, seq_len, n_embd = x.shape

        # COPY FROM ASSIGN2_4
        # raise NotImplementedError
        x = GELU(self.linear_in(x.view(batch_size * seq_len, n_embd)))
        x = self.dropout(self.linear_out(x)).view(batch_size, seq_len, n_embd)

        return x

class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float=0.1, ln_eps: float=1e-8, bias: bool=True, backend: TensorBackend=None, use_fused_kernel: bool=False):
        super().__init__()
        """A Transformer Layer in a Pre-LN Transformer.

        Args: 
            n_embd : Dimensionality of embeddings and hidden states
            n_head : Number of heads for MultiHeadAttention
            p_dropout : Dropout ratio for dropout layer
            ln_eps : A value added for numerical stability in LayerNorm
            bias : If bias should be added in linear layers
        
        Attributes:
            ln_1 : First LayerNorm1d layer before MultiHeadAttention
            ln_2 : Second LayerNorm1d layer after MultiHeadAttention
            attention : MultiHeadAttention layer
            ff : FeedForward layer
        """
        
        # COPY FROM ASSIGN2_4
        # self.attention
        # self.ff
        # raise NotImplementedError
        self.backend = backend
        self.attention = MultiHeadAttention(n_embd=n_embd, n_head=n_head, p_dropout=p_dropout, bias=bias, backend=self.backend)
        self.ff = FeedForward(n_embd=n_embd, middle_dim=4*n_embd, p_dropout=p_dropout, bias=bias, backend=self.backend)

        self.use_fused_kernel = use_fused_kernel
        if not self.use_fused_kernel:
            # COPY FROM ASSIGN2_4
            # self.ln_1
            # self.ln_2
            # raise NotImplementedError
            self.ln_1 = LayerNorm1d(dim=n_embd, eps=ln_eps, backend=self.backend)
            self.ln_2 = LayerNorm1d(dim=n_embd, eps=ln_eps, backend=self.backend)
        else:
            # BEGIN ASSIGN3_3
            # raise NotImplementedError
            self.gamma1 = Parameter(tensor([1.0] * n_embd, backend=self.backend))
            self.beta1 = Parameter(tensor([0.0] * n_embd, backend=self.backend))
            self.gamma2 = Parameter(tensor([1.0] * n_embd, backend=self.backend))
            self.beta2 = Parameter(tensor([0.0] * n_embd, backend=self.backend))
            # END ASSIGN3_3

    def forward(self, x):
        """
        The forward function of a Transformer Layer for a PRENORM Transformer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """
        batch_size, seq_len, x_dim = x.shape
        
        if not self.use_fused_kernel:
            # COPY FROM ASSIGN2_4
            # raise NotImplementedError
            attn_ln = self.ln_1(x)
            attn_out = self.attention(attn_ln)
            x = x + attn_out
            ff_ln = self.ln_2(x)
            ff_out = self.ff(ff_ln)
            return x + ff_out
        else:
            # BEGIN ASSIGN3_3
            # raise NotImplementedError
            from .tensor_functions import LayerNorm
            reshaped_x = x.view(batch_size * seq_len, x_dim)
            attn_ln = LayerNorm.apply(reshaped_x, self.gamma1.value, self.beta1.value)
            attn_ln = attn_ln.view(batch_size, seq_len, x_dim)
            attn_out = self.attention(attn_ln)
            x = x + attn_out
            reshaped_x = x.view(batch_size * seq_len, x_dim)
            ff_ln = LayerNorm.apply(reshaped_x, self.gamma2.value, self.beta2.value)
            ff_ln = ff_ln.view(batch_size, seq_len, x_dim)
            ff_out = self.ff(ff_ln)
            return x + ff_out
            # END ASSIGN3_3

        return x


class DecoderLM(Module):
    def __init__(
        self, 
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        p_dropout: float=0.1,
        ln_eps: float=1e-5, 
        bias: bool=True,
        backend: TensorBackend=None,
        use_fused_kernel: bool=False,
    ):
        super().__init__()
        """A Full Decoder-only Pre-LN Transformer with 4 Transformer Layers.

        Args:
            n_vocab : Vocabulary size defines the number of different tokens that can be represented by the input.
            n_embd  :  Dimensionality of the embeddings and hidden states.
            n_head  : Number of attention heads for each attention layer in the Transformer.
            n_positions : The maximum sequence length that this model might ever be used with.
            p_dropout : The dropout ratio for any dropout layer.
            ln_eps : The epsilon to use in the layer normalization layers.
            bias : If linear layers should include a bias.
        
        Attributes:
            token_embeddings : Embedding layer for tokens.
            position_embeddings : Embedding layer for token positions.
            t_layer_1 : 1st Transformer Layer.
            t_layer_2 : 2nd Transformer Layer.
            t_layer_3 : 3rd Transformer Layer.
            t_layer_4 : 4th Transformer Layer.
            dropout : Dropout layer before first transformer layer.
            ln : LayerNorm layer after last transformer layer.
            lm_head : Linear layer for projection from (*, n_embd) to (*, n_vocab)
        """
        self.backend             = backend
        self.n_embd              = n_embd
        self.n_vocab             = n_vocab
        
        # COPY FROM ASSIGN2_4
        # self.token_embeddings    = 
        # self.position_embeddings = 
        # self.t_layer_1           = 
        # self.t_layer_2           = 
        # self.t_layer_3           = 
        # self.t_layer_4           = 
        # self.dropout             = 
        # self.lm_head             = 
        # raise NotImplementedError
        self.token_embeddings    = Embedding(n_vocab, n_embd, backend)
        self.position_embeddings = Embedding(n_positions, n_embd, backend)
        self.t_layer_1           = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_2           = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_3           = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_4           = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.dropout             = Dropout(p_dropout)
        self.lm_head             = Linear(n_embd, n_vocab, bias, backend)

        self.use_fused_kernel = use_fused_kernel
        if not self.use_fused_kernel:
            # COPY FROM ASSIGN2_4
            # self.ln                  = 
            # raise NotImplementedError
            self.ln              = LayerNorm1d(n_embd, eps=ln_eps, backend=backend)
        else:
            # BEGIN ASSIGN3_3
            # raise NotImplementedError
            self.gamma_final = Parameter(tensor([1.0] * n_embd, backend=self.backend))
            self.beta_final = Parameter(tensor([0.0] * n_embd, backend=self.backend))            
            # END ASSIGN3_3
        
    def forward(self, idx):
        """A Forward pass of a Decoder-only Transformer Language model.
        Args: 
            idx: input of shape (batch_size, seq_len)
        
        Returns: 
            logits: logits of shape (batch_size, seq_len, n_vocab)
        """
        
        batch_size, seq_len = idx.shape
        pos = tensor([i for i in range(seq_len)], backend=self.backend).view(1, seq_len)

        if not self.use_fused_kernel:
            # COPY FROM ASSIGN2_4
            # raise NotImplementedError
            token_embeddings = self.token_embeddings(idx)

            # positions = tensor_from_numpy(np.arange(seq_len).reshape(1, seq_len).repeat(batch_size, axis=0), 
            #                 backend=self.backend)
            # positional_embeddings = self.position_embeddings(positions)
            positional_embeddings = self.position_embeddings(pos)

            x = self.dropout(token_embeddings + positional_embeddings)
            # Pass through each transformer Layer
            x = self.t_layer_1(x)
            x = self.t_layer_2(x)
            x = self.t_layer_3(x)
            x = self.t_layer_4(x)
            # Final LayerNorm
            x = self.ln(x)
            # Get correct shape
            x = self.lm_head(x)
        else:
            # BEGIN ASSIGN3_3
            # raise NotImplementedError
            # 稍微注意一下上面改动的部分
            from .tensor_functions import LayerNorm
            token_embeddings = self.token_embeddings(idx)
            positional_embeddings = self.position_embeddings(pos)
            x = self.dropout(token_embeddings + positional_embeddings)
            x = self.t_layer_1(x)
            x = self.t_layer_2(x)
            x = self.t_layer_3(x)
            x = self.t_layer_4(x)
            
            reshaped_x = x.view(batch_size * seq_len, self.n_embd)
            normalized_x = LayerNorm.apply(reshaped_x, self.gamma_final.value, self.beta_final.value)
            x = normalized_x.view(batch_size, seq_len, self.n_embd)
            x = self.lm_head(x)           
            # END ASSIGN3_3

        return x