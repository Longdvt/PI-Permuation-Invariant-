import numpy as np
import torch
import torch.nn as nn
import argparse

def pos_table(n, dim):
    """
    Create a table of position encodings
    
    Inputs:
        n: number of positions
        dim: dimension of the encoding
    Returns:
        table: (n, dim) position encoding table
    """
    def get_angle(pos, i):
        return pos / np.power(10000, 2 * (i // 2) / dim)
    
    def get_angle_vector(pos):
        return [get_angle(pos, i) for i in range(dim)]
    
    table = np.array([get_angle_vector(pos) for pos in range(n)]).astype(np.float32)
    table[:, 0::2] = np.sin(table[:, 0::2])
    table[:, 1::2] = np.cos(table[:, 1::2])
    return table

class AttentionMatrix(nn.Module):
    """
    Self-attention matrix.

    Inputs:
        dim_in_q: dimension of the query
        dim_in_k: dimension of the key
        msg_dim: dimension of the message
        bias: whether to use bias
        scale: whether to scale the message
    """
    def __init__(self,
                 dim_in_q,
                 dim_in_k,
                 msg_dim,
                 bias=True,
                 scale=True):
        super(AttentionMatrix, self).__init__()
        self.proj_q = nn.Linear(dim_in_q, msg_dim, bias=bias)
        self.proj_k = nn.Linear(dim_in_k, msg_dim, bias=bias)
        if scale:
            self.scale = np.sqrt(msg_dim)
        else:
            self.scale = 1.0

    def forward(self, data_q, data_k):
        q = self.proj_q(data_q)
        k = self.proj_k(data_k)
        if data_q.ndim == data_k.ndim == 2:
            dot = torch.matmul(q, k.T)
        else:
            dot = torch.bmm(q, k.permute(0, 2, 1))
        return torch.div(dot, np.sqrt(self.scale))
    
class SelfAttentionMatrix(AttentionMatrix):
    """"
    Self-attention matrix.
    """
    def __init__(self,
                 dim_in,
                 msg_dim,
                 bias=True,
                 scale=True):
        super(SelfAttentionMatrix, self).__init__(
            dim_in_q=dim_in,
            dim_in_k=dim_in,
            msg_dim=msg_dim,
            bias=bias,
            scale=scale)

class AttentionLayer(nn.Module):
    """
    The attention mechanism.

    Inputs:
        dim_in_q: dimension of the query
        dim_in_k: dimension of the key
        dim_in_v: dimension of the value
        msg_dim: dimension of the message
        out_dim: dimension of the output
    """
    def __init__(self,
                 dim_in_q,
                 dim_in_k,
                 dim_in_v,
                 msg_dim,
                 out_dim):
        super(AttentionLayer, self).__init__()
        self.attention_matrix = AttentionMatrix(
            dim_in_q=dim_in_q,
            dim_in_k=dim_in_k,
            msg_dim=msg_dim)
        self.proj_v = nn.Linear(dim_in_v, out_dim)
        self.mostly_attened_entries = None

    def forward(self, data_q, data_k, data_v):
        a = torch.softmax(self.attention_matrix(data_q, data_k), dim=-1)
        self.mostly_attened_entries = set(torch.argmax(a, dim=-1).numpy())
        v = self.proj_v(data_v)
        return torch.matmul(a, v)
        
class AttentionNeuronLayer(nn.Module):
    """
    Permutation invariant layer. (Permute the dimension of the input)

    Inputs:
        act_dim: dimension of the action
        hidden_dim: dimension of the hidden layer
        msg_dim: dimension of the message
        pos_em_dim: dimension of the position embedding
        bias: whether to use bias
        scale: whether to scale the message
    """
    def __init__(self,
                 act_dim,
                 hidden_dim,
                 msg_dim,
                 pos_em_dim,
                 bias=True,
                 scale=True):
        super(AttentionNeuronLayer, self).__init__()
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.pos_em_dim = pos_em_dim
        self.pos_embedding = torch.from_numpy(pos_table(n=self.hidden_dim, dim=self.pos_em_dim)).float()
        self.hx = None
        self.lstm = nn.LSTMCell(
            input_size=1+self.act_dim,
            hidden_size=pos_em_dim)
        self.attention = SelfAttentionMatrix(
            dim_in=pos_em_dim,
            msg_dim=self.msg_dim,
            bias=bias,
            scale=scale)
        
    def forward(self, obs, prev_act):
        """
        Inputs:
            obs: observation [obs_dim]
            prev_act: previous action [act_dim]
        Returns:
            output: output vector [hidden_dim]
        """
        if isinstance(obs, np.ndarray):
            x = torch.from_numpy(obs.copy()).float().unsqueeze(-1)
        else:
            x = obs.unsqueeze(-1)
        obs_dim = x.shape[0]

        x_aug = torch.cat([x, torch.vstack([prev_act] * obs_dim)], dim=-1)
        if self.hx is None:
            self.hx = (torch.zeros(obs_dim, self.pos_em_dim).to(x.device),
                       torch.zeros(obs_dim, self.pos_em_dim).to(x.device))
            
        self.hx = self.lstm(x_aug, self.hx)
        w = torch.tanh(self.attention(data_q=self.pos_embedding.to(x.device),
                                      data_k=self.hx[0]))
        output = torch.matmul(w, x)
        return torch.tanh(output)
    
    def reset(self):
        self.hx = None

class VisionAttentionNeuronLayer(nn.Module):
    """
    Permutation invariant layer for vision tasks. (Permute the patches of image)
    """
    def __init__(self,
                 act_dim,
                 hidden_dim,
                 msg_dim,
                 pos_em_dim,
                 patch_size=6,
                 stack_k=4,
                 with_learnable_ln_params=False,
                 stack_dim_first=False):
        super(VisionAttentionNeuronLayer, self).__init__()
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.pos_em_dim = pos_em_dim
        self.patch_size = patch_size
        self.stack_k = stack_k
        self.stack_dim_first = stack_dim_first
        self.pos_embedding = torch.from_numpy(pos_table(n=self.hidden_dim, dim=self.pos_em_dim)).float()
        self.attention = AttentionLayer(
            dim_in_q=self.pos_em_dim,
            dim_in_k=(self.stack_k-1) * self.patch_size * self.patch_size + self.act_dim,
            dim_in_v=self.stack_k * self.patch_size * self.patch_size,
            msg_dim=self.msg_dim,
            out_dim=self.msg_dim)
        # The normalization layers have no learnable parameters
        self.input_ln = nn.LayerNorm(
            normalized_shape=(self.patch_size**2),
            elementwise_affine=with_learnable_ln_params)
        self.input_ln.eval()
        self.output_ln = nn.LayerNorm(
            normalized_shape=self.msg_dim,
            elementwise_affine=with_learnable_ln_params)
        self.output_ln.eval()

    def get_patches(self, x):
        """"
        Inputs:
            x: input image [h, w, c]
        Returns:
            patches: [batch, patch_size, patch_size, c]
        """
        h, w, c = x.size()
        assert h % self.patch_size == 0 and w % self.patch_size == 0
        patches = x.unfold(0, self.patch_size, self.patch_size).unfold(1, self.patch_size, self.patch_size)
        patches = patches.reshape(-1, self.patch_size, self.patch_size, c)
        return patches
    
    def forward(self, obs, prev_act, drop_some_patches=False, num_patches_drop=None):
        k, h, w = obs.shape
        assert k == self.stack_k
        if drop_some_patches:
            assert num_patches_drop is not None 
            num_patches = num_patches_drop.size
        else:
            num_patches = (h // self.patch_size) * (w // self.patch_size)
        # AttentionNeuron is the first layer, so obs is numpy array
        x_obs = torch.div(torch.from_numpy(obs).float(), 255.0)
        # create Key
        x_k = torch.diff(x_obs, dim=0).permute(1, 2, 0)
        x_k = self.get_patches(x_k)
        if drop_some_patches:
            x_k = x_k[num_patches_drop]
        assert x_k.shape == (num_patches, self.patch_size, self.patch_size, self.stack_k-1)
        if self.stack_dim_first:
            x_k = x_k.permute(0, 3, 1, 2)
        x_k = torch.cat([
            torch.flatten(x_k, start_dim=1),
            torch.repeat_interleave(prev_act.unsqueeze(0), num_patches, dim=0)], dim=-1)
        # create Value
        x_v = self.get_patches(x_obs.permute(1, 2, 0)).permute(0, 3, 1, 2)
        if drop_some_patches:
            x_v = x_v[num_patches_drop]
        x_v = self.input_ln(torch.flatten(x_v, start_dim=2))

        x = self.attention(
            data_q=self.pos_embedding,
            data_k=x_k,
            data_v=x_v.reshape(num_patches, -1))
        return self.output_ln(torch.relu(x))

class AttentionNeuronSequenceVector(nn.Module):
    """
    Permutation invariant sequence. (Permute the time dimension of the input)
    """
    def __init__(self,
                 obs_dim,
                 act_dim,
                 hidden_dim,
                 msg_dim,
                 pos_em_dim,
                 out_dim):
        super(AttentionNeuronSequenceVector, self).__init__()
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.msg_dim = msg_dim
        self.pos_em_dim = pos_em_dim
        self.out_dim = out_dim
        self.pos_embedding = torch.from_numpy(pos_table(n=self.hidden_dim, dim=self.pos_em_dim)).float()
        self.attention = AttentionLayer(dim_in_q=self.pos_em_dim,
                                        dim_in_k=obs_dim + self.act_dim,
                                        dim_in_v=obs_dim,
                                        msg_dim=self.msg_dim,
                                        out_dim=self.out_dim)

    def forward(self, sequence, pre_a):
        assert sequence.ndim == 2
        seq_len, _ = sequence.shape
        if isinstance(sequence, np.ndarray):
            x = torch.from_numpy(sequence.copy()).float()
        else:
            x = sequence
        x_aug = torch.cat([x, torch.vstack([pre_a] * seq_len)], dim=-1)
        w = torch.tanh(self.attention(data_q=self.pos_embedding,
                                        data_k=x_aug,
                                        data_v=x))
        return w

class Policy_Invariant(nn.Module):
    """
    Permutation invariant policy.
    """
    def __init__(self,
                 act_dim,
                 hidden_dim,
                 msg_dim,
                 pos_em_dim):
        super(Policy_Invariant, self).__init__()
        self.attention_neuron = AttentionNeuronLayer(
            act_dim=act_dim,
            hidden_dim=hidden_dim,
            msg_dim=msg_dim,
            pos_em_dim=pos_em_dim)
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh())
        
    def forward(self, obs, prev_act):
        msg = self.attention_neuron(obs, prev_act)
        return self.policy_net(msg.T)

    
if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--test_attention_neuron_sequence", type=bool, default=True)
    argument_parser.add_argument("--obs_dim", type=int, default=64, help="Dimension of the observation")
    argument_parser.add_argument("--act_dim", type=int, default=8, help="Dimension of the previous action")
    argument_parser.add_argument("--hidden_dim", type=int, default=128, help="Number of positions (Number of Query)")
    argument_parser.add_argument("--pos_em_dim", type=int, default=1024, help="Dimension of the position embedding (Dim of Query)")
    argument_parser.add_argument("--msg_dim", type=int, default=512, help="Dimension of the message (Dim of Key and Value)")
    argument_parser.add_argument("--out_dim", type=int, default=256, help="Dimension of the output")
    argument_parser.add_argument("--device", type=str, default="cpu", help="Device to use")
    args = argument_parser.parse_args()

    device = torch.device(args.device)
    if args.test_attention_neuron_sequence:
        print("Return a tensor shape of (hidden_dim, msg_dim)")
        model = AttentionNeuronSequenceVector(obs_dim=args.obs_dim,
                                              act_dim=args.act_dim,
                                              hidden_dim=args.hidden_dim,
                                              msg_dim=args.msg_dim,
                                              pos_em_dim=args.pos_em_dim,
                                              out_dim=args.out_dim).to(device)
        x = torch.randn(10, args.obs_dim)
        pre_a = torch.randn(args.act_dim)
        output = model(x, pre_a)
        print(output.shape)
        x1 = x[torch.randperm(10)]
        output1 = model(x1, pre_a)
        print(output1.shape)
        print(torch.allclose(output, output1, atol=1e-6))

    
    # model = Policy_Invariant(
    #     act_dim=8,
    #     hidden_dim=32,
    #     msg_dim=32,
    #     pos_em_dim=8).to(torch.device("cpu"))
    
    # model.attention_neuron.reset()
    # obs = torch.randn(28)
    # prev_act = torch.randn(8)
    # output = model(obs, prev_act)
    # print(output)
    # obs1 = obs[torch.randperm(28)]
    # output1 = model(obs1, prev_act)
    # print(output1)

    # model = VisionAttentionNeuronLayer(act_dim=3,
    #                                    hidden_dim=400,
    #                                    msg_dim=16,
    #                                    pos_em_dim=1024,
    #                                    patch_size=6,
    #                                    stack_k=4,
    #                                    with_learnable_ln_params=False,
    #                                    stack_dim_first=False).to(torch.device("cpu"))
    # obs = np.random.randint(0, 256, size=(4, 96, 96))
    # prev_act = torch.randn(3)
    # output = model(obs, prev_act)
    # patches = model.get_patches(torch.from_numpy(obs).float().permute(1, 2, 0))
    # patches = patches[torch.randperm(patches.shape[0])]
    # unfolded_shape = (16, 16, 4, 6, 6)
    # patches = patches.reshape(unfolded_shape).permute(2, 0, 3, 1, 4).reshape(4, 96, 96)
    # obs_permute = patches.numpy()
    # output_permute = model(obs_permute, prev_act)
    # print(torch.allclose(output, output_permute, atol=1e-6))
