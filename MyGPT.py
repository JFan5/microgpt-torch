import os
import random
import torch
import torch.nn.functional as F

# Data: a list of English names, each name is a document, each letter is a token | 数据：一系列英文名字，每个名字看成一个文档，每个字母看成一个token
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")


# Tokenizer: assign an id to each letter, 26 letters + 1 special BOS token | 分词器：给每个字母一个编号，总共26个字母 + 1个特殊的BOS token
uchars = sorted(set(''.join(docs))) # unique character set | 单独的字符集合
BOS = len(uchars) # BOS token id, placed at the end, marks the start of each name | BOS token的编号，放在最后，用来标记每个名字的开始
vocab_size = len(uchars) + 1 # number of unique characters: 26 letters + 1 BOS token | 单独字符的数量：26个字母 + 1个BOS token
print(f"vocab size: {vocab_size}")

# Device selection: prefer CUDA GPU | 设备选择：优先使用 CUDA GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

n_embd = 16  # encode each letter as a 16-dim vector | 把一个字母编码成16维的向量
n_head = 4 # number of attention heads, each captures a different relationship between positions | 注意力头的数量，每个头捕捉输入序列中不同位置之间的一种关系
n_layer = 1 # number of transformer blocks, each contains a self-attention and an MLP module | transformer架构的数量，每个包含一个self-attention模块和一个MLP模块
block_size = 16 # max number of output tokens, 16 letters is enough for names | 允许输出token的最大数量，对于名字来说16个字母已经足够了
head_dim = n_embd // n_head # dimension per attention head: 16-dim embedding split into 4 heads of 4-dim each | 每个注意力头的维度：16维embedding分成4个头，每个头4维

# Parameter initialization with torch tensors | 用torch张量初始化参数
matrix = lambda nout, nin, std=0.08: torch.randn(nout, nin, device=device) * std

# Create state_dict: wte (word token embedding), wpe (position embedding), lm_head (language model head) | 创建state_dict：wte（词token嵌入），wpe（位置嵌入）和lm_head（语言模型头）
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}

# For each layer, create attention and MLP weight matrices | 对每一个layer，创建注意力和MLP的权重矩阵
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

# Set all parameters to trainable | 设置所有参数为可训练
params = list(state_dict.values())
for p in params:
    p.requires_grad = True
num_params = sum(p.numel() for p in params)
print(f"num params: {num_params}")


def rmsnorm(x):
    # RMS normalization, x: [seq_len, n_embd] | RMS归一化，x: [seq_len, n_embd]
    ms = (x * x).mean(dim=-1, keepdim=True)
    return x * torch.rsqrt(ms + 1e-5)


# Forward pass: input token sequence, output [seq_len, vocab_size] logits | 前向传播：输入token序列，输出 [seq_len, vocab_size] 的logits
def gpt(token_ids):
    # token_ids: list of int, length = seq_len | token_ids：整数列表，长度为seq_len
    seq_len = len(token_ids)
    tok_emb = state_dict['wte'][token_ids]          # token embedding | 词嵌入
    pos_emb = state_dict['wpe'][:seq_len]            # position embedding | 位置嵌入
    x = rmsnorm(tok_emb + pos_emb)                   # joint embedding + normalize | 联合嵌入 + 归一化

    for li in range(n_layer):
        # 1) Multi-head attention block | 多头注意力块
        x_residual = x
        x_norm = rmsnorm(x)
        # Linear projections for query, key, value | 线性变换得到query、key、value
        q = x_norm @ state_dict[f'layer{li}.attn_wq'].T  # [seq_len, n_embd]
        k = x_norm @ state_dict[f'layer{li}.attn_wk'].T  # [seq_len, n_embd]
        v = x_norm @ state_dict[f'layer{li}.attn_wv'].T  # [seq_len, n_embd]

        # Split into multiple heads: [seq_len, n_head, head_dim] -> [n_head, seq_len, head_dim] | 分成多个头
        q = q.view(seq_len, n_head, head_dim).transpose(0, 1)
        k = k.view(seq_len, n_head, head_dim).transpose(0, 1)
        v = v.view(seq_len, n_head, head_dim).transpose(0, 1)

        # Compute attention weights with causal mask to prevent attending to future tokens | 计算注意力权重，使用causal mask确保只看之前的token
        attn_logits = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)  # [n_head, seq_len, seq_len]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        attn_logits = attn_logits.masked_fill(causal_mask, float('-inf'))
        attn_weights = torch.softmax(attn_logits, dim=-1)  # [n_head, seq_len, seq_len]

        # Weighted sum of values to get attention output | 用注意力权重加权求和v，得到注意力输出
        attn_out = attn_weights @ v  # [n_head, seq_len, head_dim]
        # Concatenate heads and project back | 拼接所有头并线性映射回n_embd维
        attn_out = attn_out.transpose(0, 1).contiguous().view(seq_len, n_embd)  # [seq_len, n_embd]

        # Output projection + residual connection | 输出投影 + 残差连接
        x = attn_out @ state_dict[f'layer{li}.attn_wo'].T + x_residual

        # 2) MLP block | MLP块
        x_residual = x
        x = rmsnorm(x)
        x = x @ state_dict[f'layer{li}.mlp_fc1'].T  # project to 4*n_embd | 映射到4*n_embd维
        x = torch.relu(x)                            # ReLU activation | ReLU激活
        x = x @ state_dict[f'layer{li}.mlp_fc2'].T  # project back to n_embd | 映射回n_embd维
        x = x + x_residual                           # residual connection | 残差连接

    logits = x @ state_dict['lm_head'].T  # project to vocab_size for next-token prediction | 映射到vocab_size维，用于下一个token的预测
    return logits


# Training | 训练
learning_rate = 0.01
optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.85, 0.99), eps=1e-8)

num_epochs = 40
num_docs = len(docs)
total_steps = num_epochs * num_docs
# Linear learning rate decay | 线性学习率衰减
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1 - step / total_steps)

step = 0
for epoch in range(num_epochs):
    for doc_idx in range(num_docs):

        # Take a document, tokenize it, surround with BOS token | 取一个文档，分词，两端加BOS token
        doc = docs[doc_idx]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        # Forward pass: compute logits for the entire sequence at once | 前向传播：一次性计算整个序列
        input_ids = tokens[:n]
        target_ids = torch.tensor(tokens[1:n+1], device=device)

        logits = gpt(input_ids)  # [n, vocab_size]
        loss = F.cross_entropy(logits, target_ids)

        # Backward pass + optimizer update | 反向传播 + 优化器更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        step += 1
        print(f"epoch {epoch+1:3d}/{num_epochs:3d} | step {step:4d}/{total_steps:4d} | loss {loss.item():.4f}")

# Inference: generate new names | 推理：生成新名字
temperature = 0.5 # in (0, 1], controls "creativity" of generated text | 控制生成文本的"创造性"，值越低越保守
print("\n--- inference (new, hallucinated names) ---")
with torch.no_grad():
    for sample_idx in range(20):
        token_ids = [BOS]
        for pos_id in range(block_size):
            logits = gpt(token_ids)
            # Take logits of the last token only | 只取最后一个token的logits
            next_logits = logits[-1] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            token_id = torch.multinomial(probs, 1).item()
            if token_id == BOS:
                break
            token_ids.append(token_id)
        name = ''.join(uchars[t] for t in token_ids[1:])
        print(f"sample {sample_idx+1:2d}: {name}")
