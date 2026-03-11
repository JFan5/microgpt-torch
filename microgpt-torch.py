import time, os, random, torch, torch.nn.functional as F, json

start = time.time()

if not os.path.exists('input.txt'):
    import urllib.request
    urllib.request.urlretrieve('https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt', 'input.txt')
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()]
random.shuffle(docs)
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'docs={len(docs)} vocab={vocab_size} device={device}')

n_embd, n_head, n_layer, block_size = 16, 4, 1, 16
head_dim = n_embd // n_head
matrix = lambda r, c: torch.randn(r, c, device=device) * 0.08
sd = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
for i in range(n_layer):
    for k in ['attn_wq', 'attn_wk', 'attn_wv', 'attn_wo']:
        sd[f'layer{i}.{k}'] = matrix(n_embd, n_embd)
    sd[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    sd[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = list(sd.values())
for p in params:
    p.requires_grad = True
print(f'params={sum(p.numel() for p in params)}')

def gpt(tids):
    s = len(tids)
    x = F.rms_norm(sd['wte'][tids] + sd['wpe'][:s], (n_embd,), eps=1e-5)
    for li in range(n_layer):
        xr = x; xn = F.rms_norm(x, (n_embd,), eps=1e-5)
        q = F.linear(xn, sd[f'layer{li}.attn_wq'])
        k = F.linear(xn, sd[f'layer{li}.attn_wk'])
        v = F.linear(xn, sd[f'layer{li}.attn_wv'])
        q = q.view(s, n_head, head_dim).transpose(0, 1)
        k = k.view(s, n_head, head_dim).transpose(0, 1)
        v = v.view(s, n_head, head_dim).transpose(0, 1)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(0, 1).contiguous().view(s, n_embd)
        x = F.linear(o, sd[f'layer{li}.attn_wo']) + xr
        xr = x; x = F.rms_norm(x, (n_embd,), eps=1e-5)
        x = F.relu(F.linear(x, sd[f'layer{li}.mlp_fc1']))
        x = F.linear(x, sd[f'layer{li}.mlp_fc2']) + xr
    return F.linear(x, sd['lm_head'])

opt = torch.optim.Adam(params, lr=0.01, betas=(0.85, 0.99), eps=1e-8)
ne = 40; nd = len(docs); ts = ne * nd
sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda s: 1 - s / ts)
epoch_losses = []
step = 0
for ep in range(ne):
    ls = 0.0
    for di in range(nd):
        t = [BOS] + [uchars.index(c) for c in docs[di]] + [BOS]
        n = min(block_size, len(t) - 1)
        logits = gpt(t[:n])
        loss = F.cross_entropy(logits, torch.tensor(t[1:n+1], device=device))
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        ls += loss.item(); step += 1
    al = ls / nd
    epoch_losses.append(al)
    print(f'epoch {ep+1:3d}/40 | avg loss {al:.4f}')

elapsed = time.time() - start
print(f'\nTraining time: {elapsed:.1f}s ({elapsed/60:.1f} min)')

with open('training_losses.json', 'w') as f:
    json.dump({'epoch_losses': epoch_losses, 'elapsed_seconds': elapsed}, f)

print('\n--- inference ---')
with torch.no_grad():
    for si in range(20):
        tids = [BOS]
        for _ in range(block_size):
            l = gpt(tids)
            p = F.softmax(l[-1] / 0.5, -1)
            ti = torch.multinomial(p, 1).item()
            if ti == BOS: break
            tids.append(ti)
        print(f'sample {si+1:2d}: {"".join(uchars[t] for t in tids[1:])}')
