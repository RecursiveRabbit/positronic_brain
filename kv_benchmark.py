import os
import glob
import time
import argparse
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download
from safetensors.torch import load_file  # zero-copy loader for .safetensors

# Configuration
MODEL_NAME = "moonshotai/Kimi-VL-A3B-Thinking"
TRUST_REMOTE_CODE = True

# ── Slim safetensors loader with debug printing and config ──────────────────
def load_slim_weights(model_name=MODEL_NAME, debug=False):
    """
    Fully dynamic loader for any safetensors transformer:
    - Auto-discovers layer prefixes and all Q/K/V/rope keys
    - Infers all dimensions from tensors, not config
    - Handles arbitrary naming/prefixing/nesting
    - Only loads inv_freq, computes RoPE on-the-fly
    - Handles optional bias
    - Maps all discovered weights to standardized internal keys
    - Handles MQA/GQA
    """
    import re, fnmatch
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=TRUST_REMOTE_CODE)
    repo_dir = snapshot_download(model_name, repo_type="model", allow_patterns=["*.safetensors"])
    sd = {}
    for path in sorted(glob.glob(os.path.join(repo_dir, "model-*.safetensors"))):
        sd.update(load_file(path))
    # Find embedding
    embed_key = next((k for k in ["model.embed_tokens.weight","embed_tokens.weight"] if k in sd), None)
    if embed_key is None:
        embed_key = next(k for k in sd if 'embed' in k.lower() and k.endswith('.weight'))
    wte = sd[embed_key]
    vocab_size, hidden_size = wte.shape
    # Store embedding key name for debugging
    slim = {
        "wte.weight": wte,
        "wte.key_name": embed_key,  # Store original key name
        "_vocab_size": vocab_size,
        "_hidden_size": hidden_size
    }
    # Discover layer prefixes using regex for any ...layers.N... or similar
    layer_num_re = re.compile(r"^(.*?\b(?:layer|layers|block|blocks)[_.]?)(\d+)([_.].*)?\.((?:self_attn|attention|attn)[_.])?")
    prefixes = set()
    for k in sd:
        m = layer_num_re.match(k)
        if m:
            # Compose prefix up to layer number
            base = m.group(1) + m.group(2)
            prefixes.add(base)
    layer_prefixes = sorted(prefixes, key=lambda p: int(re.findall(r"(\d+)", p)[-1]))
    if debug:
        print(f"DEBUG loaded {len(sd)} tensors")
        for k in list(sd.keys())[:20]:
            print(f"  - {k}")
        print("Discovered layer prefixes:", layer_prefixes)
    # Filter for language model layers only
    language_layer_prefixes = [p for p in layer_prefixes if "language_model" in p]
    if debug:
        print(f"Filtered down to {len(language_layer_prefixes)} language model prefixes: {language_layer_prefixes}")
    # Key patterns for Q/K/V/rope
    q_patterns = ["*q_proj.weight", "*query_proj.weight", "*q_proj_with_mqa.weight", "*q_proj_with_gqa.weight", "*kv_a_proj_with_mqa.weight", "*q_proj*weight"]
    k_patterns = ["*k_proj.weight", "*key_proj.weight", "*k_proj_with_mqa.weight", "*kv_a_proj_with_mqa.weight", "*k_proj*weight"]

    inv_patterns = ["*inv_freq"]
    bias_patterns = [p.replace("weight","bias") for p in (q_patterns + k_patterns)]
    def find_first_key(patterns, keys):
        for pat in patterns:
            for k in keys:
                if fnmatch.fnmatch(k, pat):
                    return k
        return None
    # Compose slim dict

    import textwrap
    for idx, prefix in enumerate(language_layer_prefixes):
        # Find all keys for this layer
        layer_keys = [k for k in sd if k.startswith(prefix)]
        # Q
        qk = find_first_key(q_patterns, layer_keys)
        if qk is None:
            if debug:
                print(f"\nERROR: Could not find Q projection weight for layer {idx} (prefix: {prefix})")
                print(f"  Tried patterns: {q_patterns}")
                print(f"  Available keys in layer ({len(layer_keys)} total):")
                keys_str = ", ".join(sorted(layer_keys))
                wrapped_keys = textwrap.fill(keys_str, width=100, subsequent_indent='    ')
                print(f"    {wrapped_keys}")
            raise ValueError(f"Missing Q projection weight for layer {idx} (prefix: {prefix}). See debug log for available keys.")
        # K
        kk = find_first_key(k_patterns, layer_keys)
        if kk is None:
            if debug:
                print(f"\nERROR: Could not find K projection weight for layer {idx} (prefix: {prefix})")
                print(f"  Tried patterns: {k_patterns}")
                print(f"  Available keys in layer ({len(layer_keys)} total):")
                keys_str = ", ".join(sorted(layer_keys))
                wrapped_keys = textwrap.fill(keys_str, width=100, subsequent_indent='    ')
                print(f"    {wrapped_keys}")
            raise ValueError(f"Missing K projection weight for layer {idx} (prefix: {prefix}). See debug log for available keys.")
        slim[f"l{idx}.q_proj.w"] = sd[qk]
        slim[f"l{idx}.q_proj.key_name"] = qk

        # V (uses K's key)
        vk = kk
        slim[f"l{idx}.k_proj.w"] = sd[kk]
        slim[f"l{idx}.k_proj.key_name"] = kk
        slim[f"l{idx}.v_proj.w"] = sd[vk]
        slim[f"l{idx}.v_proj.key_name"] = vk

        # inv_freq
        invk = find_first_key(inv_patterns, layer_keys)
        if invk is None:
            if debug:
                print(f"\nERROR: Could not find inv_freq for layer {idx} (prefix: {prefix})")
                print(f"  Tried patterns: {inv_patterns}")
                print(f"  Available keys in layer ({len(layer_keys)} total):")
                keys_str = ", ".join(sorted(layer_keys))
                wrapped_keys = textwrap.fill(keys_str, width=100, subsequent_indent='    ')
                print(f"    {wrapped_keys}")
            raise ValueError(f"Missing rotary inv_freq for layer {idx} (prefix: {prefix}). See debug log for available keys.")
        slim[f"l{idx}.inv_freq"] = sd[invk]
        slim[f"l{idx}.inv_freq.key_name"] = invk

        # Debug prints
        if debug:
            print(f"Layer {idx} prefix {prefix}")
            print(f"  Q: {qk}")
            print(f"  K: {kk}")
            print(f"  V: {vk} (Using K key)")
            print(f"  inv_freq: {invk}")

        # Biases
        qbk = find_first_key([qk.replace("weight","bias")] if qk else [], layer_keys)
        kbk = find_first_key([kk.replace("weight","bias")] if kk else [], layer_keys)
        vbk = find_first_key([vk.replace("weight","bias")] if vk else [], layer_keys) # This now correctly checks for kk's bias for V
        slim[f"l{idx}.q_proj.b"] = sd.get(qbk)
        slim[f"l{idx}.k_proj.b"] = sd.get(kbk)
        slim[f"l{idx}.v_proj.b"] = sd.get(vbk)

        # inv_freq
        slim[f"l{idx}.inv_freq"] = sd[invk]
        slim[f"l{idx}.inv_freq.key_name"] = invk
    # Use head_dim from inv_freq (first layer)
    slim["_head_dim"] = slim["l0.inv_freq"].shape[0] * 2
    slim["_n_layers"] = len(language_layer_prefixes)

    # --- Debug shape check and projection search ---
    if debug:
        print("\nDEBUG: Checking loaded tensor shapes...")
        # Retrieve stored original key names
        embed_key_name = slim.get('wte.key_name')
        q0_key_name = slim.get("l0.q_proj.key_name")
        k0_key_name = slim.get("l0.k_proj.key_name")
        v0_key_name = slim.get("l0.v_proj.key_name")

        if embed_key_name and embed_key_name in sd:
            print(f"  Embedding ({embed_key_name}): {sd[embed_key_name].shape}")
            embed_dim = sd[embed_key_name].shape[1]
        else:
            print("  Embedding key name not found in slim dict or sd.")
            raise ValueError("Failed to retrieve embedding dimension for shape checks.")

        target_hidden_dim = None
        if q0_key_name and q0_key_name in sd:
            print(f"  Layer 0 Q ({q0_key_name}): {sd[q0_key_name].shape}")
            q_in_dim = sd[q0_key_name].shape[1]
            print(f"    -> Q expects input dim: {q_in_dim}")
            if embed_dim != q_in_dim:
                print(f"    !!! MISMATCH with embedding dim: {embed_dim} !!!")
                target_hidden_dim = q_in_dim
            else:
                print(f"    Matches embedding dim: {embed_dim}")
                target_hidden_dim = embed_dim
        else:
            print("  Layer 0 Q key not found for shape check.")
            raise ValueError("Failed to retrieve Layer 0 Q projection for shape checks.")

        if k0_key_name and k0_key_name in sd:
            print(f"  Layer 0 K ({k0_key_name}): {sd[k0_key_name].shape}")
        if v0_key_name and v0_key_name in sd:
            print(f"  Layer 0 V ({v0_key_name}): {sd[v0_key_name].shape}")

        # --- Add Projection Search (only if mismatch detected) ---
        if embed_dim != target_hidden_dim:
            print(f"\nSearching for Projection Weight ({target_hidden_dim}, {embed_dim})...")
            found_proj_key = None
            found_proj_bias = None
            for k, v in sd.items():
                if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape == (target_hidden_dim, embed_dim):
                    print(f"  >>> Found Potential Projection Weight: {k} (Shape: {v.shape})")
                    found_proj_key = k
                    bias_key = k.replace(".weight", ".bias")
                    if bias_key in sd:
                        print(f"      Found Potential Bias: {bias_key} (Shape: {sd[bias_key].shape})")
                        found_proj_bias = bias_key
                    break
            if found_proj_key:
                slim['embed_proj.w_key'] = found_proj_key
                slim['embed_proj.b_key'] = found_proj_bias
                print("  Stored projection keys in slim dict.")
            else:
                print("  <<< Projection Weight NOT FOUND.")
                raise ValueError(f"Required projection layer (shape {(target_hidden_dim, embed_dim)}) not found in weights.")
        else:
            print("\nNo projection layer needed (dimensions match).")
    # --- End Debug shape check ---
    return slim, len(language_layer_prefixes), config

# ── KVOnlyModel corrected with proj naming and bias handling ─────────────────
class KVOnlyModel(nn.Module):
    def __init__(self, slim, n_layers, config, device, debug=False):
        super().__init__()
        self.vocab_size = slim["_vocab_size"]
        self.hidden_size = slim["_hidden_size"]
        self.head_dim = slim["_head_dim"]
        self.device = device
        self.n_layers = n_layers
        self.debug = debug
        self.embed = nn.Embedding(self.vocab_size, self.hidden_size, device=device)
        self.embed.weight.data.copy_(slim["wte.weight"].to(device))
        # If a projection is needed (embed_proj.w_key in slim), add it
        self.embed_proj = None
        if 'embed_proj.w_key' in slim:
            w = slim['embed_proj.w_key']
            b = slim['embed_proj.b_key'] if slim.get('embed_proj.b_key') else None
            out_dim, in_dim = w.shape
            self.embed_proj = nn.Linear(in_dim, out_dim, bias=(b is not None), device=device)
            self.embed_proj.weight.data.copy_(w.to(device))
            if b is not None:
                self.embed_proj.bias.data.copy_(b.to(device))
        self.layers = nn.ModuleList()
        self.n_heads = None
        self.n_kv_heads = None
        # Build layers
        for i in range(n_layers):
            blk = nn.Module()
            # Q/K/V
            for proj in ("q_proj","k_proj","v_proj"):
                w = slim[f"l{i}.{proj}.w"]
                b = slim[f"l{i}.{proj}.b"]
                out_dim, in_dim = w.shape
                lin = nn.Linear(in_dim, out_dim, bias=(b is not None), device=device)
                lin.weight.data.copy_(w.to(device))
                if b is not None:
                    lin.bias.data.copy_(b.to(device))
                setattr(blk, proj, lin)
            # RoPE
            inv = slim[f"l{i}.inv_freq"].to(device)
            blk.register_buffer("inv_freq", inv)
            self.layers.append(blk)
        # Infer n_heads and n_kv_heads from q_proj and k_proj shapes
        q_proj_w = slim["l0.q_proj.w"]
        k_proj_w = slim["l0.k_proj.w"]
        self.n_heads = q_proj_w.shape[0] // self.head_dim
        self.n_kv_heads = k_proj_w.shape[0] // self.head_dim
        if self.n_kv_heads < self.n_heads:
            self.gqa = True
        else:
            self.gqa = False
        self.eval()

    def apply_rotary(self, q, k, inv_freq, pos_id):
        # q, k: [B, 1, n_heads/n_kv_heads, head_dim]
        # inv_freq: [head_dim/2]
        # pos_id: [B, 1]
        angles = pos_id.unsqueeze(-1).to(inv_freq.device) * inv_freq  # [B, 1, head_dim/2]
        # Unsqueeze for broadcasting to heads
        cos = torch.cos(angles).unsqueeze(-2)  # [B, 1, 1, head_dim/2]
        sin = torch.sin(angles).unsqueeze(-2)  # [B, 1, 1, head_dim/2]
        def rope(x):
            # x: [B, 1, n_heads, head_dim]
            x1 = x[..., ::2]  # [B, 1, n_heads, head_dim/2]
            x2 = x[..., 1::2] # [B, 1, n_heads, head_dim/2]
            # Broadcast cos/sin over n_heads
            term1 = x1 * cos - x2 * sin
            term2 = x1 * sin + x2 * cos
            x_rope = torch.stack([term1, term2], dim=-1) # [B, 1, n_heads, head_dim/2, 2]
            return x_rope.flatten(start_dim=-2)  # [B, 1, n_heads, head_dim]
        return rope(q), rope(k)


    @torch.no_grad()
    def forward_kv_step(self, token_id, pos_id, past=None):
        token_id = token_id.to(self.device)
        pos_id = pos_id.to(self.device)
        hidden = self.embed(token_id)
        new_past = []
        for idx, blk in enumerate(self.layers):
            q = blk.q_proj(hidden)
            k = blk.k_proj(hidden)
            v = blk.v_proj(hidden)
            # Reshape for heads
            B = q.size(0)
            q = q.view(B, 1, self.n_heads, self.head_dim)
            k = k.view(B, 1, self.n_kv_heads, self.head_dim)
            v = v.view(B, 1, self.n_kv_heads, self.head_dim)
            # RoPE
            q, k = self.apply_rotary(q, k, blk.inv_freq, pos_id)
            # Permute to [B, n_heads/n_kv_heads, 1, head_dim]
            q = q.permute(0,2,1,3)
            k = k.permute(0,2,1,3)
            v = v.permute(0,2,1,3)
            if past is None:
                new_past.append((k, v))
            else:
                pk, pv = past[idx]
                new_past.append((torch.cat([pk, k], 2), torch.cat([pv, v], 2)))
        return new_past

# ── Benchmark: full model past_key_values timing ───────────────────────────────
def benchmark_full(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = inputs.input_ids.to(device)
    mask = torch.ones_like(ids, device=device)
    past, cpu_times, gpu_times = None, [], []
    # warm-up
    with torch.no_grad():
        _ = model(ids[:, :1], attention_mask=mask[:, :1], past_key_values=None, use_cache=True)
    for i in range(ids.size(1)):
        tok = ids[:, i:i+1]
        m   = mask[:, i:i+1]
        t0 = time.perf_counter()
        if device.type == "cuda":
            e0,e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
        with torch.no_grad():
            out = model(input_ids=tok, attention_mask=m, past_key_values=past, use_cache=True)
        past = out.past_key_values
        if device.type == "cuda":
            e1.record(); torch.cuda.synchronize(); gpu_times.append(e0.elapsed_time(e1)*1e-3)
        cpu_times.append(time.perf_counter() - t0)
    return cpu_times, gpu_times

# ── Benchmark: slim KV-only timing ─────────────────────────────────────────────
def benchmark_slim(text, kv_model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    ids = inputs.input_ids.to(device)
    pos_ids = torch.arange(ids.size(1), device=device).unsqueeze(0)
    past, cpu_times, gpu_times = None, [], []
    for i in range(ids.size(1)):
        tok = ids[:, i:i+1]
        pos = pos_ids[:, i:i+1]
        t0 = time.perf_counter()
        if device.type == "cuda":
            e0,e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
        past = kv_model.forward_kv_step(tok, pos, past)
        if device.type == "cuda":
            e1.record(); torch.cuda.synchronize(); gpu_times.append(e0.elapsed_time(e1)*1e-3)
        cpu_times.append(time.perf_counter() - t0)
    return cpu_times, gpu_times

# ── Main CLI entrypoint ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text",       type=str, required=True)
    parser.add_argument("--device",     type=str, default="cuda")
    parser.add_argument("--mode",       choices=["full","slim"], default="slim")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--debug",      action="store_true", help="Enable debug printing for slim loader")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if args.mode == "slim":
        print("→ Slim KV-only benchmark")
        slim, n_layers, cfg = load_slim_weights(args.model_name, debug=args.debug)
        kv_model = KVOnlyModel(slim, n_layers, cfg, device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=TRUST_REMOTE_CODE)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        cpu_times, gpu_times = benchmark_slim(args.text, kv_model, tokenizer, device)
    else:
        print("→ Full model benchmark")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=TRUST_REMOTE_CODE)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        ).eval().to(device)
        cpu_times, gpu_times = benchmark_full(args.text, model, tokenizer, device)

    n = len(cpu_times)
    total_cpu = sum(cpu_times)
    print(f"\nTokens: {n}")
    print(f"CPU total: {total_cpu:.3f}s  (avg {total_cpu/n:.4f}s/token)")
    if device.type == "cuda":
        total_gpu = sum(gpu_times)
        print(f"GPU total: {total_gpu:.3f}s  (avg {total_gpu/n:.4f}s/token)")

if __name__ == "__main__":
    main()
