"""
OpenVLA Internal Dynamics Analysis
===================================
Analyze the residual stream of OpenVLA during prefill to understand
how instruction and visual information is encoded across layers.

This is the VLA equivalent of RUDDER's Figure 7 (LLaVA internal dynamics).

What this script does:
1. Load OpenVLA and run a forward pass with instruction + image
2. Hook every layer's self-attention output to capture residual updates
3. Compute per-layer metrics:
   - Absolute strength of SA residual updates (||delta_l||)
   - Relative strength (||delta_l|| / ||h_l_pre||)
   - Directional coherence across tokens within each layer
   - Cosine similarity between instruction-token updates and image-token updates
4. Generate diagnostic plots

Usage:
    python scripts/analyze_internals.py \
        --model_path /scratch/work/zouz1/VLA_intent/models/openvla-7b \
        --output_dir outputs/internals
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless rendering for HPC
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pathlib import Path


def create_dummy_image(size=224):
    """Create a simple test image (colored rectangles simulating a tabletop scene)."""
    img = Image.new('RGB', (size, size), color=(200, 200, 200))
    pixels = img.load()
    # Red block (simulating an object)
    for x in range(60, 120):
        for y in range(80, 140):
            pixels[x, y] = (200, 50, 50)
    # Blue block
    for x in range(140, 190):
        for y in range(90, 130):
            pixels[x, y] = (50, 50, 200)
    return img


def load_model_and_processor(model_path, device="cuda:0"):
    """Load OpenVLA model and processor."""
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print(f"Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True
    )

    print(f"Loading model from {model_path}...")
    # Skip flash_attention_2 to avoid dependency issues; eager attention works fine for analysis
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    model.eval()
    print(f"Model loaded. Device: {device}")
    return model, processor


def find_llm_layers(model):
    """
    Find the LLM decoder layers in the model.
    OpenVLA uses Llama-2, so the layers are at model.language_model.model.layers
    or similar path. We search for it dynamically.
    """
    # Try common paths for the LLM backbone
    candidates = [
        lambda m: m.language_model.model.layers,   # OpenVLA / Prismatic
        lambda m: m.model.layers,                    # direct Llama
        lambda m: m.language_model.layers,           # alternative
    ]
    for getter in candidates:
        try:
            layers = getter(model)
            print(f"Found {len(layers)} decoder layers")
            return layers
        except AttributeError:
            continue

    # If none worked, print model structure to help debug
    print("Could not find decoder layers automatically. Model structure:")
    for name, _ in model.named_modules():
        if 'layer' in name.lower() or 'block' in name.lower():
            print(f"  {name}")
    raise RuntimeError("Cannot locate LLM decoder layers. Check model structure above.")


def analyze_residual_updates(model, processor, image, prompt, device="cuda:0"):
    """
    Run a forward pass and capture self-attention residual updates at every layer.

    Returns:
        layer_data: list of dicts, one per layer, containing:
            - 'sa_output': tensor [seq_len, hidden_dim] - the SA sublayer output
            - 'pre_sa_hidden': tensor [seq_len, hidden_dim] - hidden state before SA
    """
    layers = find_llm_layers(model)
    num_layers = len(layers)

    # Storage for hook data
    sa_outputs = {}
    pre_sa_hiddens = {}

    def make_pre_hook(layer_idx):
        """Capture hidden state BEFORE self-attention."""
        def hook_fn(module, args):
            # args[0] is the hidden states input to the layer
            if isinstance(args, tuple) and len(args) > 0:
                h = args[0]
                if isinstance(h, torch.Tensor):
                    pre_sa_hiddens[layer_idx] = h.detach().cpu().float()
        return hook_fn

    def make_post_hook(layer_idx):
        """Capture the full layer output (which includes SA + FFN residual)."""
        def hook_fn(module, args, output):
            # For Llama layers, output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            if isinstance(h, torch.Tensor):
                sa_outputs[layer_idx] = h.detach().cpu().float()
        return hook_fn

    # Register hooks on each decoder layer
    handles = []
    for i, layer in enumerate(layers):
        h1 = layer.register_forward_pre_hook(make_pre_hook(i))
        h2 = layer.register_forward_hook(make_post_hook(i))
        handles.append(h1)
        handles.append(h2)

    # Prepare inputs
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
    input_ids = inputs.get("input_ids", inputs.get("input_token_ids"))
    seq_len = input_ids.shape[1] if input_ids is not None else None
    print(f"Input sequence length: {seq_len}")

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)

    # Remove hooks
    for h in handles:
        h.remove()

    # Compute residual updates (delta_l = output_l - input_l)
    layer_data = []
    for i in range(num_layers):
        if i in sa_outputs and i in pre_sa_hiddens:
            pre = pre_sa_hiddens[i].squeeze(0)   # [seq_len, hidden_dim]
            post = sa_outputs[i].squeeze(0)       # [seq_len, hidden_dim]
            delta = post - pre                     # residual update
            layer_data.append({
                'delta': delta,          # [seq_len, hidden_dim]
                'pre_hidden': pre,       # [seq_len, hidden_dim]
                'post_hidden': post,     # [seq_len, hidden_dim]
            })
        else:
            print(f"Warning: missing data for layer {i}")
            layer_data.append(None)

    # Get token info for analysis
    token_info = get_token_info(processor, inputs, prompt)

    return layer_data, token_info


def get_token_info(processor, inputs, prompt):
    """
    Identify which tokens are image tokens vs instruction tokens.
    Returns a dict with token type masks.
    """
    input_ids = inputs.get("input_ids", inputs.get("input_token_ids"))
    if input_ids is None:
        return {"total_tokens": 0}

    input_ids = input_ids.cpu().squeeze(0)
    total_tokens = len(input_ids)

    # Try to decode tokens to identify instruction vs image tokens
    # Image tokens in OpenVLA are typically special tokens (e.g., token_id = 32000)
    # We use a heuristic: image tokens have a specific ID range
    try:
        tokenizer = processor.tokenizer
        tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())

        # Identify image patch tokens (usually <image_patch> or similar)
        image_mask = torch.zeros(total_tokens, dtype=torch.bool)
        text_mask = torch.zeros(total_tokens, dtype=torch.bool)
        for idx, (tid, tok) in enumerate(zip(input_ids, tokens)):
            if 'image' in str(tok).lower() or 'patch' in str(tok).lower() or tid >= 32000:
                image_mask[idx] = True
            else:
                text_mask[idx] = True

        n_image = image_mask.sum().item()
        n_text = text_mask.sum().item()
        print(f"Token breakdown: {n_image} image tokens, {n_text} text tokens, {total_tokens} total")
    except Exception as e:
        print(f"Could not parse tokens: {e}")
        # Fallback: assume first ~256 tokens are image, rest are text
        image_mask = torch.zeros(total_tokens, dtype=torch.bool)
        text_mask = torch.zeros(total_tokens, dtype=torch.bool)
        n_image_approx = min(256, total_tokens // 2)
        image_mask[:n_image_approx] = True
        text_mask[n_image_approx:] = True
        n_image = n_image_approx
        n_text = total_tokens - n_image_approx
        print(f"Token breakdown (estimated): {n_image} image, {n_text} text, {total_tokens} total")

    return {
        'total_tokens': total_tokens,
        'image_mask': image_mask,
        'text_mask': text_mask,
        'n_image': n_image,
        'n_text': n_text,
    }


def compute_metrics(layer_data, token_info):
    """
    Compute per-layer metrics from the captured residual updates.

    Metrics (matching RUDDER Figure 7):
    1. SA strength: mean ||delta_l|| across tokens
    2. Relative strength: mean ||delta_l|| / ||h_l_pre|| across tokens
    3. Directional coherence: mean pairwise cosine sim of delta vectors
    4. Image-text alignment: cosine sim between mean(delta_image) and mean(delta_text)
    """
    num_layers = len(layer_data)
    metrics = {
        'sa_strength': [],           # ||delta|| per layer
        'relative_strength': [],     # ||delta|| / ||h_pre|| per layer
        'coherence': [],             # directional coherence per layer
        'image_text_alignment': [],  # cos(mean_delta_image, mean_delta_text)
        'instruction_norm': [],      # ||mean delta for text tokens||
        'image_norm': [],            # ||mean delta for image tokens||
    }

    image_mask = token_info.get('image_mask')
    text_mask = token_info.get('text_mask')

    for i, data in enumerate(layer_data):
        if data is None:
            for k in metrics:
                metrics[k].append(0.0)
            continue

        delta = data['delta']      # [seq_len, hidden_dim]
        pre = data['pre_hidden']   # [seq_len, hidden_dim]

        # 1. Absolute SA strength
        token_norms = delta.norm(dim=-1)  # [seq_len]
        metrics['sa_strength'].append(token_norms.mean().item())

        # 2. Relative strength
        pre_norms = pre.norm(dim=-1).clamp(min=1e-8)
        relative = (token_norms / pre_norms)
        metrics['relative_strength'].append(relative.mean().item())

        # 3. Directional coherence (sample subset for efficiency)
        n_sample = min(50, delta.shape[0])
        indices = torch.randperm(delta.shape[0])[:n_sample]
        sampled = delta[indices]
        sampled_normed = sampled / sampled.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cos_matrix = sampled_normed @ sampled_normed.T
        # Mean of upper triangle (excluding diagonal)
        mask_upper = torch.triu(torch.ones_like(cos_matrix, dtype=torch.bool), diagonal=1)
        coherence = cos_matrix[mask_upper].mean().item() if mask_upper.sum() > 0 else 0.0
        metrics['coherence'].append(coherence)

        # 4. Image vs Text token alignment
        if image_mask is not None and text_mask is not None:
            delta_image = delta[image_mask].mean(dim=0) if image_mask.any() else torch.zeros(delta.shape[-1])
            delta_text = delta[text_mask].mean(dim=0) if text_mask.any() else torch.zeros(delta.shape[-1])

            metrics['image_norm'].append(delta_image.norm().item())
            metrics['instruction_norm'].append(delta_text.norm().item())

            # Cosine similarity between image and text directions
            cos_sim = torch.nn.functional.cosine_similarity(
                delta_image.unsqueeze(0), delta_text.unsqueeze(0)
            ).item()
            metrics['image_text_alignment'].append(cos_sim)
        else:
            metrics['image_norm'].append(0.0)
            metrics['instruction_norm'].append(0.0)
            metrics['image_text_alignment'].append(0.0)

    return metrics


def plot_metrics(metrics, output_dir):
    """Generate diagnostic plots (analogous to RUDDER Figure 7)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    num_layers = len(metrics['sa_strength'])
    x = list(range(num_layers))

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. SA Strength (RUDDER Fig 7a)
    ax = axes[0, 0]
    ax.bar(x, metrics['sa_strength'], color='steelblue', alpha=0.8)
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Mean ||δ||', fontsize=12)
    ax.set_title('Self-Attention Update Strength per Layer', fontsize=14)
    ax.grid(True, alpha=0.3)

    # 2. Relative Strength (RUDDER Fig 7c)
    ax = axes[0, 1]
    ax.bar(x, metrics['relative_strength'], color='coral', alpha=0.8)
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Mean ||δ|| / ||h_pre||', fontsize=12)
    ax.set_title('Relative SA Update Strength per Layer', fontsize=14)
    ax.grid(True, alpha=0.3)

    # 3. Directional Coherence (RUDDER Fig 7b)
    ax = axes[0, 2]
    ax.plot(x, metrics['coherence'], 'o-', color='darkgreen', linewidth=2)
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Mean Pairwise Cosine Sim', fontsize=12)
    ax.set_title('Directional Coherence per Layer', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.0)

    # 4. Image vs Text Update Norms
    ax = axes[1, 0]
    width = 0.35
    ax.bar([i - width/2 for i in x], metrics['image_norm'], width, label='Image tokens', color='royalblue', alpha=0.8)
    ax.bar([i + width/2 for i in x], metrics['instruction_norm'], width, label='Text tokens', color='orangered', alpha=0.8)
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('||mean δ||', fontsize=12)
    ax.set_title('Image vs Instruction Update Norms', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # 5. Image-Text Alignment
    ax = axes[1, 1]
    colors = ['green' if v > 0 else 'red' for v in metrics['image_text_alignment']]
    ax.bar(x, metrics['image_text_alignment'], color=colors, alpha=0.7)
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Image-Text Update Direction Alignment', fontsize=14)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    # 6. Summary: optimal layer selection
    ax = axes[1, 2]
    # Intervention score: high strength + low coherence = good for steering
    strength = np.array(metrics['relative_strength'])
    coherence = np.array(metrics['coherence'])
    # Normalize both to [0, 1]
    s_norm = (strength - strength.min()) / (strength.max() - strength.min() + 1e-8)
    c_norm = (coherence - coherence.min()) / (coherence.max() - coherence.min() + 1e-8)
    # Score: high strength, low coherence
    score = s_norm * (1 - c_norm)
    ax.bar(x, score, color='purple', alpha=0.7)
    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Intervention Score', fontsize=12)
    ax.set_title('Candidate Layers for Steering\n(high strength × low coherence)', fontsize=14)
    ax.grid(True, alpha=0.3)
    best_layer = int(np.argmax(score))
    ax.annotate(f'Best: L{best_layer}', xy=(best_layer, score[best_layer]),
                fontsize=12, fontweight='bold', color='darkred',
                xytext=(best_layer + 2, score[best_layer]),
                arrowprops=dict(arrowstyle='->', color='darkred'))

    plt.suptitle('OpenVLA Internal Dynamics Analysis\n(VLA equivalent of RUDDER Figure 7)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'openvla_internal_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_dir / 'openvla_internal_dynamics.png'}")

    # Save raw metrics
    serializable = {k: [float(v) for v in vals] for k, vals in metrics.items()}
    serializable['best_steering_layer'] = best_layer
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Metrics saved to {output_dir / 'metrics.json'}")


def main(args):
    print("=" * 60)
    print("OpenVLA Internal Dynamics Analysis")
    print("=" * 60)

    # Load model
    model, processor = load_model_and_processor(args.model_path, device=args.device)

    # Create test image
    image = create_dummy_image()
    print("Using synthetic test image (224x224)")

    # Format prompt (OpenVLA format)
    instruction = "pick up the red block and place it on the blue block"
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    print(f"Instruction: {instruction}")

    # Run analysis
    print("\nRunning forward pass and capturing residual updates...")
    layer_data, token_info = analyze_residual_updates(model, processor, image, prompt, device=args.device)

    # Compute metrics
    print("\nComputing per-layer metrics...")
    metrics = compute_metrics(layer_data, token_info)

    # Plot
    print("\nGenerating plots...")
    plot_metrics(metrics, args.output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    best = int(np.argmax(
        np.array(metrics['relative_strength']) *
        (1 - np.array(metrics['coherence']).clip(0, 1))
    ))
    print(f"Total layers: {len(metrics['sa_strength'])}")
    print(f"Best candidate layer for steering: {best}")
    print(f"  - Relative strength at L{best}: {metrics['relative_strength'][best]:.4f}")
    print(f"  - Coherence at L{best}: {metrics['coherence'][best]:.4f}")
    print(f"  - Image-text alignment at L{best}: {metrics['image_text_alignment'][best]:.4f}")

    # Also test with different instructions to check consistency
    if args.multi_instruction:
        print("\n\nRunning multi-instruction comparison...")
        instructions = [
            "pick up the red block and place it on the blue block",
            "push the cup to the left side of the table",
            "open the drawer and put the apple inside",
            "stack the plates on the counter",
        ]
        all_metrics = []
        for inst in instructions:
            p = f"In: What action should the robot take to {inst}?\nOut:"
            ld, ti = analyze_residual_updates(model, processor, image, p, device=args.device)
            m = compute_metrics(ld, ti)
            all_metrics.append(m)
            print(f"  Instruction: '{inst[:50]}...' -> best layer: {int(np.argmax(np.array(m['relative_strength']) * (1 - np.array(m['coherence']).clip(0,1))))}")

        # Plot consistency across instructions
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        for i, (inst, m) in enumerate(zip(instructions, all_metrics)):
            ax.plot(m['relative_strength'], label=inst[:40] + '...', alpha=0.7, linewidth=2)
        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Relative SA Strength', fontsize=12)
        ax.set_title('Consistency Across Instructions', fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        output_dir = Path(args.output_dir)
        plt.savefig(output_dir / 'multi_instruction_consistency.png', dpi=150)
        plt.close()
        print(f"Multi-instruction plot saved")

    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenVLA Internal Dynamics Analysis')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to OpenVLA model')
    parser.add_argument('--output_dir', type=str, default='outputs/internals',
                        help='Output directory for plots and metrics')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run on')
    parser.add_argument('--multi_instruction', action='store_true',
                        help='Also run analysis with multiple instructions for consistency check')
    args = parser.parse_args()
    main(args)
