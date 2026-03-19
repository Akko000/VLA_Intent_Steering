"""
OpenVLA Rollout Alignment Analysis
====================================
Simulate autoregressive action generation and track how the hidden state's
alignment with the initial intent anchor changes over decoding steps.

Core hypothesis: as the model generates more action tokens, the hidden state
drifts away from the instruction-grounded intent direction.

Usage:
    python scripts/analyze_rollout.py \
        --model_path /scratch/work/zouz1/VLA_intent/models/openvla-7b \
        --output_dir outputs/rollout
"""

import os
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from PIL import Image
from pathlib import Path
from collections import defaultdict
import time


def create_test_images():
    """Create several distinct test images to simulate different scenes."""
    images = []
    descriptions = []

    img = Image.new('RGB', (224, 224), (180, 180, 160))
    px = img.load()
    for x in range(70, 130):
        for y in range(90, 140):
            px[x, y] = (200, 40, 40)
    images.append(img)
    descriptions.append("red block on table")

    img = Image.new('RGB', (224, 224), (180, 180, 160))
    px = img.load()
    for x in range(140, 190):
        for y in range(60, 130):
            px[x, y] = (40, 40, 200)
    images.append(img)
    descriptions.append("blue cup near edge")

    img = Image.new('RGB', (224, 224), (180, 180, 160))
    px = img.load()
    for x in range(30, 80):
        for y in range(80, 130):
            px[x, y] = (200, 40, 40)
    for x in range(120, 170):
        for y in range(70, 120):
            px[x, y] = (40, 180, 40)
    for x in range(80, 120):
        for y in range(140, 180):
            px[x, y] = (40, 40, 200)
    images.append(img)
    descriptions.append("multiple colored blocks")

    return images, descriptions


def load_model(model_path, device="cuda:0"):
    """Load OpenVLA."""
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    print("Loading model...")
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.eval()
    print(f"Model loaded on {device}")
    return model, processor


def find_llm_layers(model):
    """Find LLM decoder layers."""
    candidates = [
        lambda m: m.language_model.model.layers,
        lambda m: m.model.layers,
        lambda m: m.language_model.layers,
    ]
    for getter in candidates:
        try:
            layers = getter(model)
            return layers
        except AttributeError:
            continue
    raise RuntimeError("Cannot find decoder layers")


def extract_intent_anchor(model, processor, image, prompt, target_layers, device="cuda:0"):
    """
    Extract intent anchor from the prefill stage.
    For each target layer, capture the SA residual update during prefill,
    pool over all tokens, and L2-normalize to get a direction vector.
    """
    layers = find_llm_layers(model)
    pre_hiddens = {}
    post_hiddens = {}

    def make_pre_hook(idx):
        def fn(module, args):
            if isinstance(args, tuple) and len(args) > 0 and isinstance(args[0], torch.Tensor):
                pre_hiddens[idx] = args[0].detach()
        return fn

    def make_post_hook(idx):
        def fn(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            if isinstance(h, torch.Tensor):
                post_hiddens[idx] = h.detach()
        return fn

    handles = []
    for l in target_layers:
        handles.append(layers[l].register_forward_pre_hook(make_pre_hook(l)))
        handles.append(layers[l].register_forward_hook(make_post_hook(l)))

    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
    with torch.no_grad():
        model(**inputs)

    for h in handles:
        h.remove()

    anchors = {}
    for l in target_layers:
        if l in pre_hiddens and l in post_hiddens:
            delta = (post_hiddens[l] - pre_hiddens[l]).squeeze(0).float()
            pooled = delta.mean(dim=0)
            anchor = pooled / pooled.norm().clamp(min=1e-8)
            anchors[l] = anchor.cpu()

    return anchors


def run_autoregressive_generation_with_tracking(
    model, processor, image, prompt, target_layers,
    max_new_tokens=50, device="cuda:0"
):
    """
    Run autoregressive generation and track alignment with intent anchor
    at each decoding step.
    """
    layers = find_llm_layers(model)

    print("    Extracting intent anchors...", end=" ", flush=True)
    anchors = extract_intent_anchor(model, processor, image, prompt, target_layers, device)
    print("done", flush=True)

    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
    input_ids = inputs["input_ids"]

    step_alignments = {l: [] for l in target_layers}
    step_hidden_norms = {l: [] for l in target_layers}
    step_delta_norms = {l: [] for l in target_layers}
    generated_ids = []

    current_ids = input_ids.clone()
    generate_kwargs = {k: v for k, v in inputs.items() if k != "input_ids"}

    step_start = time.time()

    for step in range(max_new_tokens):
        step_pre = {}
        step_post = {}

        def make_pre(idx):
            def fn(module, args):
                if isinstance(args, tuple) and len(args) > 0 and isinstance(args[0], torch.Tensor):
                    step_pre[idx] = args[0].detach()
            return fn

        def make_post(idx):
            def fn(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                if isinstance(h, torch.Tensor):
                    step_post[idx] = h.detach()
            return fn

        handles = []
        for l in target_layers:
            handles.append(layers[l].register_forward_pre_hook(make_pre(l)))
            handles.append(layers[l].register_forward_hook(make_post(l)))

        with torch.no_grad():
            outputs = model(input_ids=current_ids, **generate_kwargs)

        for h in handles:
            h.remove()

        next_token_logits = outputs.logits[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token.item())

        for l in target_layers:
            if l in step_post and l in step_pre:
                h_last = step_post[l][:, -1, :].squeeze(0).float().cpu()
                h_last_normed = h_last / h_last.norm().clamp(min=1e-8)
                delta_last = (step_post[l][:, -1, :] - step_pre[l][:, -1, :]).squeeze(0).float().cpu()
                cos_sim = torch.dot(h_last_normed, anchors[l]).item()
                step_alignments[l].append(cos_sim)
                step_hidden_norms[l].append(h_last.norm().item())
                step_delta_norms[l].append(delta_last.norm().item())

        current_ids = torch.cat([current_ids, next_token], dim=-1)

        # Progress bar
        progress = int((step + 1) / max_new_tokens * 30)
        elapsed = time.time() - step_start
        speed = (step + 1) / elapsed if elapsed > 0 else 0
        eta = (max_new_tokens - step - 1) / speed if speed > 0 else 0
        print(f"\r    Tokens: [{'='*progress}{' '*(30-progress)}] "
              f"{step+1}/{max_new_tokens} ({speed:.1f} tok/s, ETA {eta:.0f}s)", end="", flush=True)

        if next_token.item() == processor.tokenizer.eos_token_id:
            print(f" => EOS", flush=True)
            break
    else:
        print(flush=True)

    return step_alignments, step_hidden_norms, step_delta_norms, generated_ids


def run_experiments(model, processor, args):
    """Run rollout analysis across multiple instructions and images."""
    images, scene_descs = create_test_images()

    instructions = [
        "pick up the red block and place it on the blue block",
        "push the cup to the left side of the table",
        "open the drawer and put the apple inside",
        "stack the blocks on top of each other",
        "move the red object to the right",
    ]

    target_layers = [5, 10, 15, 20, 25, 30, 31]
    all_results = []
    total = len(images) * len(instructions)
    current = 0
    overall_start = time.time()

    for img_idx, (image, scene) in enumerate(zip(images, scene_descs)):
        for inst_idx, instruction in enumerate(instructions):
            current += 1
            prompt = f"In: What action should the robot take to {instruction}?\nOut:"
            exp_name = f"scene{img_idx}_inst{inst_idx}"

            elapsed_so_far = time.time() - overall_start
            eta_total = elapsed_so_far / current * (total - current) if current > 0 else 0

            print(f"\n{'='*60}")
            print(f"[{current}/{total}] {exp_name}  (Total ETA: {eta_total:.0f}s)")
            print(f"  Scene: {scene}")
            print(f"  Instruction: {instruction}")

            alignments, h_norms, d_norms, gen_ids = \
                run_autoregressive_generation_with_tracking(
                    model, processor, image, prompt,
                    target_layers, max_new_tokens=args.max_tokens,
                    device=args.device
                )

            n_steps = len(gen_ids)

            decay_rates = {}
            for l in target_layers:
                if len(alignments[l]) >= 2:
                    a = np.array(alignments[l])
                    x = np.arange(len(a))
                    slope = np.polyfit(x, a, 1)[0]
                    decay_rates[l] = slope

            print(f"    Results ({n_steps} tokens):")
            for l in target_layers:
                if len(alignments[l]) >= 2:
                    a = alignments[l]
                    dr = decay_rates.get(l, 0)
                    arrow = "↓" if dr < -0.001 else "↑" if dr > 0.001 else "→"
                    print(f"      L{l:2d}: {a[0]:.4f} -> {a[-1]:.4f} {arrow} (decay={dr:+.6f})")

            result = {
                'scene': scene,
                'instruction': instruction,
                'exp_name': exp_name,
                'n_steps': n_steps,
                'target_layers': target_layers,
                'alignments': {str(l): alignments[l] for l in target_layers},
                'hidden_norms': {str(l): h_norms[l] for l in target_layers},
                'delta_norms': {str(l): d_norms[l] for l in target_layers},
                'decay_rates': {str(l): decay_rates.get(l, 0) for l in target_layers},
            }
            all_results.append(result)

    total_time = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"All {total} experiments completed in {total_time:.0f}s ({total_time/60:.1f}min)")
    return all_results


def plot_results(all_results, output_dir):
    """Generate comprehensive plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_layers = all_results[0]['target_layers']
    n_layers = len(target_layers)

    # --- Plot 1: Alignment curves per layer ---
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()

    for idx, l in enumerate(target_layers):
        ax = axes[idx]
        for r in all_results:
            a = r['alignments'][str(l)]
            if len(a) > 0:
                ax.plot(a, alpha=0.5, linewidth=1.5)
        ax.set_xlabel('Decoding Step', fontsize=11)
        ax.set_ylabel('Cosine Sim with Intent Anchor', fontsize=11)
        ax.set_title(f'Layer {l}', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)

    if n_layers < len(axes):
        for i in range(n_layers, len(axes)):
            axes[i].set_visible(False)

    plt.suptitle('Alignment with Intent Anchor During Autoregressive Decoding\n'
                 '(Each line = one instruction x scene combination)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'alignment_curves_per_layer.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved alignment_curves_per_layer.png")

    # --- Plot 2: Summary ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 2a: Mean alignment curve per layer
    ax = axes[0]
    for l in target_layers:
        curves = [r['alignments'][str(l)] for r in all_results if len(r['alignments'][str(l)]) > 0]
        if curves:
            min_len = min(len(c) for c in curves)
            truncated = [c[:min_len] for c in curves]
            mean_curve = np.mean(truncated, axis=0)
            std_curve = np.std(truncated, axis=0)
            x = np.arange(min_len)
            ax.plot(x, mean_curve, linewidth=2.5, label=f'L{l}')
            ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)
    ax.set_xlabel('Decoding Step', fontsize=12)
    ax.set_ylabel('Cosine Sim with Intent Anchor', fontsize=12)
    ax.set_title('Mean Alignment per Layer\n(averaged over all experiments)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2b: Decay rate per layer
    ax = axes[1]
    mean_decay = []
    std_decay = []
    for l in target_layers:
        rates = [r['decay_rates'][str(l)] for r in all_results]
        mean_decay.append(np.mean(rates))
        std_decay.append(np.std(rates))
    bars = ax.bar(range(len(target_layers)), mean_decay,
                  yerr=std_decay, capsize=4, color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(target_layers)))
    ax.set_xticklabels([f'L{l}' for l in target_layers])
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Alignment Decay Rate (slope)', fontsize=12)
    ax.set_title('Alignment Decay Rate per Layer\n(more negative = faster drift)', fontsize=14)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    for bar, val in zip(bars, mean_decay):
        bar.set_color('darkred' if val < -0.001 else 'steelblue')

    # 2c: Initial vs Final alignment
    ax = axes[2]
    for l in target_layers:
        initials = []
        finals = []
        for r in all_results:
            a = r['alignments'][str(l)]
            if len(a) >= 2:
                initials.append(a[0])
                finals.append(a[-1])
        if initials:
            ax.scatter(np.mean(initials), np.mean(finals), s=120,
                      label=f'L{l}', zorder=5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='No drift')
    ax.set_xlabel('Initial Alignment (step 0)', fontsize=12)
    ax.set_ylabel('Final Alignment (last step)', fontsize=12)
    ax.set_title('Initial vs Final Alignment\n(below diagonal = drift occurred)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle('Rollout Alignment Analysis Summary', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'rollout_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved rollout_summary.png")

    # --- Plot 3: Hidden state norm evolution ---
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for l in target_layers:
        curves = [r['hidden_norms'][str(l)] for r in all_results if len(r['hidden_norms'][str(l)]) > 0]
        if curves:
            min_len = min(len(c) for c in curves)
            mean_curve = np.mean([c[:min_len] for c in curves], axis=0)
            ax.plot(mean_curve, linewidth=2, label=f'L{l}')
    ax.set_xlabel('Decoding Step', fontsize=12)
    ax.set_ylabel('Hidden State Norm', fontsize=12)
    ax.set_title('Hidden State Norm Evolution During Decoding', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'hidden_norm_evolution.png', dpi=150)
    plt.close()
    print(f"  Saved hidden_norm_evolution.png")


def main(args):
    print("=" * 60)
    print("OpenVLA Rollout Alignment Analysis")
    print("=" * 60)

    model, processor = load_model(args.model_path, device=args.device)

    results = run_experiments(model, processor, args)

    # Save raw results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'rollout_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {output_dir / 'rollout_results.json'}")

    # Plot
    print("\nGenerating plots...")
    plot_results(results, args.output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    target_layers = results[0]['target_layers']
    print(f"{'Layer':>7} {'Initial':>10} {'Final':>10} {'Drop':>10} {'Decay Rate':>12}")
    print("-" * 55)
    for l in target_layers:
        rates = [r['decay_rates'][str(l)] for r in results]
        mean_rate = np.mean(rates)
        initials = [r['alignments'][str(l)][0] for r in results if len(r['alignments'][str(l)]) > 0]
        finals = [r['alignments'][str(l)][-1] for r in results if len(r['alignments'][str(l)]) > 0]
        drop = np.mean(initials) - np.mean(finals)
        print(f"  L{l:2d}   {np.mean(initials):10.4f} {np.mean(finals):10.4f} {drop:10.4f} {mean_rate:+12.6f}")

    all_drops = {}
    for l in target_layers:
        initials = [r['alignments'][str(l)][0] for r in results if len(r['alignments'][str(l)]) > 0]
        finals = [r['alignments'][str(l)][-1] for r in results if len(r['alignments'][str(l)]) > 0]
        all_drops[l] = np.mean(initials) - np.mean(finals)

    worst_layer = max(all_drops, key=all_drops.get)
    print(f"\nLayer with strongest alignment drop: L{worst_layer} "
          f"(drop = {all_drops[worst_layer]:.4f})")
    print("=> Primary candidate for intent steering intervention.")
    print("\nDone!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenVLA Rollout Alignment Analysis')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='outputs/rollout')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--max_tokens', type=int, default=30,
                        help='Max action tokens to generate per rollout')
    args = parser.parse_args()
    main(args)