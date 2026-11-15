# -*- coding: utf-8 -*-
"""
chat.py —— 用训练好的权重做问答（推理）

用法：
    # 单次提问
    py -3.13 chat.py --ckpt out_tiny_sft/ckpt_final.pt --question "你好，你是谁？"

    # 进入连续对话（支持多轮、记住上下文）
    py -3.13 chat.py --ckpt out_tiny_sft/ckpt_final.pt
    输入 退出 结束，输入 清空 忘掉之前的话
"""
import argparse
import time

import torch
import torch.nn.functional as F

from data import BOT_TAG, END_TAG, USER_TAG, build_prompt
from model import GPT, GPTConfig
from tokenizer import CharTokenizer


def load_bundle(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = GPTConfig.from_dict(ck["config"])
    tok = CharTokenizer(ck.get("tokenizer_chars") or CharTokenizer.load("tokenizer.json").chars)
    model = GPT(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, tok, cfg


def _safe_print_len(text):
    """结尾如果正好和 <|结束|> 的开头重合，就先别打印，免得把标记露出来"""
    best = 0
    for k in range(1, min(len(END_TAG), len(text)) + 1):
        if text[-k:] == END_TAG[:k]:
            best = k
    return len(text) - best


@torch.no_grad()
def stream_answer(model, tok, prompt, max_new_tokens=128, temperature=0.8,
                  top_k=40, top_p=0.9, rep_pen=1.15, show=True):
    """逐字生成并打印，返回完整回答文本"""
    device = next(model.parameters()).device
    # 注意：这里不能加 <bos>，必须和 sft.py 里的编码方式完全一致
    ids = tok.encode(prompt)
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    past = None
    text = ""
    printed = 0
    banned = [tok.pad_id, tok.bos_id, tok.unk_id]
    max_len = model.cfg.max_seq_len
    t0 = time.time()

    for _ in range(max_new_tokens):
        if (0 if past is None else past[0][0].size(2)) + 1 > max_len:
            break
        cur = idx if past is None else idx[:, -1:]
        logits, _, past = model(cur, past_kv=past, use_cache=True)
        logits = logits[:, -1, :].float()

        if rep_pen != 1.0:
            uniq = torch.unique(idx[0])
            logits[0, uniq] = torch.where(logits[0, uniq] > 0,
                                          logits[0, uniq] / rep_pen, logits[0, uniq] * rep_pen)
        if banned:
            logits[:, banned] = -float("inf")

        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k and top_k > 0:
                k = min(top_k, logits.size(-1))
                thresh = torch.topk(logits, k, dim=-1).values[:, -1:]
                logits = torch.where(logits < thresh, torch.full_like(logits, -float("inf")), logits)
            probs = F.softmax(logits, dim=-1)
            if top_p and top_p < 1.0:
                sp, si = torch.sort(probs, dim=-1, descending=True)
                cum = torch.cumsum(sp, dim=-1)
                sp[cum - sp > top_p] = 0.0
                probs = torch.zeros_like(probs).scatter(-1, si, sp)
            probs = probs / probs.sum(-1, keepdim=True)
            nxt = torch.multinomial(probs, num_samples=1)

        idx = torch.cat([idx, nxt], dim=1)
        new_char = tok.decode([nxt.item()])
        if new_char:
            text += new_char
            if show:
                safe = _safe_print_len(text)
                while printed < safe and printed < len(text):
                    print(text[printed], end="", flush=True)
                    printed += 1
        if END_TAG in text or nxt.item() == tok.eos_id:
            break

    visible = text.split(END_TAG)[0].split(USER_TAG)[0]
    if show:
        while printed < len(visible):
            print(visible[printed], end="", flush=True)
            printed += 1
    answer = visible.strip()
    if show:
        print(f"\n[用时 {time.time() - t0:.1f}s，生成 {len(answer)} 字]")
    return answer


def build_history_prompt(history, tok, max_ctx, reserve):
    """历史太长就丢掉最早的对话，保证 prompt 不会超出上下文长度"""
    budget = max_ctx - reserve
    while True:
        prompt = build_prompt(history)
        if len(tok.encode(prompt)) <= budget or len(history) <= 1:
            return prompt
        history = history[2:]   # 一次丢一轮（用户+助手）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out_tiny_sft/ckpt_final.pt")
    ap.add_argument("--question", nargs="*", default=None, help="一次性提问；不填就进入连续对话")
    ap.add_argument("--max_new_tokens", type=int, default=120)
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = 每次都给同一个答案（贪心）")
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--rep_pen", type=float, default=1.15)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "auto"
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tok, cfg = load_bundle(args.ckpt, device)
    gen_kw = dict(max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                  top_k=args.top_k, top_p=args.top_p, rep_pen=args.rep_pen)

    print("=" * 66)
    print(f"权重：{args.ckpt}")
    print(f"结构：{cfg.n_layer} 层 / {cfg.d_model} 维 / 参数量 {model.num_params():,}")
    print(f"设备：{device}")
    print("=" * 66)

    # ---------- 一次性提问 ----------
    if args.question:
        q = " ".join(args.question)
        print(f"\n用户：{q}\n助手：", end="")
        stream_answer(model, tok, build_prompt([("user", q)]), **gen_kw)
        return

    # ---------- 连续对话 ----------
    print("进入对话（输入 退出 结束，输入 清空 忘记之前的内容）\n")
    history = []
    while True:
        try:
            q = input("用户：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in ("退出", "exit", "quit"):
            break
        if q == "清空":
            history = []
            print("（已清空上下文）\n")
            continue

        history.append(("user", q))
        prompt = build_history_prompt(history, tok, cfg.max_seq_len, args.max_new_tokens)
        print("助手：", end="")
        answer = stream_answer(model, tok, prompt, **gen_kw)
        history.append(("assistant", answer))
        print()

    print("对话结束。")


if __name__ == "__main__":
    main()
