# 嘉桦1.0（Jiahua-1.0）

一个**只用 Python + PyTorch、不依赖 transformers** 的中文问答大模型。
自己写模型结构、自己写训练、自己写推理，训练完只留一个 `.pt` 权重文件。

```
   拿 Python 文本训练  ->  得到一个模型文件  ->  拿 Python 做问答
```

- 主力规格：**0.75B**（24 层 / 1536 维 / 12 头 / 4 万词表，实测 755,148,288 参数）
- 已验证可跑：**嘉桦1.0-tiny**（357 万参数，笔记本 CPU 7~8 分钟训练完，问答可用）

## 快速开始

```bash
git clone https://github.com/plpy6636-ABC/Jiahua-1.0.git
cd Jiahua-1.0
py -3.13 -m pip install torch --index-url https://download.pytorch.org/whl/cpu

py -3.13 tokenizer.py build --data data/corpus.txt data/qa.jsonl --out tokenizer.json
py -3.13 train.py --config configs/tiny.json --data data/corpus.txt --out_dir out_tiny --max_steps 600 --batch_size 8 --seq_len 160
py -3.13 sft.py   --ckpt out_tiny/ckpt_final.pt --data data/qa.jsonl --out_dir out_tiny_sft --max_steps 400 --batch_size 8 --seq_len 96 --lr 2e-4
py -3.13 chat.py  --ckpt out_tiny_sft/ckpt_final.pt --question "你是谁"
```

Windows 用户也可以直接双击 `一键试跑.bat`。

---

## 一、这个仓库里都是什么

| 文件 | 作用 |
|---|---|
| `model.py` | **模型本体**：LLaMA 式结构（RMSNorm + RoPE 旋转位置编码 + SwiGLU 前馈 + 注意力 KV 缓存）。也能直接查参数量 |
| `tokenizer.py` | 分词器：把中文/英文切成 token，把 token 变回文字（字符级，零依赖） |
| `data.py` | 数据管道：预训练语料切块 + 问答数据打包 + 问答模板 |
| `train.py` | **预训练**：在纯文本上学习"下一个字是什么"，产出权重 |
| `sft.py` | **问答微调**：在问答数据上学习"问 -> 答"，只在回答部分算损失 |
| `chat.py` | **推理/问答**：加载权重，单次提问或连续多轮对话（逐字输出） |
| `configs/0.75b.json` | **0.75B 主配置** |
| `configs/tiny.json` | 微小配置（约 357 万参数），笔记本 CPU 也能几分钟跑通 |
| `data/corpus.txt` | 预训练语料（现在放的是示例文本，**换成你自己的大文本**） |
| `data/qa.jsonl` | 问答数据，一行一个 `{"q":"问题","a":"回答"}` |
| `一键试跑.bat` | 一条龙：建词表 → 预训练 → 问答微调 → 自动提问 |
| `一键问答.bat` | 打开连续对话 |
| `一键训练0.75B.bat` | 用 0.75B 配置正式训练（**必须有显卡**） |

> **权重去哪了？** 训练好的小模型权重（13.6 MB）没有放进本仓库（仓库只收代码和文本）。
> 想立刻看到它说话，跑一遍 `一键试跑.bat` 或下面的手动命令，就会生成 `out_tiny_sft/ckpt_final.pt`；
> `chat.py` 可以加载任何你自己训练出来的 `.pt` 权重。

---

## 二、三种玩法

### 玩法 1：先跑通（约 7~8 分钟，笔记本 CPU 就行）
双击 **`一键试跑.bat`**，它会自己建词表、训练、微调，最后问你一句"你好，你是谁？"。

或者手动敲：
```powershell
py -3.13 tokenizer.py build --data data/corpus.txt data/qa.jsonl --out tokenizer.json
py -3.13 train.py --config configs/tiny.json --data data/corpus.txt --out_dir out_tiny --max_steps 600 --batch_size 8 --seq_len 160 --eval_every 150
py -3.13 sft.py   --ckpt out_tiny/ckpt_final.pt --data data/qa.jsonl --out_dir out_tiny_sft --max_steps 400 --batch_size 8 --seq_len 96 --lr 2e-4
py -3.13 chat.py  --ckpt out_tiny_sft/ckpt_final.pt --question "你好，你是谁？"
```

本机实测（CPU，约 1 分钟/100 步）：
```
用户：你是谁
助手：我是一个用Python训练出来的语言模型，会一点问答。

用户：一加一等于几
助手：一加一等于二。

用户：什么是注意力
助手：注意力负责决定回答时该重点看哪些字。
```

### 玩法 2：连续问答
双击 **`一键问答.bat`**（或 `py -3.13 chat.py --ckpt out_tiny_sft/ckpt_final.pt`）

```
用户：你会做什么
助手：我会聊天、回答问题、也可以帮你念一段文字。
用户：清空          <- 忘掉前面的对话
用户：退出
```

### 玩法 3：真的训练 0.75B
```powershell
py -3.13 model.py --config configs/0.75b.json     # 先看参数：755,148,288（约 0.75B）
py -3.13 train.py --config configs/0.75b.json --data data/corpus.txt --out_dir out_0.75b `
        --max_steps 20000 --batch_size 4 --grad_accum 8
py -3.13 sft.py   --ckpt out_0.75b/ckpt_last.pt --data data/qa.jsonl --out_dir out_0.75b_sft --max_steps 2000
py -3.13 chat.py  --ckpt out_0.75b_sft/ckpt_final.pt
```

---

## 三、0.75B 配置到底多大

| 项目 | 数值 |
|---|---|
| 层数 n_layer | 24 |
| 隐藏维度 d_model | 1536 |
| 注意力头 n_head | 12（每头 128 维） |
| 前馈维度 ffn_hidden | 4224（SwiGLU） |
| 词表 vocab_size | 40000 |
| 上下文 max_seq_len | 1024 |
| **总参数量** | **755,148,288（7.55 亿 ≈ 0.75B）** |
| 半精度权重体积 | 约 1.4 GB |
| 训练显存需求 | 约 10 GB 起（AdamW 状态），建议 16 GB 以上显卡；A100/4090 上可再开 batch 4 + 梯度累积 8 |

对比一下：GPT-2 small 是 1.24 亿，GPT-2 XL 是 15.5 亿。0.75B 大概是"GPT-2 的 6 倍、GPT-3 的 1/233"。

---

## 四、换成你自己的数据

**预训练语料** `data/corpus.txt`：纯文本，多长都行（几十 KB 到几十 GB）。程序会自动切块。

**问答数据** `data/qa.jsonl`：一行一条，中文不用转义，但换行要写成 `\n`：
```json
{"q": "天空为什么是蓝色的", "a": "因为阳光穿过空气时，蓝光被散射得最多。"}
```

**注意**：语料变了必须**删掉 `tokenizer.json` 重新建词表**，否则会出现大量"不认识的字符"。

想让模型有点真本事，语料量级参考：认字 → 1 MB；说人话 → 100 MB；能答常识 → 1 GB 以上。

---

## 五、常见问题

**1) 报错 `No module named torch`**
用 `py -3.13` 运行（同一台机器上 `python` 可能指向没装 torch 的另一个解释器）。
装/更新：`py -3.13 -m pip install torch --index-url https://download.pytorch.org/whl/cpu`
（有 N 卡想用 GPU 版：`py -3.13 -m pip install torch --index-url https://download.pytorch.org/whl/cu126`）

**2) 0.75B 在 CPU 上能不能训？**
不能，一步要几分钟，跑 2 万步等于几年。程序会自动拦住你（除非你加 `--force_cpu`）。先用 tiny 跑通流程，大模型换带显卡的机器或云服务器。

**3) 回答驴唇不对马嘴？**
正常。示例语料只有 2000 多字，小模型等于只背了一本书的一页。想让它像样：换大语料 + 多训几轮 + 问答数据写得更细。

**4) 输出里有乱码/怪符号？**
分词器是在旧语料上建的，删 `tokenizer.json` 重建；另外 `<unk>` 已经在生成时禁掉了。

**5) 想加大上下文？**
把配置里的 `max_seq_len` 调大（1024 → 2048），显存占用跟着涨。

---

## 六、训练都在干什么（看懂日志）

```
step     150/600 | loss  4.xxx | lr 6.5e-04 | 3000 tok/s
  [验证] val_loss 4.xxx  <- 新最优，已保存 ckpt_best.pt
```

- **loss**：猜下一个字猜错了多少，越低越好
- **val_loss**：在没学过的文本上的分数。它开始上升就说明在死记硬背
- **lr**：学习率，先热身变大，后余弦退火变小
- **tok/s**：每秒处理多少 token，CPU 大约 3000，显卡能到几万

权重文件里同时存了模型参数、优化器状态、步数、配置和词表，所以 `--resume out_tiny/ckpt_last.pt` 可以断点续训，`chat.py` 也能单独靠一个 `.pt` 跑起来。
