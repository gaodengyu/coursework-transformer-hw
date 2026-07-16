# Transformer Homework

> Transformer 架构实现 | BSAI @ M.U.S.T.

---

## 内容

从头实现 Transformer 架构并在算术任务上验证。

| 文件 | 说明 |
|------|------|
| `model.py` | Transformer 模型定义（Multi-Head Attention, Positional Encoding, Feed-Forward） |
| `train.py` | 训练循环与评价 |
| `data_gen.py` | 算术数据集生成（加法任务） |
| `main.py` | 入口脚本 |

## 任务

给定两个数字（如 `123 + 456`），模型输出正确和（`579`）。通过自回归解码逐字符生成结果。

## 关键技术

- Multi-Head Self-Attention
- Positional Encoding（序列位置感知）
- Encoder-Decoder 架构
- Teacher Forcing 训练

---

## 运行

```bash
python main.py
```

## 运行环境

- Python 3.10+
- PyTorch
