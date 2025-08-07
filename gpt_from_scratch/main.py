# main.py

import torch
from utils.bpe import BPETokenizer
from utils.data import TokenDataset, get_batch
from models.transformer import GPTConfig, GPTModel

# === Configuration ===
device = 'cuda' if torch.cuda.is_available() else 'cpu'
block_size = 64
batch_size = 32
num_epochs = 30
lr = 3e-4

# === Load and tokenize data ===
def load_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

train_text = load_file("D:\coxi\Building GPT from scratch\Blockseminar_Building_GPT_from_scratch\corpora\gpt_from_scratch\corpora\Shakespeare_clean_train.txt")
val_text   = load_file("D:\coxi\Building GPT from scratch\Blockseminar_Building_GPT_from_scratch\corpora\gpt_from_scratch\corpora\Shakespeare_clean_valid.txt")
test_text  = load_file("gpt_from_scratch/corpora/Shakespeare_clean_test.txt")

# === Initialize and train BPE tokenizer ===
bpe = BPETokenizer(num_merges=500)
bpe.learn_bpe(bpe.normalize(train_text))

train_ids = bpe.encode(train_text)
val_ids   = bpe.encode(val_text)
test_ids  = bpe.encode(test_text)

vocab_size = len(bpe.token2id)
print(f"✅ Vocab size: {vocab_size}")

# === Prepare datasets ===
train_dataset = TokenDataset(train_ids, block_size)
val_dataset   = TokenDataset(val_ids, block_size)
test_dataset  = TokenDataset(test_ids, block_size)

# === Initialize model ===
config = GPTConfig(
    vocab_size=vocab_size,
    block_size=128,       # ⬆️ from 64
    n_layer=6,            # ⬆️ from 4
    n_head=8,             # ⬆️ from 4
    n_embd=256,           # ⬆️ from 128
    dropout=0.1
)
model = GPTModel(config).to(device)
# === Optimizer and Scheduler ===
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)

# === Training Loop ===
print("🚀 Training started...")
for epoch in range(num_epochs):
    model.train()
    losses = []

    for _ in range(100):  # steps per epoch
        xb, yb = get_batch(train_dataset, batch_size)
        xb, yb = xb.to(device), yb.to(device)
        logits, loss = model(xb, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")

# === Evaluation ===
model.eval()
@torch.no_grad()
def evaluate(dataset):
    xb, yb = get_batch(dataset, batch_size=128)
    xb, yb = xb.to(device), yb.to(device)
    _, loss = model(xb, yb)
    return torch.exp(loss).item()

val_ppl  = evaluate(val_dataset)
test_ppl = evaluate(test_dataset)

print("\n📊 Evaluation Perplexity:")
print(f"Validation PPL: {val_ppl:.2f}")
print(f"Test PPL:      {test_ppl:.2f}")

# === Generation ===
print("\n📝 Generated Text:")
seed = "love"
encoded = bpe.encode(seed)
context = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
generated = model.generate(context, max_new_tokens=50)[0].tolist()
output = bpe.decode(generated)
print(output)
