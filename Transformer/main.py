import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data_gen import AdditionDataset 
from model import TinyTransformerLM

batch_size = 64
epochs = 20
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    print("正在准备数据集...")
    train_dataset = AdditionDataset(num_samples=20000)
    val_dataset = AdditionDataset(num_samples=2000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    vocab_size = train_dataset.vocab_size
    model = TinyTransformerLM(vocab_size=vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    train_losses, val_losses = [], []

    print(f"开始在 {device} 上训练...")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs.view(-1, vocab_size), y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train = total_train_loss / len(train_loader)
        train_losses.append(avg_train)
        
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                val_loss = criterion(outputs.view(-1, vocab_size), y.view(-1))
                total_val_loss += val_loss.item()
        
        avg_val = total_val_loss / len(val_loader)
        val_losses.append(avg_val)
        scheduler.step(avg_val)
        print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train:.4f} - Val Loss: {avg_val:.4f}  lr: {optimizer.param_groups[0]['lr']:.6f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    print("\n--- 最终模型实测 ---")
    test_cases = ["123+456=", "999+1=", "520+131="]
    model.eval()
    
    for test_str in test_cases:
        input_indices = [train_dataset.stoi[c] for c in test_str]
        input_tensor = torch.tensor(input_indices).unsqueeze(0).to(device)
        
        generated = test_str
        for _ in range(6):
            with torch.no_grad():
                logits = model(input_tensor)
                next_token = torch.argmax(logits[0, -1, :]).item()
                next_char = train_dataset.itos[next_token]
                
                if next_char == " ": 
                    break
                generated += next_char
            
                new_token_tensor = torch.tensor([[next_token]]).to(device)
                input_tensor = torch.cat([input_tensor, new_token_tensor], dim=1)
        
        print(f"输入: {test_str:10} -> 模型输出: {generated}")
    torch.save(model.state_dict(), "addition_model.pth")
    print("\n模型权重已保存为 addition_model.pth")

if __name__ == "__main__":
    train()