
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VOCAB, MaestroDataset
from model import PianoTransformer

input_dim = 128  # n_mels
num_tokens = len(VOCAB)
d_model = 256
batch_size = 8
epochs = 10
lr = 1e-3

PAD_TOKEN_ID = -100

dataset = MaestroDataset(split='train', num_samples=1_000_000)
dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

model = PianoTransformer(input_dim, num_tokens, d_model=d_model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
        features_batch, tokens_batch = zip(*batch)
        features_batch = nn.utils.rnn.pad_sequence(list(features_batch), batch_first=True)
        tokens_batch = nn.utils.rnn.pad_sequence(list(tokens_batch), batch_first=True, padding_value=PAD_TOKEN_ID)
        features_batch = features_batch.to(device)
        tokens_batch = tokens_batch.to(device)

        optimizer.zero_grad()
        logits = model(features_batch)  # (batch, seq_len, num_tokens)

        if logits.shape[1] < tokens_batch.shape[1]:
            pad_size = tokens_batch.shape[1] - logits.shape[1]
            logits = torch.nn.functional.pad(logits, (0, 0, 0, pad_size))
        elif logits.shape[1] > tokens_batch.shape[1]:
            logits = logits[:, :tokens_batch.shape[1], :]

        logits = logits.reshape(-1, num_tokens)
        targets = tokens_batch.reshape(-1)

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

    torch.save(model.state_dict(), "pianomodel.pt")
