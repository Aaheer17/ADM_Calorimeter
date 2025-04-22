#source code taken from: https://towardsdatascience.com/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1/

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import math
import numpy as np
from utils import *
from transformers import BertConfig, BertModel


# -------------------- Positional Encoding --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len=45):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))

        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, x):
        return self.dropout(x + self.pos_encoding[:x.size(0), :])

# -------------------- Hugging Face-Based Transformer Model --------------------
class CaloHFTransformer(nn.Module):
    def __init__(self, dim_model=64):
        super().__init__()
        config = BertConfig(
            hidden_size=dim_model,
            num_hidden_layers=4,
            num_attention_heads=4,
            intermediate_size=dim_model * 4,
            max_position_embeddings=128,
            is_decoder=True,
            add_cross_attention=True
        )
        self.encoder_proj = nn.Linear(1, dim_model)
        self.decoder_proj = nn.Linear(1, dim_model)

        self.encoder = BertModel(config)
        self.decoder = BertModel(config)
        self.output_proj = nn.Linear(dim_model, 1)

    def forward(self, src, tgt):
        src = self.encoder_proj(src.unsqueeze(1))  # (B, 1, D)
        tgt = self.decoder_proj(tgt)               # (B, 45, D)

        enc_outputs = self.encoder(inputs_embeds=src)
        dec_outputs = self.decoder(inputs_embeds=tgt, encoder_hidden_states=enc_outputs.last_hidden_state)

        return self.output_proj(dec_outputs.last_hidden_state)  # (B, 45, 1)


# -------------------- Transformer Model --------------------
class CaloTransformer(nn.Module):
    def __init__(self, dim_model=128, num_heads=4, num_encoder_layers=2, num_decoder_layers=2, dropout_p=0.1,max_len=45):
        super().__init__()
        self.dim_model = dim_model

        self.encoder_proj = nn.Linear(1, dim_model)
        self.decoder_proj = nn.Linear(1, dim_model)

        self.positional_encoder = PositionalEncoding(dim_model, dropout_p)

        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p
        )

        self.output_proj = nn.Linear(dim_model, 1)

    def forward(self, src, tgt, tgt_mask=None):
        # src: (B, 1), tgt: (B, 45, 1)
        src = src.unsqueeze(1)  
        src = self.encoder_proj(src)       # (B, 1, D)
        tgt = self.decoder_proj(tgt)       # (B, 45, D)
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        
        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        out = out.permute(1, 0, 2)

        return self.output_proj(out)       # (B, 45, 1)


# -------------------- Sampling Function --------------------
@torch.no_grad()
def sampling_validate(model, incident_energies, num_layers=45):
    model.eval()
    B = incident_energies.shape[0]
    device = incident_energies.device

    generated = torch.zeros(B, 0, 1, device=device)

    for i in range(num_layers):
        tgt_mask = model.transformer.generate_square_subsequent_mask(generated.size(1) + 1).to(device)
        current_input = torch.cat([generated, torch.zeros(B, 1, 1, device=device)], dim=1)
        out = model(incident_energies, current_input, tgt_mask=tgt_mask)
        print("shape of out: ",out.shape)
        next_val = out[:, -1:, :]
        generated = torch.cat([generated, next_val], dim=1)
    print("shape of validation sampling: ", generated.shape)
    return generated


# -------------------- Train Loop --------------------
def train_loop(model, optimizer, loss_fn, dataloader, device):
    model.train()
    total_loss = 0

    for batch in dataloader:
        layers, Einc = batch  # Einc: (B,1), layers: (B,45,1)
        Einc, layers = Einc.to(device), layers.to(device)

        input_seq = layers[:, :-1, :]
        target_seq = layers[:, 1:, :]

        tgt_mask = model.transformer.generate_square_subsequent_mask(input_seq.shape[1]).to(device)
        pred = model(Einc, input_seq, tgt_mask=tgt_mask)

        loss = loss_fn(pred, target_seq)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# -------------------- Validation Loop --------------------
@torch.no_grad()
def validation_loop(model, loss_fn, dataloader, device):
    model.eval()
    total_loss = 0
    
    for batch in dataloader:
        layers, Einc = batch
        Einc, layers = Einc.to(device), layers.to(device)

        input_seq = layers[:, :-1, :]
        target_seq = layers[:, 1:, :]

        tgt_mask = model.transformer.generate_square_subsequent_mask(input_seq.shape[1]).to(device)
        pred = model(Einc, input_seq, tgt_mask=tgt_mask)

        loss = loss_fn(pred, target_seq)
        total_loss += loss.item()

    return total_loss / len(dataloader)
@torch.no_grad()
def sampling_val_loop(model, dataloader,device,epoch):
   

    model.eval()
    generated = []
    reference = []
    energies = []
    num_samples=1000
    results_folder = Path("./Bert_output")
    results_folder.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        sample_count = 0
        for batch in dataloader:
            # Adjust this depending on your dataloader structure
            real_shower, Einc = batch  # (B, 45, 1), (B, 1)
            batch_size = real_shower.shape[0]

            # Generate synthetic showers
            gen_shower = sampling_validate(model, Einc)
            
            generated.append(gen_shower.cpu())
            reference.append(real_shower.cpu())
            energies.append(Einc.cpu())

            sample_count += batch_size
            if sample_count >= num_samples:
                break

    # Stack all outputs

    generated = torch.cat(generated, dim=0)[:num_samples]  # (N, 45, 1)
    reference = torch.cat(reference, dim=0)[:num_samples]  # (N, 45, 1)
    energies = torch.cat(energies, dim=0)[:num_samples]    # (N, 1)
    
    ## need to update the path
    torch.save(generated,results_folder /f'generated_embed_128_{epoch}.pt')

    torch.save(reference,results_folder /f"reference_embed_128_{epoch}.pt")



# -------------------- Fit Function --------------------
def fit(model, optimizer, loss_fn, train_loader, val_loader, device, epochs=10):
    loss_t=[]
    loss_v=[]
    for epoch in range(epochs):
        train_loss = train_loop(model, optimizer, loss_fn, train_loader, device)
        val_loss = validation_loop(model, loss_fn, val_loader, device)
        sampling_val_loop(model, val_loader,device,epoch)
        loss_t.append(train_loss)
        loss_v.append(val_loss)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
    plot_losses(loss_t,loss_v,file_name='transformer_loss.pdf')


