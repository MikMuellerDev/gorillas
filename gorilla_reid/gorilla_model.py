from typing import List, Tuple
import torch
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader

device = 'cuda:0'

def train(
    model: nn.Module,
    epochs: int,
    learning_rate: float,
    # Datasets
    dataloader_train: DataLoader,
    dataloader_val: DataLoader
    ):
    with wandb.init(project="GORILLA_FACE", name="training", config={"epochs": epochs}) as run:
        calculate_loss = nn.TripletMarginLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model = model.to(device)

        run.watch(model, log_freq=100)

        for epoch in range(epochs):
            # Training.
            model.train()
            print(f'Starting epoch {epoch}')

            run.log({
            "epoch": epoch,
            })

            train_avg_loss = 0.0

            for [idx, batch] in enumerate(dataloader_train):
                _, this_class, in_class, out_class = batch
                
                this_class = this_class.to(device)
                in_class = in_class.to(device)
                out_class = out_class.to(device)

                # data = data.to(device)
                # labels = cpu_labels.to(device)
                
                optimizer.zero_grad()
                
                output_anchor = model(this_class)
                in_class_emb = model(in_class)
                out_class_emb = model(out_class)

                loss = calculate_loss(output_anchor, in_class_emb, out_class_emb)
                train_avg_loss += loss.item()

                if idx % 50 == 0:
                    run.log({
                    "batch_loss": loss.item(),
                    "batch": epoch * len(dataloader_train) + idx
                    })

                loss.backward()
                
                optimizer.step()

            train_avg_loss = train_avg_loss / len(dataloader_train)
            
            # Validation
            val_avg_loss = 0.0
            model.eval()
            
            with torch.no_grad():
                for [idx, batch] in enumerate(dataloader_val):
                    _, this_class, in_class, out_class = batch
                    
                    this_class = this_class.to(device)
                    in_class = in_class.to(device)
                    out_class = out_class.to(device)

                    # data = data.to(device)
                    # labels = cpu_labels.to(device)
                    
                    optimizer.zero_grad()
                    
                    output_anchor = model(this_class)
                    in_class_emb = model(in_class)
                    out_class_emb = model(out_class)

                    loss = calculate_loss(output_anchor, in_class_emb, out_class_emb)
                    val_avg_loss += loss.item()


            val_avg_loss = val_avg_loss / len(dataloader_val)

            run.log({
            "train/loss": train_avg_loss,
            "val/loss": val_avg_loss,
            "epoch": epoch,
            })
