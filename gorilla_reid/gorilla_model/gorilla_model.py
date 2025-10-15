import os
import torch
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader
# from gorilla_reid.k_nearest_neighbor import calculate_accuracy, get_dataset_embeddings
from . import k_nearest_neighbor

device = "cuda:0"


def train(
    model: nn.Module,
    epochs: int,
    learning_rate: float,
    # Datasets
    dataloader_train: DataLoader,
    dataloader_val: DataLoader,
    checkpoint_dir: str = "./checkpoints",
    save_every: int = 1,  # save every n epochs
):
    os.makedirs(checkpoint_dir, exist_ok=True)
    # --- Try to resume from checkpoint ---
    latest_checkpoint = None
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
        if checkpoints:
            # Get the latest checkpoint (by modification time)
            latest_checkpoint = max([os.path.join(checkpoint_dir, f) for f in checkpoints], key=os.path.getmtime)

    calculate_loss = nn.TripletMarginLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if latest_checkpoint:
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        print(checkpoint.keys())
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_epoch = checkpoint["epoch"]
        train_loss = checkpoint["train_loss"]
        val_loss = checkpoint["val_loss"]
        start_epoch = checkpoint["epoch"]
        print(f"Resumed from epoch {start_epoch}")
    else:
        start_epoch = 0
        print("No checkpoint found — starting from scratch.")

    model = model.to(device)

    with wandb.init(project="GORILLA_FACE", name="training", config={"epochs": epochs}) as run:
        run.watch(model, log_freq=100)

        best_val_loss = float("inf")
        for epoch in range(start_epoch, epochs):
            # Training.
            model.train()
            print(f"Starting epoch {epoch}")

            run.log(
                {
                    "epoch": epoch,
                }
            )

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
                    run.log({"batch_loss": loss.item(), "batch": epoch * len(dataloader_train) + idx})

                loss.backward()

                optimizer.step()

            train_avg_loss = train_avg_loss / len(dataloader_train)

            # Validation
            val_avg_loss = 0.0
            val_accuracy_sum = 0
            model.eval()

            with torch.no_grad():
                annotated_search_space = k_nearest_neighbor.get_dataset_embeddings(dataloader=dataloader_val, model=model, device=device)

                for [idx, batch] in enumerate(dataloader_val):
                    labels, this_class, in_class, out_class = batch

                    this_class = this_class.to(device)
                    in_class = in_class.to(device)
                    out_class = out_class.to(device)

                    optimizer.zero_grad()

                    output_anchor = model(this_class)
                    in_class_emb = model(in_class)
                    out_class_emb = model(out_class)

                    query_embeddings = [output_anchor[index] for index in range(output_anchor.shape[0])]
                    annotated_query_embeddings = list(zip(labels, query_embeddings))
                    val_accuracy_sum += k_nearest_neighbor.calculate_accuracy(
                        annotated_search_space=annotated_search_space, annotated_queries=annotated_query_embeddings
                    )

                    loss = calculate_loss(output_anchor, in_class_emb, out_class_emb)
                    val_avg_loss += loss.item()

            val_avg_loss = val_avg_loss / len(dataloader_val)
            val_avg_accuracy = val_accuracy_sum / len(dataloader_val)

            run.log(
                {
                    "train/loss": train_avg_loss,
                    "val/loss": val_avg_loss,
                    "val/accuracy": val_avg_accuracy,
                    "epoch": epoch,
                }
            )

            # Save checkpoint
            if (epoch + 1) % save_every == 0 or val_avg_loss < best_val_loss:
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")

                # Track best model
                if val_avg_loss < best_val_loss:
                    best_val_loss = val_avg_loss
                    best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
                    torch.save(model.state_dict(), best_model_path)
                    print(f"Best model updated at epoch {epoch+1}")

                # Save everything you need to resume training perfectly
                torch.save(
                    {
                        "epoch": epoch + 1,  # next epoch to start from
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_avg_loss,
                        "val_loss": val_avg_loss,
                    },
                    f"{checkpoint_dir}/epoch_{epoch+1}.pt",
                )

                print(f"Checkpoint saved at {checkpoint_path}")
