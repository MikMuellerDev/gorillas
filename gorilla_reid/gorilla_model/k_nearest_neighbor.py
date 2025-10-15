from typing import Counter, List, Tuple
import torch
from tqdm.notebook import tqdm


def find_nearest_neighbors(
    query: torch.Tensor, search_space: List[torch.Tensor], k: int, is_query_in_search_space: bool = False
) -> List[int]:
    search_space_tensor = torch.stack(search_space)
    distances = torch.cdist(query.unsqueeze(0), search_space_tensor)

    k_to_request = k + 1 if is_query_in_search_space else k
    top_values, top_indexes = torch.topk(distances, k=k_to_request, largest=False, dim=1)

    top_k_values = top_values[:, 1:] if is_query_in_search_space else top_values
    top_k_indexes = top_indexes[:, 1:] if is_query_in_search_space else top_values

    return top_k_indexes.squeeze(0).tolist()


def find_knn_label(
    query: torch.Tensor, annotated_search_space: List[Tuple[str, torch.Tensor]], k: int, is_query_in_search_space: bool = False
):
    search_space = [embedding for label, embedding in annotated_search_space]

    top_k_indexes = find_nearest_neighbors(
        query=query, search_space=search_space, k=k, is_query_in_search_space=is_query_in_search_space
    )
    top_k_labels = [annotated_search_space[index][0] for index in top_k_indexes]
    predicted_label = Counter(top_k_labels).most_common(1)[0][0]

    return predicted_label


def get_dataset_embeddings(
    dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, device: str, show_progress: bool = True
) -> List[Tuple[str, torch.Tensor]]:
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        total_files = len(dataloader)
        pbar = tqdm(total=total_files, desc="Visualizing embeddings...") if show_progress else None
        for batch in dataloader:
            labels, this_class, _, _ = batch

            all_labels.extend(labels)

            this_class = this_class.to(device)
            output_emb = model(this_class)

            all_embeddings.append(output_emb.to(device))

            if pbar:
                pbar.update(1)

    all_embeddings = torch.cat(all_embeddings, dim=0)

    return list(zip(all_labels, all_embeddings))


def calculate_accuracy(
    annotated_search_space: List[Tuple[str, torch.Tensor]],
    annotated_queries: List[Tuple[str, torch.Tensor]] = None,
) -> float:
    correct_label_count = 0

    annotated_queries = annotated_queries or annotated_search_space

    for label, embedding in annotated_queries:
        predicted_label = find_knn_label(
            query=embedding, annotated_search_space=annotated_search_space, is_query_in_search_space=True, k=5
        )

        if label == predicted_label:
            correct_label_count += 1

    return correct_label_count / len(annotated_queries)


def calculate_dataset_accuracy(dataloader: torch.utils.data.DataLoader, model: torch.nn.Module, device: str) -> float:
    all_labels, all_embeddings = get_dataset_embeddings(dataloader=dataloader, model=model, device=device)
    return calculate_accuracy(all_labels=all_labels, all_embeddings=all_embeddings, model=model, device=device)
