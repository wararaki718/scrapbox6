from typing import List, Union

import torch


def try_gpu(x: Union[torch.Tensor, torch.nn.Module]) -> Union[torch.Tensor, torch.nn.Module]:
    if torch.cuda.is_available():
        return x.cuda()
    else:
        return x


def show_data(
    queries: List[str],
    positive_documents: List[str],
    negative_documents: List[str],
) -> None:
    print(f"queries: {len(queries)}")
    print(f"positive documents: {len(positive_documents)}")
    print(f"negative documents: {len(negative_documents)}")
    print()
