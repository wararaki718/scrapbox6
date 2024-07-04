import torch
import torch.nn as nn


def main() -> None:
    # (batchsize, n_data, dim)
    query = torch.randn((2, 5, 4))
    key = torch.randn((2, 5, 4))
    value = torch.randn((2, 5, 4))

    print("inputs:")
    print(query.shape)
    print(key.shape)
    print(value.shape)
    print()

    dim = 4
    n_heads = 2
    model = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads)
    
    output, weights = model(query, key, value)
    print("output:")
    print(output)
    print()

    print("weights:")
    print(weights)
    print()

    print("DONE")


if __name__ == "__main__":
    main()
