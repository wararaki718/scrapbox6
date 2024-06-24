import torch


def main() -> None:
    #x_a = torch.Tensor([[1, 2, 3]])
    #x_b = torch.Tensor([[5, 6, 7]])
    x_a = torch.randn(1, 128)
    x_b = torch.randn(10, 128)
    y = torch.nn.functional.cosine_similarity(x_a, x_b)
    print(f"cosine similarity: {y}")
    print()
    
    y = torch.dot(x_a[0], x_b[0])
    print(f"dot product: {y}")
    print()

    y = torch.matmul(x_a, x_b.T)
    print(f"matmul: {y}")
    print()

    x_a_norm = torch.norm(x_a, dim=1)
    x_b_norm = torch.norm(x_b, dim=1)
    # print(x_a_norm)
    # print(x_b_norm)
    # print()
    y = torch.matmul(x_a, x_b.T) / (x_a_norm * x_b_norm)
    print(f"cos-sim (matmul with norm): {y}")
    print()

    x_a_norm = x_a / torch.norm(x_a, dim=1).reshape(-1, 1)
    x_b_norm = x_b / torch.norm(x_b, dim=1).reshape(-1, 1)
    y = torch.matmul(x_a_norm,  x_b_norm.T)
    print(f"dot product: {y}")
    print()
    print("DONE")


if __name__ == "__main__":
    main()
