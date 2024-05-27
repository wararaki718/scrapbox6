import torch


cosine_similarity = torch.nn.CosineSimilarity()

a = torch.randn(1, 128)
b = torch.randn(5, 128)

output = cosine_similarity(a, b)
print(output.shape)
print(output)

pairwise_distance = torch.nn.PairwiseDistance()
output = pairwise_distance(a, b)
print(output.shape)
print(output)
