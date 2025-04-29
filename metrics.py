import torch

# def InfoNCE(view1, view2, temperature):
#     view1, view2 = torch.nn.functional.normalize(
#         view1, dim=1), torch.nn.functional.normalize(view2, dim=1)
#     pos_score = (view1 * view2).sum(dim=-1)
#     pos_score = torch.exp(pos_score / temperature)
#     ttl_score = torch.matmul(view1, view2.transpose(0, 1))
#     ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
#     cl_loss = -torch.log(pos_score / ttl_score)
#     return torch.mean(cl_loss)


def InfoNCE(view1, view2, temperature):
    view1 = torch.nn.functional.normalize(view1, dim=1)
    view2 = torch.nn.functional.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score)
    return torch.mean(cl_loss)


# def InfoNCE_i(view1, view2, view3,temperature,gama):
#     view1, view2,view3 = torch.nn.functional.normalize(
#         view1, dim=1), torch.nn.functional.normalize(view2, dim=1), torch.nn.functional.normalize(view3, dim=1)
#     pos_score = (view1 * view2).sum(dim=-1)
#     pos_score = torch.exp(pos_score / temperature)
#     ttl_score_1 = torch.matmul(view1, view2.transpose(0, 1))
#     ttl_score_1 = torch.exp(ttl_score_1 / temperature).sum(dim=1)
#     ttl_score_2 = torch.matmul(view1, view3.transpose(0, 1))
#     ttl_score_2 = torch.exp(ttl_score_2 / temperature).sum(dim=1)

#     cl_loss = -torch.log(pos_score / (gama*ttl_score_2+ttl_score_1+pos_score))
#     return torch.mean(cl_loss)

def InfoNCE_i(view1, view2, view3, temperature, gama):
    view1 = torch.nn.functional.normalize(view1, dim=1)
    view2 = torch.nn.functional.normalize(view2, dim=1)
    view3 = torch.nn.functional.normalize(view3, dim=1)

    pos_score = (view1 * view2).sum(dim=-1)  # [batch_size]

    # Support scalar or vector temperature
    if isinstance(temperature, torch.Tensor):
        # temperature: [batch_size] => [batch_size, 1] for broadcasting
        temperature = temperature.view(-1, 1)

    # Compute exp(pos / T)
    pos_score_exp = torch.exp(pos_score / temperature.squeeze())

    # Total scores
    ttl_score_1 = torch.matmul(view1, view2.transpose(0, 1))  # [batch_size, batch_size]
    ttl_score_2 = torch.matmul(view1, view3.transpose(0, 1))  # [batch_size, batch_size]

    ttl_score_1 = torch.exp(ttl_score_1 / temperature)  # [batch_size, batch_size]
    ttl_score_2 = torch.exp(ttl_score_2 / temperature)  # [batch_size, batch_size]

    ttl_score_1_sum = ttl_score_1.sum(dim=1)
    ttl_score_2_sum = ttl_score_2.sum(dim=1)

    denom = gama * ttl_score_2_sum + ttl_score_1_sum + pos_score_exp
    cl_loss = -torch.log(pos_score_exp / denom)

    return torch.mean(cl_loss)

