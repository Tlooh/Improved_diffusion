import torch 
from torch.nn import functional as F
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

from geomloss import SamplesLoss

def loss1(map_a, map_b) -> torch.Tensor:
    """ Computes the IOU loss using the maximum attention value for each token. """
    scale_factor = 100
    map_a *= scale_factor
    map_b *= scale_factor
    map_a = F.softmax(map_a.float(), dim=-1)
    map_b = F.softmax(map_b.float(), dim=-1)


    min_numerator = torch.min(map_a, map_b).sum()
    sum_denominator = (map_a + map_b).sum()

    loss = torch.div(min_numerator, sum_denominator)

    return loss


def loss2(map_a, map_b) -> torch.Tensor:

    scale_factor = 100

    map_a *= scale_factor
    map_b *= scale_factor

    map_a = F.softmax(map_a.float(), dim=-1)
    map_b = F.softmax(map_b.float(), dim=-1)
    # # # 构建 矩阵，值为 1
    a_one = torch.where(map_a > 0.02, 1, 0)
    b_one = torch.where(map_b > 0.02, 1, 0)

    # 并集，交集, 如果（i,j）像素位置有重叠，就为 1
    intersection = torch.mul(a_one, b_one)
    # union = a_one + b_one - intersection

    min_numerator = torch.min(torch.mul(map_a, intersection), torch.mul(map_b, intersection)).sum()

    sum_denominator = (torch.mul(map_a, a_one) + torch.mul(map_b, b_one)).sum()

    loss = torch.div(torch.mul(map_a, intersection).sum(), torch.mul(map_a, a_one).sum()) + torch.div(
        torch.mul(map_b, intersection).sum(), torch.mul(map_b, b_one).sum())

    return loss


def loss3(map_a, map_b) -> torch.Tensor:

    scale_factor = 100

    map_a *= scale_factor
    map_b *= scale_factor

    
    Loss =  SamplesLoss("sinkhorn", blur=0.05,)

    probs1 = F.softmax(map_a.float(), dim=-1)
    probs2 = F.softmax(map_b.float(), dim=-1)

    # 计算Wasserstein距离
    wasserstein_distance =  Loss(probs1, probs2)


    loss = max(0, 1. - wasserstein_distance)



    return loss


def loss4(map_a, map_b) -> torch.Tensor:

    scale_factor = 100

    map_a *= scale_factor
    map_b *= scale_factor

    map_a = F.softmax(map_a, dim=-1)
    map_b = F.softmax(map_b, dim=-1)

    avg_attn_a = torch.mean(map_a)
    avg_attn_b = torch.mean(map_b)
    # print(avg_attn_a)
    # # # 构建 矩阵，值为 1
    print(f"map_a mean:{torch.mean(map_a)}, map_a max: {torch.max(map_a)}")

    a_one = torch.where(map_a > avg_attn_a, 1, 0)
    b_one = torch.where(map_b > avg_attn_b, 1, 0)


    # 并集，交集, 如果（i,j）像素位置有重叠，就为 1
    intersection = a_one * b_one

    map_a_intersection = torch.mul(map_a, intersection)
    map_a_all = torch.mul(map_a, a_one)
    loss_a = torch.div(map_a_intersection.sum(), map_a_all.sum())

    map_b_intersection = torch.mul(map_b, intersection)
    map_b_all = torch.mul(map_b, b_one)
    loss_b = torch.div(map_b_intersection.sum(), map_b_all.sum())

    loss = loss_a + loss_b 
    return loss