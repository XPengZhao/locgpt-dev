import os
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
l = 4

# p1 = torch.tensor([[3.0017452239990234,3.983450412750244,0.34878233075141907]])
# p2 = torch.tensor([[-2.931766986846924,4.035231113433838,0.34878233075141907]])
# p3 = torch.tensor([[0.17407386004924774,-4.984780788421631,0.34878233075141907]])

def spherical2pointandline(x):
    # print(x.shape)
    alpha, beta = x[:, 0].reshape((-1, 1)), x[:, 1].reshape((-1, 1))
    return torch.cat((torch.sin(beta) * torch.cos(alpha), torch.sin(beta) * torch.sin(alpha), torch.cos(beta)), dim=1)


def intersect_or_distance(p1, v1, p2, v2):
    p1.expand(v1.shape[0], 3)
    p2.expand(v2.shape[0], 3)
    if torch.all(torch.cross(v1, v2) == 0):
        dist = distance_between_lines(p1, v1, p2, v2)
        return 1, dist
    s = compute_intersection(p1, v1, p2, v2)
    if isinstance(s, int):
        return 3, distance_between_lines(p1, v1, p2, v2)
    return 2, s


def compute_intersection(p1, v1, p2, v2):
    v1_ = v1.unsqueeze(-1).reshape(-1, 3, 1)
    v2_ = v2.unsqueeze(-1).reshape(-1, 3, 1)
    one = torch.ones(v1_.shape)
    A = torch.cat((v1_, v2_, one.to(v1.device)), dim=2)
    A_inv = torch.pinverse(A)
    b = (p2 - p1).unsqueeze(-1)
    # x = torch.linalg.solve(A, b)
    x = torch.matmul(A_inv, b).squeeze()
    zero = x[:, 2].unsqueeze(-1)
    if not torch.allclose(zero, torch.zeros_like(zero)):
        return 1
    p = p1 + v1 * x[:, 0].unsqueeze(-1)
    return p


def distance_between_lines(p1, v1, p2, v2):
    # Calculate the direction vector perpendicular to both lines
    cross_product = torch.cross(v1, v2, dim=1)
    # cross_product = torch.clamp(cross_product, min=1e-15)
    # Calculate the distance between the two lines
    distance = torch.abs(torch.einsum('ij, ij->i', (p2 - p1, cross_product))) / torch.norm(cross_product, dim=1)

    return distance.unsqueeze(1)


def area(x1, x2, x3, gateway_pos):
    p1,p2,p3 = gateway_pos[:,0:3], gateway_pos[:,3:6], gateway_pos[:,6:9]
    v1 = spherical2pointandline(x1)
    v2 = spherical2pointandline(x2)
    v3 = spherical2pointandline(x3)
    res = [intersect_or_distance(p1, v1, p2, v2), intersect_or_distance(p2, v2, p3, v3),
           intersect_or_distance(p3, v3, p1, v1)]
    zero = torch.zeros((x1.shape[0], 1))
    s = zero.to(v1.device)
    if res[0][0] + res[1][0] + res[2][0] == 6:
        s += tri(res[0][1], res[1][1], res[2][1])
    else:
        for flag, r in res:
            if flag != 2:
                s += r
    # print(s)
    return s



def tri(a, b, d):
    def angle2point(alpha, beta, d):
        return torch.cat(
            (d * torch.sin(beta) * torch.cos(alpha), d * torch.sin(beta) * torch.sin(alpha), d * torch.cos(beta)),
            dim=1)

    def triangle_area(p_1, p_2, p_3):
        a_ = torch.norm(p_2 - p_3, dim=1)
        b_ = torch.norm(p_1 - p_3, dim=1)
        c = torch.norm(p_1 - p_2, dim=1)
        s_ = (a_ + b_ + c) / 2
        area_ = torch.sqrt(s_ * (s_ - a_) * (s_ - b_) * (s_ - c))
        return area_

    p1_ = angle2point(a[:, 0].reshape(-1, 1), b[:, 0].reshape(-1, 1), d[:, 0].reshape(-1, 1))
    p2_ = angle2point(a[:, 1].reshape(-1, 1), b[:, 1].reshape(-1, 1), d[:, 1].reshape(-1, 1))
    p3_ = angle2point(a[:, 2].reshape(-1, 1), b[:, 2].reshape(-1, 1), d[:, 2].reshape(-1, 1))
    s = triangle_area(p1_, p2_, p3_).reshape((-1, 1))
    print(s)
    return s





