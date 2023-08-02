# -*- coding: utf-8 -*-
"""combine all seperate data into one file for pretrain
"""
import yaml
import torch

confpath = "conf/pretrain/pretrain-exp3.yaml"
datapath = "data/mcbench/"
savepath = "data/pretrain/"

if __name__ == '__main__':

    with open(confpath) as f:
        kwargs = yaml.safe_load(f)
        f.close()
    gateway_pos = kwargs['dataset']['gateways_pos']
    train_scenes = kwargs['dataset']['train_scenes']
    test_scenes = kwargs['dataset']['test_scenes']

    # train_scenes = ["s01",        "s03", "s04"]

    trainset = torch.empty(0)
    for scene in train_scenes:
        train_data = torch.load(f"{datapath}train_data-{scene}-seq1.pt")  # [N, 1, dim]
        test_data = torch.load(f"{datapath}test_data-{scene}-seq1.pt")  # [N, 1, dim]
        all_data = torch.concat([train_data, test_data], dim=0)
        gateway_ind = int(scene[1:]) - 1
        all_gateway_pos = torch.tensor(gateway_pos[gateway_ind]).view(-1) #[9]
        all_gateway_pos = all_gateway_pos.unsqueeze(0).unsqueeze(0).repeat(all_data.shape[0], 1, 1)

        if trainset.shape[0] == 0:
            trainset = torch.cat([all_gateway_pos, all_data], dim=-1)
        else:
            temp = torch.cat([all_gateway_pos, all_data], dim=-1)
            trainset = torch.cat([trainset, temp], dim=0)
        print(f"len train_data-{scene}-seq1.t", len(all_data))
        print(f"len trainset", len(trainset))

    torch.save(trainset, f"{savepath}train_data-all-seq1-normgateway.pt")

