# -*- coding: utf-8 -*-
"""combine all seperate data into one file for pretrain
"""
import yaml
import torch

confpath = "conf/pretrain/pretrain-exp3.yaml"
datapath = "data/mcbench/"
savepath = "data/fine-tune/"

if __name__ == '__main__':

    with open(confpath) as f:
        kwargs = yaml.safe_load(f)
        f.close()
    gateway_pos = kwargs['dataset']['gateways_pos']
    train_scenes = kwargs['dataset']['train_scenes']
    test_scenes = kwargs['dataset']['test_scenes']
    test_scenes = ["s20"]


    for scene in test_scenes:
        train_data = torch.load(f"{datapath}train_data-{scene}-seq1.pt")  # [N, 1, dim]
        test_data = torch.load(f"{datapath}test_data-{scene}-seq1.pt")  # [N, 1, dim]
        all_data = torch.concat([train_data, test_data], dim=0)
        print(f"len all_data-{scene}-seq1.t", len(all_data))

        gateway_ind = int(scene[1:]) - 1
        data_gateway_pos = torch.tensor(gateway_pos[gateway_ind]).view(-1) #[9]
        data_gateway_pos = data_gateway_pos.unsqueeze(0).unsqueeze(0).repeat(all_data.shape[0], 1, 1)

        allset = torch.cat([data_gateway_pos, all_data], dim=-1)
        train_len = int(len(allset) * 0.4)
        trainset = allset[:train_len]
        testset = allset[train_len:]

        print(f"len trainset", len(trainset))
        print(f"len testset", len(testset))

        torch.save(trainset, f"{savepath}train_data-{scene}-40-60-seq1.pt")
        torch.save(testset, f"{savepath}test_data-{scene}-40-60-seq1.pt")
