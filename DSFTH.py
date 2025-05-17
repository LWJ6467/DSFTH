


#from utils.tools import *
from utils.tools1 import *
#from network import *
from TransformerModel.modeling import VisionTransformer, VIT_CONFIGS
import argparse
import os
import random
import torch
import torch.optim as optim
import time
import numpy as np
from scipy.linalg import hadamard
torch.multiprocessing.set_sharing_strategy('file_system')
from TransformerModel.dfformer import dfformer_s18

torch.multiprocessing.set_sharing_strategy('file_system')
def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_random_seed(42, deterministic=True)

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy('file_system')


def get_config(dataset):
    config = {
        "lambda": 0.0001,"alpha": 1,
        "optimizer": {"type": optim.AdamW, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -5}},
        "info": "[CSQ]",
        "shuff": "shuff",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 32,
        "net_name":dfformer_s18, "net_print": "dfformer_s18",
        "net": dfformer_s18,
        #"net": ResNet,
        "dataset": dataset,
        #"dataset": "imagenet",        
        #"dataset": "coco",            
        #"dataset": "nuswide",         
        # "dataset": "cifar10",         
        "epoch": 100,
        "Init_epoch": 10,
        "test_map": 10,
        "save_path": "1",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [16,32,64],
    }



    config = config_dataset(config)
    return config


class CSQLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(CSQLoss, self).__init__()
        self.is_single_label = config["dataset"] not in {"nuswide", "nuswide_21_m", "coco"}
        self.hash_targets = self.get_hash_targets(config["n_class"], bit).to(config["device"])
        self.multi_label_random_center = torch.randint(2, (bit,)).float().to(config["device"])
        self.criterion = torch.nn.BCELoss().to(config["device"])

        # Initialize self.Y to store the label matrix (for likelihood_loss calculation)
        self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])  # Training hash codes
        self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

    def forward(self, u, y, ind, config):
        u = u.tanh()
        hash_center = self.label2center(y)
        center_loss = self.criterion(0.5 * (u + 1), 0.5 * (hash_center + 1))

        Q_loss = (u.abs() - 1).pow(2).mean()


        # 更新训练样本的哈希码（U）和标签（Y）
        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        # 计算likelihood_loss
        # 计算哈希码和标签之间的相似性
        s = (y @ self.Y.t() > 0).float()  # 计算标签间的相似性，s为二进制矩阵
        inner_product = u @ self.U.t() * 0.5  # 计算哈希码和训练样本之间的内积

        # likelihood_loss的计算
        likelihood_loss = (1 + (-(inner_product.abs())).exp()).log() + inner_product.clamp(min=0) - s * inner_product

        # 计算平均likelihood_loss
        likelihood_loss = likelihood_loss.mean()

        return center_loss + config["lambda"] * Q_loss + config["alpha"] * likelihood_loss
        #return center_loss + config["lambda"] * Q_loss

    def label2center(self, y):
        if self.is_single_label:
            hash_center = self.hash_targets[y.argmax(axis=1)]
        else:
            # to get sign no need to use mean, use sum here
            center_sum = y @ self.hash_targets
            random_center = self.multi_label_random_center.repeat(center_sum.shape[0], 1)
            center_sum[center_sum == 0] = random_center[center_sum == 0]
            hash_center = 2 * (center_sum > 0).float() - 1
        return hash_center

    # use algorithm 1 to generate hash centers
    def get_hash_targets(self, n_class, bit):
        H_K = hadamard(bit)
        H_2K = np.concatenate((H_K, -H_K), 0)
        hash_targets = torch.from_numpy(H_2K[:n_class]).float()

        if H_2K.shape[0] < n_class:
            hash_targets.resize_(n_class, bit)
            for k in range(20):
                for index in range(H_2K.shape[0], n_class):
                    ones = torch.ones(bit)
                    # Bernouli distribution
                    sa = random.sample(list(range(bit)), bit // 2)
                    ones[sa] = -1
                    hash_targets[index] = ones
                # to find average/min  pairwise distance
                c = []
                for i in range(n_class):
                    for j in range(n_class):
                        if i < j:
                            TF = sum(hash_targets[i] != hash_targets[j])
                            c.append(TF)
                c = np.array(c)

                # choose min(c) in the range of K/4 to K/3
                # see in https://github.com/yuanli2333/Hadamard-Matrix-for-hashing/issues/1
                # but it is hard when bit is  small
                if c.min() > bit / 4 and c.mean() >= bit / 2:
                    print(c.min(), c.mean())
                    break
        return hash_targets



def evaluate_inference_speed(model, test_loader, device):
    model.eval()  # 设置为评估模式
    total_time = 0
    total_samples = 0
    
    with torch.no_grad():  # 禁止梯度计算，节省内存
        for images, labels, _ in test_loader:
            images = images.to(device)
            
            start_time = time.time()  # 记录开始时间
            outputs = model(images)  # 执行推理
            end_time = time.time()  # 记录结束时间
            
            total_time += end_time - start_time  # 累加时间
            total_samples += images.size(0)  # 累加样本数量
    
    # 计算每个样本的平均推理时间
    avg_inference_time = total_time / total_samples
    # 计算吞吐量（每秒处理的样本数）
    throughput = total_samples / total_time
    
    return avg_inference_time, throughput

def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train

    net = config["net"](bit).to(device)
    

    optimizer = config["optimizer"]["type"](net.parameters(), **(config["optimizer"]["optim_params"]))

    criterion = CSQLoss(config, bit)

    Best_mAP = 0

    map_idx = [i * 10 for i in range(1, 16)]
    loss_idx = [i+1 for i in range(config["epoch"])]
    map_rec = []
    los_rec = []
    time_rec = []

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")
        

        start_time = time.time()

        net.train()



        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            u = net(image)

            loss = criterion(u, label.float(), ind, config)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))



        ###
        # 获取当前 GPU 内存使用情况
        allocated_memory = torch.cuda.memory_allocated(device) / 1024**2  # 单位 MB
        cached_memory = torch.cuda.memory_reserved(device) / 1024**2  # 单位 MB
        print(f"Allocated memory: {allocated_memory:.2f} MB")
        print(f"Cached memory: {cached_memory:.2f} MB")
        ###


        finish_time = time.time()
        cos_time = finish_time -start_time
        print("time cost: %.3f" % (finish_time - start_time))
        time_rec = time_rec + [cos_time]



        ###
        # 推理速度测量
        avg_inference_time, throughput = evaluate_inference_speed(net, test_loader, device)
        print(f"Average inference time: {avg_inference_time:.4f} seconds per sample")
        print(f"Throughput: {throughput:.2f} samples per second")
        ###




        # if (epoch == config["epoch"] - 1):
        time_data = {
            "index": loss_idx,
            "loss": time_rec
        }
        os.makedirs(os.path.dirname(config["time_path"]), exist_ok=True)
        with open(config["time_path"], 'w') as f:
            f.write(json.dumps(time_data))


        los_rec = los_rec + [train_loss]

        loss_data = {
            "index": loss_idx,
            "loss": los_rec
        }
        os.makedirs(os.path.dirname(config["loss_path"]), exist_ok=True)
        with open(config["loss_path"], 'w') as f:
            f.write(json.dumps(loss_data))
        # if :
        if ((epoch + 1) % config["test_map"] == 0)&((epoch + 1)>=config["Init_epoch"]):
            Best_mAP,mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)
            map_rec = map_rec + [mAP]
            map_data = {
                "index": map_idx,
                "mAP": map_rec
            }
            os.makedirs(os.path.dirname(config["map_path"]), exist_ok=True)
            with open(config["map_path"], 'w') as f:
                f.write(json.dumps(map_data))

        # if (epoch + 1) % config["test_map"] == 0:
        #     Best_mAP = validate(config, Best_mAP, test_loader, dataset_loader, net, bit, epoch, num_dataset)



# if __name__ == "__main__":
#     config = get_config()
#     print(config)
#     for bit in config["bit_list"]:
#         config["pr_curve_path"] = f"log/likelihood_loss+CAFqCAbEPA+stghb+dfformer_s18.32.10.1e-5.a =1/CSQ_{config['dataset']}_{bit}.json"
#         train_val(config, bit)

if __name__ == "__main__":
    dataset="cifar10" #"imagenet","coco","cifar10-1","nuswide",,"coco","cifar10-1"
    config = get_config(dataset=dataset)
    print(config)
    config["bit_list"] = [64] #,3216,
    data_list = [dataset]#,"imagenet","coco","cifar10-1","nuswide",,"coco","cifar10-1"
    for dataset in data_list:
        config['dataset'] = dataset
        config = config_dataset(config)
        for bit in config["bit_list"]:
            config[
                "pr_curve_path"] = f"./save_{config['info']}/{config['net_name']}_{config['shuff']}/pr_{config['info']}/"
            config[
                "loss_path"] = f"./save_{config['info']}/{config['net_name']}_{config['shuff']}/loss_{config['info']}/{config['dataset']}_{bit}.json"
            config[
                "map_path"] = f"./save_{config['info']}/{config['net_name']}_{config['shuff']}/map_{config['info']}/{config['dataset']}_{bit}.json"
            config[
                "time_path"] = f"./save_{config['info']}/{config['net_name']}_{config['shuff']}/time_{config['info']}/{config['dataset']}_{bit}.json"
            config[
                "save_path"] = f"./save_{config['info']}/{config['net_name']}_{config['shuff']}/ckp_{config['info']}/"
            train_val(config, bit)