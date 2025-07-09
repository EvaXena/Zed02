hey 本库基于hls4ml开发 旨在fpga上部署midas_small与sml 实现水下场景的使用


0707 v1
model文件夹中是 模型原型 
现已完成 Midas_samll 的 tensorflow复现
=======================================
Total params: 12035041 (45.91 MB)
Trainable params: 11960929 (45.63 MB)
Non-trainable params: 74112 (289.50 KB)
=======================================

添加environment.yaml 保证环境稳定 在宿主机装有CUDA时 记得unset LD_LIBRARY_PATH


0709 v4
现已完成 服务器训练全流程
训练了基于NYU深度数据集的midas_small
模型保存在服务器的result文件夹中
最新的conda环境在environment1.txt

run_midas v11  最初dataloader版本 导致内存爆炸
          v12  可以在笔记本上训练的小pipeline
          v13  服务器特化版本 使用python实现并行 GPU利用率低
          v14  服务器优化完成版本 高并行度光速训练 速度是v2的30倍

dataloader 中bv11 v12分别匹配 run的 v11 v12

要使用这个库进行基于NYU-v2的tensorflow训练
    0.cd 到工程根目录
    1.conda env create -f environment1.txt
    2.进入kerasv20
    3.下载NYU数据集 创建dataset文件夹 放到该文件夹中
    4.运行run_midas_v14(服务器) 或者 run_midas_v12（小规模）

下一步做 1.剪枝   2.使用Qkeras进行替换 实现量化 3.送入hls4ml进行综合
完成后 同理移植SML 进行仿真 查看效果与资源 
再下一步 实现完整的 GA+VIO+SML流程