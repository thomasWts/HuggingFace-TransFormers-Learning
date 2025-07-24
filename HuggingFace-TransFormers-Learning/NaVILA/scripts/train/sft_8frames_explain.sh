OUTPUT="./checkpoints/navila-8b-8f-sft"

    OUTPUT 是模型训练的输出目录，也就是保存训练中间结果和最终模型的文件夹。

    navila-8b-8f-sft 是你自定义的模型名称。

torchrun --nnodes=$n_node --nproc_per_node=$GPUS_PER_NODE --master_port=$MASTER_PORT \
    --master_addr $MASTER_ADDR --node_rank=$CURRENT_RANK \

这一段用的是 PyTorch 的分布式训练工具 torchrun，意思是：

    --nnodes=$n_node：总共使用的节点数量（比如你有多台服务器协同训练，这里就要大于1）。

    --nproc_per_node=$GPUS_PER_NODE：每个节点使用多少张 GPU。

    --master_port=$MASTER_PORT：分布式训练中的主节点端口号。

    --master_addr=$MASTER_ADDR：主节点的 IP 地址。

    --node_rank=$CURRENT_RANK：当前节点在所有节点中的编号（从 0 开始）。

⚠️ 你需要在训练前自己设置这些变量。
✅ 第二部分：模型训练主命令

llava/train/train_mem.py

    这是你要执行的 Python 脚本文件，位于 llava/train/ 路径下，文件名是 train_mem.py。

    这是一个用于训练 Navila 模型（包含视觉和语言多模态模型）的脚本。

✅ 第三部分：模型配置相关参数

--longvila_sampler True

    使用 longvila_sampler，控制如何抽样多模态输入（长文本 + 图像 + 视频帧）。

--deepspeed ./scripts/zero3.json

    使用 DeepSpeed 来加速训练，这里用的是 Zero Stage 3（节省显存的一种策略）。

    zero3.json 是配置文件，定义了具体策略。

--model_name_or_path a8cheng/navila-siglip-llama3-8b-v1.5-pretrain

    使用预训练好的模型作为初始化权重。

    来自 Hugging Face 的模型地址。

--version llama_3

    指定使用的是 LLaMA3 版本的语言模型。

--seed 10

    设置随机种子，确保训练过程可复现（相同的结果）。