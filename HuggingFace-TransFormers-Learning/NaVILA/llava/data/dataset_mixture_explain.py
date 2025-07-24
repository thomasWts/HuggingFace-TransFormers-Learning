 #定义一个数据集的数据结构（用 @dataclass 方式）

@dataclass
class Dataset:
    # 数据集名称（如"r2r"、"scanqa"）
    dataset_name: str

    # 数据集类型，比如"torch"（PyTorch格式）、"vlnce"、"envdrop"等
    dataset_type: str = field(default="torch")

    # 训练数据路径（一般是 annotation json）
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})

    # 元数据路径，仅在 WebDataset 格式中可能使用
    meta_path: str = field(default=None, metadata={"help": "Path to the meta data for webdataset."})

    # 图像或视频帧的路径
    image_path: str = field(default=None, metadata={"help": "Path to the training image data."})

    # 如果是重新生成 caption，这里可以指定哪个目录下的 caption
    caption_choice: str = field(default=None, metadata={"help": "Path to the caption directory for recaption."})

    # 对该数据集的描述，包括来源、标注方式、用途、大小等
    description: str = field(
        default=None,
        metadata={
            "help": "Detailed description of where the data is from, how it is labelled, intended use case and the size of the dataset."
        },
    )

    # 测试脚本路径（目前未使用，仅占位）
    test_script: str = (None,)

    # 数据集维护者信息（目前未使用，仅占位）
    maintainer: str = (None,)

    # 以下是冗余定义，`caption_choice` 再次定义了（建议删除重复项）
    caption_choice: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})

    # 第二个caption目录路径（可能用于对比不同caption）
    caption_choice_2: str = field(default=None, metadata={"help": "Path to the captions for webdataset."})

    # 数据集起始索引（可用于选择子集）
    start_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})

    # 数据集结束索引
    end_idx: float = field(default=-1, metadata={"help": "Start index of the dataset."})

#✅ 定义一个空的全局字典用于保存所有注册的数据集

DATASETS_LEGACY = {}

#✅ 用于添加数据集的函数

def add_dataset(dataset):
    # 如果数据集名称已存在，发出警告
    if dataset.dataset_name in DATASETS_LEGACY:
        warnings.warn(f"{dataset.dataset_name} already existed in DATASETS. Make sure the name is unique.")
    
    # 数据集名不能包含加号（“+”）
    assert "+" not in dataset.dataset_name, "Dataset name cannot include symbol '+'."
    
    # 把数据集添加到全局字典中
    DATASETS_LEGACY.update({dataset.dataset_name: dataset})

#✅ 注册所有数据集的函数

def register_datasets_mixtures():

#以下每一段都是注册一个数据集：

    video_chatgpt = Dataset(
        dataset_name="video_chatgpt",
        dataset_type="torch",
        data_path="/PATH_TO_DATA/annotations.json",
        image_path="/PATH_TO_DATA/videos",
    )
    add_dataset(video_chatgpt)

#上面表示添加一个名为 video_chatgpt 的数据集，它是 PyTorch 格式，注释文件为 annotations.json，图像为 /PATH_TO_DATA/videos。

#以下是其它数据集：

    sharegpt_video = Dataset(
        dataset_name="sharegpt_video",
        dataset_type="torch",
        data_path="/PATH_TO_DATA/annotations.json",
        image_path="/PATH_TO_DATA/videos",
    )
    add_dataset(sharegpt_video)

    sharegpt4v_sft = Dataset(
        dataset_name="sharegpt4v_sft",
        dataset_type="torch",
        data_path="/PATH_TO_DATA/annotations.json",
        image_path="/PATH_TO_DATA/videos",
    )
    add_dataset(sharegpt4v_sft)

#以下是 VLN 相关的数据集（视觉语言导航任务）

    envdrop = Dataset(
        dataset_name="envdrop",
        dataset_type="envdrop",
        data_path="/PATH_TO_DATA/NaVILA-Dataset/EnvDrop/annotations.json",
        image_path="/PATH_TO_DATA/NaVILA-Dataset/EnvDrop/videos",
        description="VLN_CE Envdrop.",
    )
    add_dataset(envdrop)

    scanqa = Dataset(
        dataset_name="scanqa",
        dataset_type="torch",
        data_path="/PATH_TO_DATA/NaVILA-Dataset/ScanQA/annotations/ScanQA_v1.0_train_reformat.json",
        image_path="/PATH_TO_DATA/NaVILA-Dataset/ScanQA/videos",
        description="ScanQA training set.",
    )
    add_dataset(scanqa)

    r2r = Dataset(
        dataset_name="r2r",
        dataset_type="vlnce",
        data_path="/PATH_TO_DATA/NaVILA-Dataset/R2R/annotations.json",
        image_path="/PATH_TO_DATA/NaVILA-Dataset/R2R/train",
        description="350K VLN-CE R2R data. (augmented with duplicate samples)",
    )
    add_dataset(r2r)

    rxr = Dataset(
        dataset_name="rxr",
        dataset_type="vlnce",
        data_path="/PATH_TO_DATA/NaVILA-Dataset/RxR/annotations.json",
        image_path="/PATH_TO_DATA/NaVILA-Dataset/RxR/train",
        description="400K RxR data. (augmented with duplicate stops only - 5x)",
    )
    add_dataset(rxr)

    human = Dataset(
        dataset_name="human",
        dataset_type="vlnce",
        data_path="/PATH_TO_DATA/NaVILA-Dataset/Human/annotations.json",
        image_path="/PATH_TO_DATA/NaVILA-Dataset/Human/raw_frames/",
        description="560K Real augmented, no direction is included. (augmented with duplicate stops only - 5x)",
    )
    add_dataset(human)