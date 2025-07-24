# Git与HuggingFace Transformers学习指南

## Git基础操作

### 1. 设置远程仓库URL
$ git remote set-url origin https://github.com/thomasWts/HuggingFace-TransFormers-Learning.git
$ git remote -v

### 2. 常规工作流程
$ git status
$ git add .
$ git commit -m "描述性提交信息"
$ git push -u origin master

## 身份验证问题解决
# 403错误解决方案：
$ git remote set-url origin git@github.com:thomasWts/HuggingFace-TransFormers-Learning.git
$ ssh -T git@github.com

## HuggingFace Transformers环境配置
# 安装依赖
$ python -m venv transformers-env
$ source transformers-env/bin/activate  # Linux/macOS
$ .\transformers-env\Scripts\activate   # Windows
$ pip install transformers datasets torch

# 验证安装
from transformers import pipeline
classifier = pipeline("sentiment-analysis")
result = classifier("HuggingFace Transformers is amazing!")
print(result)

## 首次使用指南
### 1. 克隆仓库
$ git clone git@github.com:thomasWts/HuggingFace-TransFormers-Learning.git
$ cd HuggingFace-TransFormers-Learning

### 2. 设置用户信息
$ git config user.name "您的姓名"
$ git config user.email "您的邮箱"

### 3. 开发工作流
$ git checkout -b feature/your-feature
$ # 进行修改...
$ git add .
$ git commit -m "feat: 添加新功能"
$ git push -u origin feature/your-feature

### HuggingFace资源
- 文档: https://huggingface.co/docs/transformers
- 模型: https://huggingface.co/models
- 数据集: https://huggingface.co/datasets
- 教程: https://huggingface.co/learn