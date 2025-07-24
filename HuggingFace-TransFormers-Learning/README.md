# git 与 HuggingFace TransFormer 学习

# git 创建文件

## 替换远程仓库URL指令

```bash
git remote set-url origin https://github.com/thomasWts/HuggingFace-TransFormers-Learning.git
```

## 验证是否修改成功

```bash
git remote -v
```

输出

```bash
origin  https://github.com/thomasWts/HuggingFace-TransFormers-Learning.git (fetch)
origin  https://github.com/thomasWts/HuggingFace-TransFormers-Learning.git (push)
```

```bash
git status
git add .
git commit -m "A"
```

登陆github帐号

```bash
ssh -T git@github.com
```

发现现在直接上传会报错

```bash
fatal: unable to access 'https://github.com/thomasWts/HuggingFace-TransFormers-Learning.git/': The requested URL returned error: 403
```

设置远程地址为SSH格式

```bash
git remote set-url origin git@github.com:thomasWts/HuggingFace-TransFormers-Learning.git
```

```bash
git push -u origin master
```
成功上传