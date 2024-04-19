# 快速开始
## step1：下载模型(自动加载可跳过)  
`falcon模型下载地址 https://huggingface.co/tiiuae/falcon-7b`  

### 模型文件存放位置  
`./model-7b`  

## step2：安装依赖环境 
### 创建conda环境
`conda create -n FalconEnvi python=3.8.10`
### 激活conda环境
`conda activate FalconEnvi `
### 安装依赖包
`pip install -r requirements.txt `  
  
## step3：运行代码文件  
### 本地加载模型 运行程序
`python annotate_raw_review.py ./model-7b `

### 自动加载模型 运行程序
`python annotate_raw_review.py tiiuae/falcon-7b`  
  
### 得到标注文件   
`./raw_data_output_ten-shot.txt`  
