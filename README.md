### 快速开始
step1：下载模型  
`falcon模型下载地址 https://huggingface.co/tiiuae/falcon-7b`  

模型文件存放位置  
`./model-7b`  

step2：安装依赖环境  
`conda create -n FalconEnvi python=3.8.10`  
`conda activate FalconEnvi `  
`pip install -r requirements.txt `  
  
step3：运行代码文件  
`python annotate_raw_review.py  `  
  
得到标注文件   
`./raw_data_output_ten-shot.txt`  
