# Strong-FastestDet

This paper was submitted to the Journal of Real-Time Image Processing.

A: How to Train？
1. Configure train: "train.txt" and val: "C:val.txt" under Strong-FastestDeT\configs\ Co.yaml
2. Configure names under Strong-FastestDeT\configs\ Cok.yaml: "configs/ Cok.names ", and store the categories under cok.names in the same-level directory.
3. run train.py

B: ONNX
1.Install the onnx environment
pip install cmake==3.30.2
pip install onnxoptimizer onnxruntime protobuf rich!=12.1.0 onnxslim
pip install onnx-simplifier -i http://mirrors.aliyun.com/pypi/simple 
python3 runtime.py

C.Environment
1. Create an independent virtual environment and activate it.
2. pip install -r requirements.txt

D.more see : https://github.com/dog-qiuqiu/FastestDet

Acknowledgments ：
@misc
 {=FastestDet,
 title={FastestDet: Ultra lightweight anchor-free real-time object detection algorithm.},
 author={xuehao.ma},
 howpublished = {\url{https://github.com/dog-qiuqiu/FastestDet}},
 year={2022}
}


