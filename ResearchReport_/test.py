# coding:utf-8
# ocr识别
import os
import jieba
from LAC import LAC
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from paddleocr import PaddleOCR

ocr = PaddleOCR(det_model_dir='./PaddleOCR/output/ch_db_mv3_inference/inference',use_angle_cls=True,use_gpu=False )
data_path = "D:/Github/MachineLearning/learngit/ResearchReport_/ResearchReport/test/1.jpeg"
result = ocr.ocr(data_path, cls=True)
ch = ""
for i in result:
    print(i)
    ch += i[-1][0]
# l = jieba.cut(result[0][-1][0])
# print(result)
lac = LAC(mode = "lac")
lac_result = lac.run(ch)
print(lac_result)
