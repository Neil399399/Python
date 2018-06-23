import logging

# 基礎設定
logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)-3s %(asctime)s %(name)-3s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers = [logging.FileHandler('my.log', 'a', 'utf-8'),])
 
# 定義 handler 輸出 sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
# 設定輸出格式
formatter = logging.Formatter('[%(levelname)s] %(asctime)-s %(name)-s : %(message)s','%Y-%m-%d %H:%M:%S')
# handler 設定輸出格式
console.setFormatter(formatter)
# 加入 hander 到 root logger
logging.getLogger('').addHandler(console)

TensorFlow_log = logging.getLogger('Tensorflow')
TFRecord_log = logging.getLogger('TFRecord')
 
