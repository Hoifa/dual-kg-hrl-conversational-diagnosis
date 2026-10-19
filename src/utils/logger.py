import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# 定义日志保存目录
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def setup_logger(name="MedicalHRL", log_file="system.log", level=logging.INFO):
    """
    配置并返回一个全局 Logger 对象
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 防止重复添加 Handler (如果模块被多次导入)
    if not logger.handlers:
        # --- 格式器 ---
        # 格式: [时间] [级别] [文件名:行号] - 信息
        formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # --- Handler 1: 控制台输出 (Stdout) ---
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # --- Handler 2: 文件输出 (支持文件回滚) ---
        # maxBytes=10MB, backupCount=5 (保留最近5个文件)
        file_handler = RotatingFileHandler(
            os.path.join(LOG_DIR, log_file),
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# 创建全局单例 logger
# 其他文件直接 import 这个 logger 即可使用
logger = setup_logger()
