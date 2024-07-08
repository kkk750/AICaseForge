import logging
import os
from logging.handlers import RotatingFileHandler
from common.config_parse import ConfigParse
from common.file_path import FilePath


class LogConfig:
    LOG_LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }

    def __init__(self):
        self.config_parser = ConfigParse(FilePath.config_file_path)
        self.log_file_path = os.getenv('LOG_FILE_PATH', self.config_parser.get_logfile_path())
        self.level = self.LOG_LEVELS.get(self.config_parser.get_logfile_level().upper(), logging.INFO)
        self.max_bytes = self.config_parser.get_logfile_max_bytes()
        self.backup_count = self.config_parser.get_logfile_backup_count()
        self._configure_logging()

    def _configure_logging(self):
        # 确保日志文件路径存在
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)
        # 创建一个日志格式器
        log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # 获取root logger并设置级别
        root_logger = logging.getLogger()
        root_logger.setLevel(self.level)
        # 如果root logger没有handlers，则添加它们
        if not root_logger.handlers:
            if self.config_parser.get_logfile_enable():
                # 创建一个循环文件处理器，设置文件大小和备份数量
                file_handler = RotatingFileHandler(self.log_file_path, maxBytes=self.max_bytes,
                                                   backupCount=self.backup_count)
                file_handler.setFormatter(log_formatter)
                root_logger.addHandler(file_handler)
            # 创建一个控制台处理器并设置格式
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_formatter)
            root_logger.addHandler(console_handler)