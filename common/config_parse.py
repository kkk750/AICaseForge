import configparser


class ConfigParse:
    # 以下是llm相关配置
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file,encoding='utf-8')

    def get_api_key(self) -> str:
        return self.config.get('api_key', 'API_KEY')

    def get_api_common_base(self) -> str:
        return self.config.get('api_key', 'API_COMMON_BASE')

    # 以下是向量数据库的配置
    def get_vearch_master_server(self) -> str:
        return self.config.get('vearch', 'master_server')

    def get_vearch_router_server(self) -> str:
        return self.config.get('vearch', 'router_server')

    # 以下是此项目相关配置
    def get_logfile_level(self) -> str:
        return self.config.get('logging', 'log_level')

    def get_logfile_enable(self) -> bool:
        return self.config.getboolean('logging', 'enable_file_logging')

    def get_logfile_path(self) -> str:
        return self.config.get('logging', 'log_file_path')

    def get_logfile_max_bytes(self) -> int:
        return self.config.getint('logging', 'log_file_max_bytes')

    def get_logfile_backup_count(self) -> int:
        return self.config.getint('logging', 'log_file_backup_count')
