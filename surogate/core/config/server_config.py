from dataclasses import dataclass
from typing import Optional
from surogate.utils.dict import DictDefault
from surogate.utils.logger import get_logger

logger = get_logger()

@dataclass
class ServerConfig:
    host: str = '0.0.0.0'
    port: int = 8888
    workers: int = 1
    log_level: str = 'info'
    database_url: str = 'sqlite+aiosqlite:///surogate.db'

    def __init__(self, cfg: DictDefault):
        self.host = cfg.get('host', self.host)
        self.port = cfg.get('port', self.port)
        self.workers = cfg.get('workers', self.workers)
        self.log_level = cfg.get('log_level', self.log_level)
        self.database_url = cfg.get('database_url', self.database_url)

    def __post_init__(self):
        pass