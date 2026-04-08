
import requests
import sys
import subprocess
import os
import logging
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import config
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
import os
import signal
from pathlib import Path
logger = logging.getLogger(__name__)

PID_FILE = Path(config.QWEN_PID_FILE)

def terminate():
    if not PID_FILE.exists():
        logger.warning("PID 文件不存在，无需终止。")
        return
    pid = int(PID_FILE.read_text().strip())
    try:
        os.kill(pid, signal.SIGTERM)
        logger.info(f"已向模型进程 {pid} 发送 SIGTERM。")
    except ProcessLookupError:
        logger.warning(f"进程 {pid} 已不存在。")
    finally:
        PID_FILE.unlink(missing_ok=True)

if __name__ == "__main__":
    terminate()