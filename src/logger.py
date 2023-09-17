import logging
import os
import sys
from datetime import datetime

LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(log_path,exist_ok=True)
log_file_path=os.path.join(log_path,LOG_FILE)
logging.basicConfig(filename=log_file_path,level=logging.INFO)


