import sys
import threading
import time
import os

def trace_calls(frame, event, arg):
    if event == 'call':
        func_name = frame.f_code.co_name
        file_name = frame.f_code.co_filename
        if "mardpg_uav" in file_name or "train.py" in file_name:
            print(f"CALL: {func_name} in {file_name}:{frame.f_lineno}", flush=True)
    return trace_calls

sys.settrace(trace_calls)

import sys
sys.path.append(os.path.abspath('mardpg-uav'))

import runpy
sys.argv = ['train.py', '--device', 'cpu', '--no-wandb']
runpy.run_path('mardpg-uav/scripts/train.py', run_name='__main__')
