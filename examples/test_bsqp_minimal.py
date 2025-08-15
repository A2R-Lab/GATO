import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

from bsqp.interface import BSQP

import torch
print(f"torch version cuda: {torch.version.cuda}")
print(f"cudnn version: {torch.backends.cudnn.version()}")
print(f"torch file: {torch.__file__}")

if __name__ == "__main__":
    print("Instantiating BSQP...")
    solver = BSQP(model_path="indy7-mpc/description/indy7.urdf", batch_size=1, N=32, dt=0.01)
    print("BSQP instance created successfully.")
