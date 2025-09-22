import random
import sys

from dvclive import Live

if len(sys.argv) != 2:
    sys.stderr.write("Arguments error. Usage:\n")
    sys.stderr.write(f"\tpython {sys.argv[0]} <epochs>\n")
    sys.exit(1)


with Live(save_dvc_exp=True) as live:
    epochs = int(sys.argv[1])
    for epoch in range(epochs):
        live.log_metric("train/accuracy", epoch + random.random())
        live.log_metric("train/loss", epochs - epoch - random.random())
        live.log_metric("val/accuracy", epoch + random.random())
        live.log_metric("val/loss", epochs - epoch - random.random())
        live.next_step()
