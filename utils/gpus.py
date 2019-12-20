import os


def set_gpus(config):
    gpus = config.devices.gpus
    if len(gpus) == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(str(gpus[0]))
    elif len(gpus) > 1:
        for gpu in gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(str(gpu))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""