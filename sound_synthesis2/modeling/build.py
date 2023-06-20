from hydra.utils import instantiate


def build_model(config, args=None):
    return instantiate(config['model'])
