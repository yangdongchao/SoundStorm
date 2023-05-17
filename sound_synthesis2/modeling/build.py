from sound_synthesis2.utils.misc import instantiate_from_config
from hydra.utils import instantiate

def build_model(config, args=None):
    return instantiate(config['model'])
