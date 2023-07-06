from soundstorm.s2.utils.misc import instantiate_from_config


def build_model(config):
    return instantiate_from_config(config['model'])
