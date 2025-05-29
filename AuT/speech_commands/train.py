from lib.utils import ConfigDict

cfg = ConfigDict()
cfg.tf = ConfigDict()
cfg.tf.embed_dim = 3560

print(f'cfg.tf.embed_dim={cfg['tf'].embed_dim}')