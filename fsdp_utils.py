
def get_train_fsdp_config():

    config = {
      "fsdp_config": {
        "sharding_strategy": "fully_shard",
        "num_shards": 1,
        "shard_optimizer": True,
        "cpu_offload": False
      }
    }
    return config