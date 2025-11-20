
dataset_builders = {}
def register_dataset(name):
    def decorator(cls):
        dataset_builders[name] = cls
        return cls
    return decorator