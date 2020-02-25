def get_model(gen_model_name, dis_model_name):
    import importlib
    module = importlib.import_module("models.%s" % gen_model_name)
    assert module
    Generator = getattr(module, "Generator")

    if dis_model_name is not None:
        module = importlib.import_module("models.%s" % dis_model_name)
        assert module
        Discriminator = getattr(module, "Discriminator")
    else:
        Discriminator = None

    return Generator, Discriminator
