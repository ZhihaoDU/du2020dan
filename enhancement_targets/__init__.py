def get_target(target_name):
    import importlib
    module = importlib.import_module("enhancement_targets.%s" % target_name)
    assert module
    calc_targets = getattr(module, "calc_targets")
    return calc_targets
