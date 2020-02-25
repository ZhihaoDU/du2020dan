def get_disc_loss(loss_name):
    import importlib
    loss = importlib.import_module("loss_functions.%s_loss" % loss_name)
    disc_g_loss = getattr(loss, "g_loss")
    disc_d_loss = getattr(loss, "d_loss")
    return disc_g_loss, disc_d_loss


def get_recon_loss(loss_name):
    import importlib
    loss = importlib.import_module("loss_functions.%s_loss" % loss_name)
    loss_function = getattr(loss, "loss_function")
    return loss_function