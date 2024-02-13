import torch


def get_list_of_devices(nb_devices, acc_gen_sampling):
    """

    :param nb_devices: number of gpu devices in use
    :param acc_gen_sampling:  dedicate one gpu for data loader
    :return:
    """
    if nb_devices > 0:
        all_devices = [torch.device(f'cuda:{i}') for i in range(int(nb_devices))]
    else:
        all_devices = [torch.device('cpu')]

    if acc_gen_sampling:
        # use the last device for sampling
        assert len(all_devices) >= 2
        dataloader_device = all_devices[-1]
        generator_devices = all_devices[:-1]
    else:
        dataloader_device = torch.device('cpu')
        generator_devices = all_devices

    return dataloader_device, generator_devices
