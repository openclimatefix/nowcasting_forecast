""" Simple model to take NWP irradence and make solar """
import numpy as np
from nowcasting_dataset.dataset.batch import Batch


def nwp_irradence_simple(batch: Batch) -> np.array:

    nwp = batch.nwp

    # take solar irradence
    print(nwp)
    print(nwp.channels)
    nwp = nwp.sel(channels=["dlwrf"])

    # take mean across all dims excpet mean
    print(nwp)
    irradence_mean = nwp.mean(axis=[1, 2, 3])

    # scale irradence to roughly mw
    irradence_mean = irradence_mean / 100

    return irradence_mean
