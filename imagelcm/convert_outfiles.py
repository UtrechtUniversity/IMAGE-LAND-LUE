"""Converts agrprod.out files."""

import numpy as np

import parameters as prm

def convert_crop_outfiles(crop_type):
    """
    Reads agr. dem. .OUT files and saves the data in numpy .npy format.

    Parameters
    ----------
    crop_type : str
                determines whether the demand files being read contain
                food crop, fodder or bioenergy crop data.
    """
    # determine whether input is food, grass or bioenergy crop demand
    crop_attrs = {"food": {"name": "AGRPRODC.OUT",
                           "n_crops": prm.NFC},
                  "grass": {"name": "AGRPRODP.OUT",
                            "n_crops": prm.NGST},
                  "bioenergy": {"name": "AGRPRODB.OUT",
                              "n_crops": prm.NBC}}.get(crop_type)

    # read and decode data
    with open(f"data\\{crop_attrs['name']}", 'rb') as file:
        data_read = file.read()
    data_str = data_read.decode()

    # print(data_str[:150])

    # split data into time blocks (the data for each year)
    time_blocks = []
    last_ind = 0
    year = 1970
    for ind in range(len(data_str)-4):
        # split string when there is a year followed by a non-alphanum character (prob. line break)
        if data_str[ind:(ind+4)]==str(year) and not data_str[ind:(ind+5)].isalnum():
            time_blocks.append(data_str[last_ind:ind])
            last_ind = ind
            year += 1
            # if last year, then append the rest of the data to the time_blocks list
            if year==2100:
                time_blocks.append(data_str[last_ind:])

    # fill in a 3D np array containing the crop demand data, of shape (n_year, n_crops, n_regions)
    crop_demands = np.zeros((131, crop_attrs['n_crops'], 26))
    for time_ind, tb in enumerate(time_blocks[1:]):
        current_block = tb.split("\r\n")[1:-1] # will have length n_crops
        for crop in range(crop_attrs['n_crops']):
            block_crop = current_block[crop].split("   ")[1:-1] # will have length n_regions
            crop_demands[time_ind, crop, :] = np.asarray(block_crop).astype(np.float32)

    # save array as .npy binary file
    np.save(f"data\\{crop_type}_crop_demands", crop_demands)

def convert_management_outfiles(manage_type):
    """
    Reads management data .OUT files; saves the data in numpy .npy format.

    Parameters
    ----------
    manage_type : str
                  determines whether the demand files being read contain
                  management factors, grazing intensities or harvest
                  fractions.
    """
    # determine whether input is food, grass or bioenergy crop demand
    crop_attrs = {"MF": {"name": "MF.OUT",
                         "n_crops": prm.NGFBC},
                  "GI": {"name": "GRAZINTENS.OUT",
                         "n_crops": prm.NGS},
                  "FH": {"name": "FRHARVCOMB.OUT",
                         "n_crops": prm.NFBFC}}.get(manage_type)

    # read and decode data
    with open(f"data\\{crop_attrs['name']}", 'rb') as file:
        data_read = file.read()
    data_str = data_read.decode()

    # print(data_str[:150])
    # split data into time blocks (the data for each year)
    time_blocks = []
    last_ind = 0
    year = 1970
    for ind in range(len(data_str)-4):
        # split string when there is a year followed by a non-alphanum character (prob. line break)
        if data_str[ind:(ind+4)]==str(year) and not data_str[ind:(ind+5)].isalnum():
            time_blocks.append(data_str[last_ind:ind])
            last_ind = ind
            year += 1
            # if last year, then append the rest of the data to the time_blocks list
            if year==2100:
                time_blocks.append(data_str[last_ind:])

    # fill in a 3D np array containing the crop demand data, of shape (n_year, n_crops, n_regions)
    management_values = np.zeros((131, crop_attrs['n_crops'], 26))
    for time_ind, tb in enumerate(time_blocks[1:]):
        current_block = tb.split("\r\n")[1:] # will have length n_crops
        for crop in range(crop_attrs['n_crops']):
            block_crop = current_block[crop].split("   ")[1:] # will have length n_regions
            management_values[time_ind, crop, :] = np.asarray(block_crop).astype(np.float32)

    # save array as .npy binary file
    np.save(f"data\\{manage_type}", management_values)
