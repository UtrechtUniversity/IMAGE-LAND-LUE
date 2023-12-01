"""
Runs np_test_module.py as a selfstanding version of IMAGE-LAND for a
single timestep, for testing purposes.
"""

import numpy as np

import np_test_module as npt


def main():
    """Main function."""
    n_regs = 2
    n_crops = 3 # excluding grass
    shape = (10, 20)

    npt.write_inputs(nr=n_regs)
    npt.np_first_reallocation(nr=n_regs)

    # load inputs for integration functions
    fracs_r1 = np.load(f"test_IO\\fractions_first_reallocation_{n_regs}_{n_crops}_{shape}.npy")
    regs = np.load(f"test_IO\\regions_{n_regs}_{n_crops}_{shape}.npy")
    suit_map = np.load(f"test_IO\\suitmap_{n_regs}_{n_crops}_{shape}.npy")
    demand_maps = np.load(f"test_IO\\demand_maps_{n_regs}_{n_crops}_{shape}.npy")
    g_area = np.load(f"test_IO\\g_area_{n_regs}_{n_crops}_{shape}.npy")
    pot_prod = np.load(f"test_IO\\potprod_{n_regs}_{n_crops}_{shape}.npy")
    is_cropland = np.load(f"test_IO\\is_cropland_{n_regs}_{n_crops}_{shape}.npy")
    demands = np.load(f"test_IO\\demands_{n_regs}_{n_crops}_{shape}.npy")

    yield_facs = pot_prod * np.stack([g_area for _ in range(n_crops+1)], axis=0)
    integrands = fracs_r1 * yield_facs

    # flatten arrays
    integrands = npt.flatten_rasters(integrands)
    regs_flat = npt.flatten_rasters(regs)
    suit_map_flat = npt.flatten_rasters(suit_map)

    # integrate
    _, demands_met, regional_prod = npt.integrate_r1_fractions(integrands, regs_flat,
                                                                         demand_maps,
                                                                         suit_map_flat)

    # get rid of fractions where demand has already been met
    fracs_r2 = npt.rework_reallocation(fracs_r1, demands_met, regs)

    # define/flatten inputs for integration_allocation
    sdpf = npt.flatten_rasters(pot_prod)
    fracs_r2_flat = npt.flatten_rasters(fracs_r2)
    yield_facs_flat = npt.flatten_rasters(yield_facs)
    demand_maps_flat = npt.flatten_rasters(demand_maps)
    is_cropland_flat = npt.flatten_rasters(is_cropland)

    # call integration allocation on existing cropland
    fracs_r3, reg_prod_updated = npt.integration_allocation(sdpf, yield_facs_flat, fracs_r2_flat, regs_flat,
                                                suit_map_flat, regional_prod, demands,
                                                demand_maps_flat, is_cropland_flat, shape)
    
    # if demand still not met, run integration-allocation again, this time to expand cropland
    if len(np.where(demands-reg_prod_updated>0)[0])>0:
        npt.integration_allocation(sdpf, yield_facs_flat, fracs_r2_flat, regs_flat,
                                                suit_map_flat, reg_prod_updated, demands,
                                                demand_maps_flat, is_cropland_flat, shape,
                                                expansion=True)

    # compute_sdp(nr=2)

if __name__=='__main__':
    main()
