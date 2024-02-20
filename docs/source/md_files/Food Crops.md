# Food Crops

IMAGE-LAND has a total of 38 food crop types; in the code, these are generally labelled from 0 to 37. Crop 0 is <a href='grass.html'>grass</a>, crops 1-16 are the rain-fed food crops, listed below, and crops 17-21 being <a href='Biofuel Crops.html'>biofuel crops</a>. Crops 22-37 are irrigated food crops - these are the same crops as those listed below, but are associated with different productivities, due to their being irrigated. The 16 food crops are:
1. Wheat                                    
2. Rice                                     
3. Maize                                    
4. Tropical cereals                         
5. Other temperate cereals                  
6. Pulses                                   
7. Soybeans                                 
8. Temperate oil crops                      
9. Tropical oil crops                        
10. Temperate roots & tubers                 
11. Tropical roots & tubers                  
12. Sugar crops                              
13. Oil & palm fruit                         
14. Vegetables & fruits                      
15. Other non-food & luxury & spices         
16. Plant based fibres.

## Demand
The demands for these food crops are output by IMAGE's ADM module, in AGRPRODC.OUT. This file is converted by IMAGE-LAND-LUE's read_crop_outfiles() function into 3 dimensional np arrays, and saved in the <a href='https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html#module-numpy.lib.format'>npy</a> format which can then be read in by the rest of IMAGE-LAND-LUE.

AGRPRODC.OUT and the npy file store both the demand for food crops and the demand for <a href='grass.html'>grass</a>, which is sometimes referred to in the code as the zeroth food crop.

## Rain-fed and irrigated crops

## Allocation
To read about how land is allocated to growing food crops - as well as grass - see <a href='IMAGE-LAND Allocation and Formulae.html'>IMAGE-LAND Allocation and Formulae</a>.