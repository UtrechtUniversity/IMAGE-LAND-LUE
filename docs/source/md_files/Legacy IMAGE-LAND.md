# Legacy IMAGE-LAND

This page offers an overview of the 3 main subroutines in the allocation and management of land for crops and grassland in the IMAGE-LAND sub-model. For an overview of IMAGE-LAND that does not delve into the individual FORTRAN files and synthesises the processes outlined here in a more general explanation and set of equations, please see <a href='IMAGE-LAND Allocation and Formulae.html'>IMAGE-LAND Allocation and Formulae</a>.

## LCM ama aeg

When reallocating existing cropland, IMAGE-LAND uses 3 quantities:

$$
    cf_1 = f_{0, t-1}\frac{d_{0, t}}{GI\times d_{0, t-1}} + \sum\limits_{c} f_{c, t-1}\frac{d_{c, t}}{d_{c, t-1}MF_c}, \tag{1}
$$
where $GI$ is the grazing intensity,

$$
    cf_2 = MAX\left[0, \frac{p_0\left(\frac{d_{0, t}}{GI} - d_{0, t-1}\right)}{M_0}\right] + \sum\limits_{c} MAX\left[0, \frac{p_0\left(\frac{d_{0, t}}{MF_c} - d_{0, t-1}\right)}{M_c}\right] \tag{2}
$$
and

$$
    cf_3 = f_{0, t-1} + \sum\limits_{c} f_{c, t-1}. \tag{3}
$$
Once these quantities have been obtained, the non-grass crops are looped through again, so that the fractions allocated to each of them can be computed, following

$$
    f_{c, t} = f_{c, t-1}MF_c\frac{d_{c, t}}{d_{c, t-1}}\frac{cf_3}{cf_1}. \tag{4}
$$
Intuitively, $cf_1$ can be understood as a measure of the management-factor-corrected fractional increase in total crop allocation. $cf_3$, by contrast, is simply the total cell fraction allocated to crops and grass in the previous timestep. Together, these are used to maintain normalisation when allocating the new fractions. Once this is complete, the program loops over all non-grass crops and completes the crop allocation; whatever fraction is not allocated after this is allocated to grass (crop 0).

$cf_2$ can be thought of as a measure of whether there is an increase in demand for any crop, and if there is, how much. In IMAGE-LAND, it is used to determine which checks for different edge cases are used: if $cf_2>0$, the program ensures that $cf_1>0$, $MF_c>0$ and $M_c>0$, whereas if $cf_2\leq0$ the program only checks that $cf_1$ and $MF_c$ are greater than 0. It is unclear what this adds to the code, and so $cf_2$ will be omitted from the new version of IMAGE-LAND.

## LCM hea anl

This is the subroutine called once ama aeg (the reallocation of existing cropland) has finished running. It loops through the grid in the order of suitability (from high to low) and checks the fractions allocated to each cell by ama aeg. As it loops through the cells, the subroutine keeps track of the cumulative potential yield for each crop, $Y_c$. It first checks the normalisation, then, for each cell, whether the allocation of the chosen fraction to that cell will surpass the demand for the corresponding crop. If not, then the fraction allocated remains the same. If so, however, the fraction is changed, following

$$
    f_{c, t} = MIN\left[\frac{d_{c, t} - Y_c}{p_c\times M_c\times A\times FH_c\times MF_c}, f_{c, t}\right]. \tag{5}
$$
For each cell, this action is completed for all non-grass crops (whether biofuel crops are included in this list depends on the options IMAGE is initialised with). The sum of the fractions allocated, $F_t$, is kept track of, and used to calculate the amount of the cell (re)allocated to grassland. However, if the demand for grass is already met, the fraction of the cell allocated to grassland is set to 0. As with the other crops, if the potential grass production of a given cell will cause the demand for grass to be surpassed, the fraction of the cell allocated to grassland is changed, following

$$
    f_{0, t} = MIN\left[\frac{d_0 - Y_0}{p_0\times M_0\times A\times GI}, f_{0, t}\right]. \tag{6}
$$

Naturally if IMAGE-LAND has been initialised with the option to forbid any land cover changes, non of the fractions are altered in this subroutine, although the total potential yield of each crop (including the amount of grass available for both grazing systems) is still computed and saved.

After these changes to the cropland (including grassland) fractions, there will be some cases in which the fractions sum to less than 1. If this is the case, the remaining fraction of the cell, $f_r$, is allocated between different crops. This part of the subroutine makes use of a quantity labelled ```sdempr```, here named SDP. It is defined

$$
    SDP = 1000p_0\frac{d_{0, t} - Y_0}{M_0} + \sum\limits_{c}1000p_c\frac{d_{c, t} - Y_c}{M_c}, \tag{7}
$$
where the factors of 1000 regard converting tonnes to kg. SDP can be thought of as the sum of the product of remaining crop and grass demands with the relative potential productivity of the grid cell currently being considered. Once SDP has been computed, non-grass crops are again looped through so that $f_r$ can be allocated to the different crops, whereby each crop's fraction in the cell increases, according to

$$
    f_{c, t} = f_{c, t} + MIN\left[f_r\times\frac{1000p_c\left(d_c - Y_c\right)}{M_c\times SDP}, 1-\sum\limits_{cc=1}^{cc=c}\left[f_r\times\frac{1000p_{cc}\left(d_{cc} - Y_{cc}\right)}{M_{cc}\times SDP}\right]\right]. \tag{8}
$$
As before, if this causes the cell to surpass the demand for crop $c$, the fraction of the cell allocated to that crop is instead increased according to

$$
    f_{c, t} = f_{c, t} + \frac{d_{c, t} - Y_c}{p_c\times M_c\times A\times FH_c\times MF_c}. \tag{9}
$$
Strangely, the management factor, $MF_c$ does not appear in this expression: **this may be a mistake in the code**, as it is used to compute $Y_c$. As this process is carried out, a sum of the additional fractions allocated from $f_r$ is kept, which we can label $F_{r}$. If, through some rounding error, $F_r>f_r$, breaking normalisation, the difference between the two is made up by subtracting $(F_r-f_r)$ from the fraction allocated to grass in that cell. The grass production and grassland area variables are updated accordingly.

In the event that $F_t$ was 0 (or smaller than the machine epsilon, which is considered 0 due to machine rounding errors), the grassland fraction is set to 0. This is the final step in the grid cell loop. Note that if a given grid cell has no non-zero fractions following ama aeg - this occurs when the cell was not agricultural land in the previous timestep - hea anl does not change this - **No expansion of agricultural land occurs in hea anl**.

Finally, the subroutine loops through all grid cells again, this time not in the order of suitability, to change the land cover types where necessary. The sum of all cropland and grassland fractions, $F_t$, is calculated again with the latest fraction values. When $F_t=0$, if the cell was previously agricultural land then it is converted to land cover type 4: regrowth forest (abandoning). As already alluded to, if the cell was previously non-agricultural, then this does not change.

## AgrLandAlloc

This subroutine handles the expansion of agricultural land. It is called by ```lcm_AgrExpansion```, which first loops through all crops in all regions to update two variables. The first, ```exagr``` is an array of length 26 (the number of regions), in which the r$^{th}$ entry is equal to the number of crops in that region for which demand has not already been met by the reallocation of existing cropland. The second variable, ```expansion```, is a boolean which reads true if any region requires the expansion of agricultural land - if any of the values in ```exagr``` are non-zero - and false if not. If ```expansion``` is true, then ```AgrLandAlloc``` is called.

```AgrLandAlloc``` loops through all grid cells in order of suitability, from high to low. For each grid cell, a further loop through each crop establishes the value of ```sdempr``` (SDP), as defined in eq. 7. SDP is only non-zero if ```exagr>0```. Note that a crop only contributes to eq. 7 if demand for that crop is greater than the existing potential yield, so that there can be no negative contributions to the summation.

If SDP is non-zero once this crop loop is complete, all non-grass crops are looped through a second time to calculate the crop fractions. The equation for determining the fractions is the same as that used to find the additional fractions in ```LCM_hea_anl``` when the remainder of the cell had to be filled,

$$
    f_{c, t} = MIN\left[\frac{1000p_c\left(d_c - Y_c\right)}{M_c\times SDP}, 1-\sum\limits_{cc=1}^{cc=c}\left[\frac{1000p_{cc}\left(d_{cc} - Y_{cc}\right)}{M_{cc}\times SDP}\right]\right]. \tag{10}
$$
The potential yield is then calculated for that crop in that cell, following equation 1 in [[IMAGE-LAND Allocation and Formulae]]. If the additional potential yield of a given crop's fraction of a given cell leads to total potential yield surpassing demand, then the fraction is recomputed, following

$$
    f_{c, t} = \frac{d_{c}-Y_c}{p_c\times M_c \times A\times FH_c \times MF_c}. \tag{11}
$$
When demand for a crop is met, the r$^{th}$ entry of ``exagr`` is decreased by 1.

For a given cell, once fractions have been allocated to all crops in need of further production, the remainder of the cell is allocated to grassland. After this step, if ```afrac```, the sum of all fractions, is greater than 0, then the land cover type of that cell is updated. If the land cover type of the cell was previously exclusively the growth of biofuels, then all biofuel fractions are set to 0. If the summation of all the potential crop yields in that cell is less than 10\% of ```MARGFRAC(r)```, the regional theoretical maximum, then the land cover type is updated to extensive grassland. Otherwise, the new land cover type is agricultural land.