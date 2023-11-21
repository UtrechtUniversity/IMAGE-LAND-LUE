# IMAGE-LAND Allocation and Formulae

This page briefly describes the allocation of land to food crops and grassland in IMAGE-LAND, detailing the equations used to calculate the fractions at each stage. The specifics of biofuel crops and timber production in IMAGE-LAND are omitted from this description.

The IMAGE-LAND grid is made up of cells with a side length of 5 arc-minutes, resulting in a 4320 by 2160 cell map. Each of these cells comprises 16 fractions, 1 for each food crop, alongside other fractions which are not explicitly considered for the time being. IMAGE-Land features 20 <a href='Land Cover Types.html'>Land Cover Types</a>, the first two of which are agricultural land - also referred to as cropland - and extensive grassland. Extensive grassland is the land cover type allocated to agricultural land which has fallen below a threshold of productivity (WHICH VALUE?).

Cells with non-cropland land-cover types are made up entirely of crop fractions of 0, with the exception of extensive grassland, which is a special case where cropland is so low productivity that it has been recategorised. Within cropland, each cell is allocated entirely to food crops and grassland.

For a given crop, $c$, allocated to a fraction $f_c$ of a cell, the (potential) yield, $y_c$, of crop $c$ in that cell is given by

$$y_c = f_c\times p_c\times M_c\times A\times FH_c\times MF_c, \tag{1}$$
where $M_c$ is the maximum potential productivity of each crop and $p_c$ (grmppc in the code) is the grid-reduced potential productivity as a fraction of $M_c$. $A$ is the area of the cell, $FH_c$ is the fraction of the planted crop that is harvested in a given timestep and $MF_c$ is the management factor. 

Each timestep, $t$, for each of the 26 world regions, IMAGE-LAND reads in the new demands for each crop and allocates grid cells in the order of a single suitability map, from the cells with the highest suitability to those with the lowest. IMAGE-LAND begins the allocation process by isolating the existing cropland areas and reallocating them. To respond to the change in demand for each crop, the fractions of each grid cell allocated to each crop are recalculated, taking into account their values in the previous timestep, $t-1$. This process uses two quantities, $cf_1$ and $cf_3$ ($cf_2$ also exists but is not really relevant):

$$cf_1 = f_{0, t-1}\frac{d_{0, t}}{GI\times d_{0, t-1}} + \sum\limits_{c} f_{c, t-1}\frac{d_{c, t}}{d_{c, t-1}MF_c},\tag{2}$$
where $GI$ is the grazing intensity and $d$ is the demand, and

$$
    cf_3 = f_{0, t-1} + \sum\limits_{c} f_{c, t-1}.\tag{3}
$$
Once these quantities have been obtained, the non-grass crops are looped through again, so that the fractions allocated to each of them can be computed, following

$$
    f_{c, t} = f_{c, t-1}MF_c\frac{d_{c, t}}{d_{c, t-1}}\frac{cf_3}{cf_1}.\tag{4}
$$
If the sum of all crop fractions, $F_{c, t}<1$, then the remainder of the cell is allocated to grass. The reallocation of existing cropland leads to **one** of 3 broad outcomes:
1. The demand for every crop is met before all existing cropland is reallocated.
2. The demand for every crop is met when all existing cropland is reallocated.
3. The demand for at least some crops is not yet met when all existing cropland is reallocated.

In the case of outcome 1, the fractions of each crop type allocated to the remaining cropland once demand has been met should be 0. This corresponds to the abandoning of agricultural land where it is no longer needed. Importantly, however, there is a subtlety to outcome 1, as well as the other outcomes. In fact, outcome 1 can describe 2 situations:

{style=lower-alpha}
1. The demand for each and every crop is met at the same time (when the same cell is reached).
2. The demand for some crops is met before others.

Under outcome 1(a) - a less likely outcome than 1(b) - the reallocation process is relatively simple: every crop fraction is allocated according to eq. (4) until demand is met, then the fractions in the remaining cropland cells are all set to 0. To address outcome 1(b), however, there is an important subtlety in IMAGE-LAND. All cropland cells have their crop fractions recalculated, regardless of whether demand is met. They are then looped through a second time, in order of suitability, to determine when demand is met for each crop and re-work the new cell fractions accordingly. Under outcome 1(b), once demand for some crops has been met, the fraction of land allocated to those crops in subsequent cells is 0. Accordingly, a new fraction of those cells, $f_r$, is freed up for the crops for which demand has not yet been met. This is achieved with the help of a new quantity,
labelled ```sdempr```, here named SDP. It is defined

$$
    SDP = 1000p_0\frac{d_{0, t} - Y_0}{M_0} + \sum\limits_{c}1000p_c\frac{d_{c, t} - Y_c}{M_c}, \tag{5}
    
$$
where the factors of 1000 regard converting tonnes to kg. $Y_c$ is the integrated potential yield over all cells already looped over. SDP can be thought of as the sum of the product of remaining crop and grass demands with the relative potential productivity of the grid cell currently being considered. Once SDP has been computed, non-grass crops are again looped through so that $f_r$ can be allocated to the different crops, whereby each crop's fraction in the cell increases, according to

$$
    f_{c, t} = f_{c, t} + MIN\left[f_r\times\frac{1000p_c\left(d_c - Y_c\right)}{M_c\times SDP}, 1-\sum\limits_{cc=1}^{cc=c}\left[f_r\times\frac{1000p_{cc}\left(d_{cc} - Y_{cc}\right)}{M_{cc}\times SDP}\right]\right]. \tag{6}
$$
This process of fraction adjustment is carried out until either demand is met (outcome 1(b)) or, in the case of outcomes 2 and 3, existing cropland is filled. Note that, in the case of outcome 1, those cells in which all new fractions are 0 (cells that will be abandoned) change land cover type to regrowth forest (abandoning).

As with outcome 1, outcome 2 can describe 2 situations:

{style=lower-alpha}
1. The demand for each and every crop is only met when the last cell in existing cropland is allocated.
2. The demand for some crops is met before others, with demand for the last crop(s) only being met when all existing cropland is allocated.

These are very similar to outcomes 1(a) and 1(b), with the difference that no cropland cells end up with 0 fractions for all crops: no agricultural land needs to be abandoned.

Outcome 3 contains 3 possibilities:

{style=lower-alpha}
1. The demand for none of the crops is met when all existing cropland has been reallocated.
2. The demand for some crops is met exactly when all existing cropland is allocated, with some unmet demand for other crops.
3. The demand for some, but not all crops is met before existing cropland is reallocated, with unmet demand for other crops.

Outcomes 3 (a), (b) and (c) all require an expansion of cropland. In the case of outcomes 3 (b) and (c), cropland only needs to be expanded for some crops. In the case of outcome 3(c), the fractions allocated to some crops are already 0 for some of the existing cropland.

The equation for determining the fractions in new cropland is the same as that used to find the additional fractions in existing cropland when the remainder of the cell had to be filled: SDP is first computed, then

$$
    f_{c, t} =  MIN\left[\frac{1000p_c\left(d_c - Y_c\right)}{M_c\times SDP}, 1-\sum\limits_{cc=1}^{cc=c}\left[\frac{1000p_{cc}\left(d_{cc} - Y_{cc}\right)}{M_{cc}\times SDP}\right]\right].\tag{7}
$$
The potential yield is then calculated for that crop in that cell, according to equation (\ref{eq:yield}). If the additional potential yield of a given crop's fraction of a given cell leads to total potential yield surpassing demand, then the fraction is recomputed, following

$$
    f_{c, t} = \frac{d_{c}-Y_c}{p_c\times M_c \times A\times FH_c \times MF_c}.\tag{8}
$$
where $r_c$ is the remaining demand. Note that when not all crops require allocation to new cropland (outcomes 3(b) and 3(c)), the remaining demand for those crops will be 0, so they are automatically removed from consideration by eq. 8. For each new cell, the remaining demand for a given crop is calculated by taking the remaining demand for that crop when the previous cell was allocated and subtracting the potential yield from that cell, using eq. 1.