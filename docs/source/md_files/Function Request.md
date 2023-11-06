# Lue Functions

Please note that all equations referred to here can be found in <a href='IMAGE-LAND Allocation and Formulae.html'>IMAGE-LAND Allocation and Formulae</a>.

A rewritten IMAGE-LAND model implemented using LUE would benefit from two inbuilt functions. The first, the integration function, which performs a cumulative sum over the cells in a raster in a specified order, determined here by the suitability map. The second, the integration-allocation function, is more intricate.

The structure of IMAGE-LAND can be replicated with map algebra operations - for instance, the 16 crop fraction rasters can be used to calculate new crop fractions for existing cropland, following eq. (4), for each of the 26 regions. A map of potential yield can then be computed, following eq. 1. Once this is complete, the integration function can be used to find, for each crop, whether and where demand for that crop is met.

## Integration Function
The function should take in as an input N integrands, in this case the potential yield rasters, alongside the map of regions and the demands for each region. The outputs should be N integrated yield rasters and N demand met ($DM$) Boolean rasters.

	\FOR{$r=1,\ldots,n_{regions}$}
		\FOR{$j=1,\ldots,n_{cells}$}
			\FOR{$c=1,\ldots,n_{crops}$}
				\STATE $Y_{j, c} = Y_{j-1, c} + y_{j, c}$
				\IF{$Y_{j, c}>d_{r, c}$}
					\STATE $DM_{j, c}=True$
				\ELSE
					\STATE $DM_{j, c}=False$
				\ENDIF
				\IF{$Y_{j, c}>d_{r, c}, \forall c$}
					\BREAK 
				\ENDIF
			\ENDFOR
		\ENDFOR
	\ENDFOR

Once this function has been run, the $DM$ rasters can be used to determine which of outcomes 1-3 and their respective sub-outcomes correspond to the situation.

## Integration-Allocation Function
The primary use for the integration-allocation function in IMAGE-LAND-LUE will be in the expansion of agricultural land, where it will allocate crop fractions based on the potential yields of the cells it has already allocated. To achieve this, the function will need to compute SDP, and from this the fractions and subsequently the yield. The rasters $SDPF_{c} = 1000\times p_{c} / M_c$ will be defined outside of the function to decrease the number of unnecessary operations within the function. Similarly the rasters $yf_c = p_c\times M_c\times A\times FH_{c} \times MF_c$ will be defined.

The function should take as inputs N+1 $SDPF$ rasters and N+1 $yf_c$ rasters, where the extra rasters relative to before are related to grass. As before, the regions map should also be an input, as well as the demands for each region. An additional set of N rasters, $f^\prime_c$ should be an input; this is explained in section 2.3.

### Integration-Allocation Algorithm
	\FOR{r=1,\ldots,n_{regions}$}
		\FOR{$j=1,\ldots,n_{cells}$}
			\STATE $SDP_{j, c}=0$
			\FOR{$c=0,\ldots,n_{crops}$}
				\IF{\NOT $DM_{j-1, c}$}
					\STATE $SDP_{j, c} = SDP_{j, c} + SDPF_{j, c}\times (d_{r, c}-Y_{r, c})$
				\ENDIF
			\ENDFOR
			\FOR{$c=1,\ldots,n_{crops}$}
				\IF{$DM_{j-1, c}$}
					\STATE $f_{j, c} = 0$
				\ELSE
					\STATE $f_{j, c} = SDPF_{j, c}\times (d_{r, c}-Y_{r, c})/SDP_{j, c}$
				\ENDIF
					\STATE $y_{j, c} = yf_{j, c}(\times f_{j, c}+f^\prime_{j, c})$
					\STATE $Y_{j, c} = Y_{j-1, c} + y_{j, c}$
				\IF{$Y_{j, c}>d_{r, c}$}
					\STATE $DM_{j, c}=True$
				\ELSE
					\STATE $DM_{j, c}=False$
				\ENDIF
				\IF{$Y_{j, c}>d_{r, c}, \forall c$}
					\BREAK 
				\ENDIF
			\ENDFOR
		\ENDFOR
	\ENDFOR
	\STATE

Note that crops are looped over twice - the first time, this loop includes grass, but the second time it does not.

The integration-allocation function should return the potential yield rasters, the integrated potential yield rasters, the crop fraction rasters and the demand met rasters.


## Use of the Functions
As mentioned, the integration function will be used to determine the areas in which demand has been surpassed by agricultural production. The integration-allocation function will form the basis for the expansion of cropland. However, the latter has an additional use: once the integration function is used, and after the remaining cell fractions have been allocated to grassland, the crop fractions for overproduced crops will be set to 0 in the regions in which they are overproduced. Consequently, as described in section 1, there will be new fractions in those cells freed up for the other crops. These remaining fractions $f_r$ can separated into new rasters and multiplied with the SDPF, and the integration-allocation function can be called with this modified SDPF to reallocate the remaining fractions. For this purpose, the $f^\prime_c$ input rasters will be set to the existing crop fractions computed prior to the use of the integration function.

## Possible Generalisation
* To maximise the generality of the integration function, the outputting of a \`demand met\' Boolean raster could be made optional.
* For the integration-allocation function, perhaps the way that the input rasters are combined with the integral to form the integrand could be a user-defined input function?