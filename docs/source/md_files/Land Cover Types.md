# Land Cover Types

IMAGE-LAND features 20 land cover types (LCTs), namely:
1. Agricultural land
2. Extensive grassland
3. Carbon plantation
4. Regrowth forest (abandoning)
5. Regrowth forest (timber)
6. Biofuels
7. Ice
8. Tundra
9. Wooded tundra
10. Boreal forest
11. Cool conifer forest
12. Temp. mixed forest
13. Temp. decid. forest
14. Warm mixed forest
15. Grassland/steppe
16. Hot desert
17. Scrubland
18. Savanna
19. Tropical woodland
20. Tropical forest.

Extensive grassland is the land cover type allocated to agricultural land, the productivity of which has fallen below 20% of the maximum productivity of each crop.

## Conversion
The two types of land conversion currently implemented in IMAGE-LAND-LUE occur when agricultural land is abandoned (following a drop in crop demands) and when agricultural land is expanded.

In the former case, abandoned agricultural land is always converted to LCT4: regrowth forest (abandoning).

In the latter case, when crop demands increase enough to cause an expansion in agricultural land, the cells with the highest suitability are converted to LCT1, no matter what their previous land cover type.