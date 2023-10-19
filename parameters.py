"""
Defines key model parameters.
"""

# whether IMAGE-LAND-LUE is running independently of the rest of IMAGE
STANDALONE = True

# number of food crops
NFC = 16

# number of food crops, including grass
NGFC = 17

# number of biofuel crop classes
NBC = 5

# number of non-grass crop classes (food RF, biofuel and food IR)
NFBFC = 37

# number of fodder (grass), food and biofuel crop classes
NGFBC = 22

# same, but without grass
NFBF = 21

# number of grazing systems
NGS = 2
NGST = NGS + 1 # NGS + 'total'
