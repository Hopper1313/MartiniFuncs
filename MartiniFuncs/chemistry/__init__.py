"""Chemistry-specific pair models: everything that names a particular reaction system lives here.

diels_alder   geometry, criteria and model base class shared by every diene + dienophile pair model
cpd_n4t       cyclopentadiene + N4T maleimide Diels-Alder pairs
anthracene_ma anthracene + maleic anhydride Diels-Alder pairs

BUILTIN_PAIR_MODELS maps each built-in model name to "module:attribute";
MartiniFuncs.pairs.model imports a model the first time it is named.
"""
BUILTIN_PAIR_MODELS = {"cpd_n4t": "MartiniFuncs.chemistry.cpd_n4t.model:CpdN4tModel",
                       "anthracene_ma": "MartiniFuncs.chemistry.anthracene_ma.model:AnthraceneMaModel"}

# Analysis caches written before the pair model was recorded were all made with this model.
UNRECORDED_CACHE_PAIR_MODEL = "cpd_n4t"
