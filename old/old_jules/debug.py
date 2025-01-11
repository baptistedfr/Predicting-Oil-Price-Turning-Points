from Lib.Framework import Framework
from Lib.Optimizers import MPGA

fw = Framework("daily")

#fw.process("04/01/2003", "04/02/2008", MPGA)



# On peut mettre show=True pour afficher les résultats
fw.analyze("results-2003_2008_try.json", show=True)


fw.visualize()
