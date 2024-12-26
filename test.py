from GQLib.Framework import Framework
from GQLib.Optimizers import MPGA, PSO

fw = Framework("daily")

# On renseigne les dates du sample et l'optimiseur à utiliser
fw.process("04/01/2003", "04/02/2008", PSO)