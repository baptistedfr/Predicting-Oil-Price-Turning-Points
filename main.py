from GQLib.MiaouFramework import MiaouFramework, DataName
from GQLib.subintervals import SubIntervalMethod
from GQLib.Optimizers import NELDER_MEAD, MPGA, SA, TABU
from GQLib.filterings import LombFilter, LPPLSConfidence, StationarityFilter
from GQLib.Models import LPPL, LPPLS
from GQLib.logging import configure_logger

configure_logger("INFO")

framework = MiaouFramework(
    data_names=[DataName.SP500],
    set_dates={
        "SP500": {
            "Period1" : ("2007-01-01", "2007-02-01"),
        }
    },
    frequency=1,
    optimizer=NELDER_MEAD(LPPLS),
    filtering_method=LPPLSConfidence,
    subinterval_method=SubIntervalMethod.DIDOU,
    window_length=1000,
    optimizer_params={'maxiter': 5000, 'maxfev': 15000, 'fatol': 1e-4, 'xatol': 1e-4}
)
framework.run()
framework.visualize(save=True)
