from GQLib.MiaouFramework import MiaouFramework, DataName
from GQLib.subintervals import SubIntervalMethod
from GQLib.Optimizers import NELDER_MEAD, MPGA
from GQLib.filterings import AbstractFilter, enculefilter
from GQLib.Models import LPPL, LPPLS


framework = MiaouFramework(
    data_names=[DataName.BTC],
    set_dates={
        "BTC": {
            "Period1" : ("2020-01-01", "2020-01-07")
        }
    },
    frequency=1,
    optimizer=NELDER_MEAD(LPPLS),
    filtering_method=enculefilter,
    subinterval_method=SubIntervalMethod.MIAOU,
    window_lenght=300
)
framework.run()
framework.visualize()