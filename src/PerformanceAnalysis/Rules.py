
from pandas import DataFrame
from PerformanceAnalysis.Metrics import summary_report

def test_trailing_stop(strat, stops):
    result = DataFrame(columns = [0] + stops)
    result[0] = summary_report(strat.trades)
    for s in stops:
        result[s] = summary_report(strat.trades.apply_trailing_stop(strat, s))
    return result

def test_stop_loss(strat, stops):
    result = DataFrame(columns = [0] + stops)
    result[0] = summary_report(strat.trades)
    for s in stops:
        result[s] = summary_report(strat.trades.apply_stop_loss(strat, s))
    return result
