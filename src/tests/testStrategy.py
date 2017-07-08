'''
Created on 21 Dec 2014

@author: Mark
'''
import unittest
from System.Strategy import ModelStrategy, Indexer, StrategyException, MeasureEnsembleStrategy,\
    ModelEnsembleStrategy, CompoundEnsembleStrategy, StrategyElement,\
    MeasureElement, ModelElement, PositionSelectionElement, StrategyContainerElement,\
    EnsembleStrategy
from mock import Mock, call
from tests.TestHelpers import buildTextDataFrame, buildNumericDataFrame,\
    buildNumericPanel
from System.Indicator import Signal, Crossover
from System.Forecast import BlockForecaster, Forecast, MeanForecastWeighting
from System.Position import SingleLargestF, Position, Returns
from System.Market import Market
from math import log
from pandas.core.frame import DataFrame


class TestStrategyConstruction(unittest.TestCase):

    def setUp(self):
        pass
        
    def testStrategyRequiresTradeTypeOnCreation(self):
        self.assertRaisesRegex(StrategyException, 
                               "Trade timing must be one of: 'OO', 'CC', 'OC', 'CO'.", 
                               ModelStrategy, "XX", "C")
        
    def testStrategyRequiresIndTiming(self):
        self.assertRaisesRegex(ValueError, 
                               "Ind_timing must be one of: 'O', 'C'", ModelStrategy, "OO", "X")
    
    def testStrategyChecksFieldsMarket(self):
        strategy = ModelStrategy("OO", "C")
        strategy.measure = Mock()
        strategy.model = Mock()
        strategy.select_positions = Mock()
        self.assertRaisesRegex(StrategyException,
                          """Missing parameters: market""", strategy.initialise)
        
    def testStrategyChecksFieldsAllEmpty(self):
        strategy = ModelStrategy("OO", "C")
        self.assertRaisesRegex(StrategyException,
                          """Missing parameters: market, measure, model, select_positions""",
                          strategy.initialise)
    

class TestStrategyInitialisation(unittest.TestCase):
    
    def setUp(self):
        self.strategy = ModelStrategy("OO", "C")
        self.indicator = Mock()
        self.forecasts = Mock()
        self.positions = Mock()
        self.strategy.market = Mock()
        self.strategy.market.tickers = ["ASX", "BHP"]
        self.strategy.measure = Mock(spec = Crossover)
        self.strategy.measure.return_value = self.indicator
        self.strategy.measure.execute = Mock()
        self.strategy.model = Mock(spec = BlockForecaster)
        self.strategy.model.return_value = self.forecasts
        self.strategy.model.execute = Mock()
        self.strategy.select_positions = Mock(spec = SingleLargestF)
        self.strategy.select_positions.return_value = self.positions
        self.strategy.select_positions.execute = Mock()
        
    def testStrategyChecksMissingFieldsOnInitialisation(self):
        self.strategy.check_fields = Mock()
        self.strategy.initialise()
        self.strategy.check_fields.assert_called_once_with()

    def testStrategyInitialisationProcedure(self):
        self.strategy.initialise()
        self.strategy.measure.assert_called_once_with(self.strategy)
        self.assertIs(self.strategy.indicator, self.indicator)
        self.strategy.model.assert_called_once_with(self.strategy)
        self.assertIs(self.strategy.forecasts, self.forecasts)
        self.strategy.select_positions.assert_called_once_with(self.strategy)
        self.assertIs(self.strategy.positions, self.positions)
        
    def testStrategyMeasureDoesNotCallIndexer(self):
        self.strategy.indexer.indicator = Mock()
        self.strategy.initialise()
        self.assertTrue(self.strategy.indexer.indicator.called == False)
        
    def testStrategyCallsIndexerForIndicator(self):
        indicator_series = buildTextDataFrame(["ASX", "BHP"], 10)
        indicator = Signal(indicator_series)
        self.strategy.indexer.indicator = Mock()
        self.strategy.indexer.indicator.return_value = indicator
        self.strategy.indicator = indicator
        result = self.strategy.lagged_indicator
        self.assertIsInstance(result, Signal)
        expected_calls = [call(indicator)]
        self.strategy.indexer.indicator.assert_has_calls(expected_calls)
        
    def testStrategyCallsIndexerForPositions(self):
        position_series = buildNumericDataFrame(["ASX", "BHP"], 10)
        positions = Position(position_series)
        self.strategy.indexer.positions = Mock()
        self.strategy.indexer.positions.return_value = positions
        self.strategy.positions = positions
        result = self.strategy.lagged_positions
        self.assertIsInstance(result, Position)
        expected_calls = [call(positions)]
        self.strategy.indexer.positions.assert_has_calls(expected_calls)
        
    def testStrategyCallsMarketForReturns(self):
        self.strategy.market.returns = Mock()
        self.strategy.market_returns
        expected_calls = [call(self.strategy.indexer)]
        self.strategy.market.returns.assert_has_calls(expected_calls)
        
    def testStrategyDoesNotRedoMeasure(self):
        self.strategy.indicator = self.indicator
        self.strategy.initialise()
        self.assertFalse(self.strategy.measure.execute.called)
        
    def testStrategyDoesNotRedoModel(self):
        self.strategy.forecasts = self.forecasts
        self.strategy.initialise()
        self.assertFalse(self.strategy.model.execute.called)
        
    def testStrategyDoesNotRedoPositions(self):
        self.strategy.positions = self.positions
        self.strategy.initialise()
        self.assertFalse(self.strategy.select_positions.execute.called)
        
        
class TestStrategyElements(unittest.TestCase):
    
    def setUp(self):
        class TestElement(StrategyElement):
            def execute(self, strategy):
                return Mock()
            def get_child(self, strategy):
                return strategy.child
                
        self.element = TestElement()
        self.strategy = Mock()
        self.strategy.child = None
    
    def testStrategyElementRequiresExecute(self):
        class ElementWithoutExecute(StrategyElement):
            def get_child(self, strategy):
                return None
        self.assertRaisesRegex(StrategyException, "Strategy Element must define 'execute'", 
                               ElementWithoutExecute(), self.strategy)
    
    def testStrategyElementRequiresGetChild(self):
        class ElementWithoutGetChild(StrategyElement):
            def execute(self, strategy):
                return None
        self.assertRaisesRegex(StrategyException, "Strategy Element must define 'get_child'", 
                               ElementWithoutGetChild(), self.strategy)
    
    def testStrategyElementProvidesID(self):
        self.assertIsNotNone(self.element.ID)
        
    def testStrategyElementAssignsIDtoChildren(self):
        child = self.element(self.strategy)
        self.assertEqual(child.parent, self.element.ID)
    
    def testStrategyElementRecognisesChildren(self):
        not_child = Mock()
        not_child.parent = "orphan"
        child = self.element(self.strategy)
        self.assertTrue(self.element.is_child(child))
        self.assertFalse(self.element.is_child(not_child))
        
    def testStrategyElementChecksForChildrenWhenCalled(self):
        self.element.is_child = Mock()
        self.element(self.strategy)
        self.element.is_child.assert_called_once_with(self.strategy.child)
        
    def testStrategyElementDoesNotExecuteIfChildExists(self):
        self.element.execute = Mock()
        self.strategy.child = Mock()
        self.strategy.child.parent = self.element.ID
        self.element(self.strategy)
        self.assertFalse(self.element.execute.called)
        
    def testMeasureElementChecksIndicator(self):
        self.measure = MeasureElement()
        self.measure.execute = Mock()
        indicator = Mock(spec = Signal)
        indicator.parent = self.measure.ID
        self.strategy.indicator = indicator
        returned = self.measure(self.strategy)
        self.assertIs(returned, indicator)
        self.assertFalse(self.measure.execute.called)

    def testModelElementChecksForecasts(self):
        self.model = ModelElement()
        self.model.execute = Mock()
        forecasts = Mock(spec = Forecast)
        forecasts.parent = self.model.ID
        self.strategy.forecasts = forecasts
        returned = self.model(self.strategy)
        self.assertIs(returned, forecasts)
        self.assertFalse(self.model.execute.called)

    def testPositionSelectionElementChecksPositions(self):
        self.select_positions = PositionSelectionElement()
        self.select_positions.execute = Mock()
        positions = Mock(spec = Position)
        positions.parent = self.select_positions.ID
        self.strategy.positions = positions
        returned = self.select_positions(self.strategy)
        self.assertIs(returned, positions)
        self.assertFalse(self.select_positions.execute.called)
        
    def testStrategyContainerElementImplementsShift(self):
        time_series = buildNumericDataFrame(["ASX", "BHP", "CBA"], 10)
        indicator = StrategyContainerElement()
        indicator.data = time_series
        lagged_indicator = indicator.shift(2)
        self.assertTrue(time_series.shift(2).equals(lagged_indicator.data))
        


class TestStrategyQueries(unittest.TestCase):
    
    def setUp(self):
        self.strategy = ModelStrategy("OO", "C")
        self.position_series = buildNumericDataFrame(["ASX", "BHP", "CBA"], 20)
        self.positions = Position(self.position_series)
        self.returns = buildNumericDataFrame(["ASX", "BHP", "CBA"], 20) / 10
        self.strategy.measure = Mock()
        self.strategy.positions = self.positions
        self.strategy.market = Mock()
        self.strategy.market.returns = Mock()
        self.strategy.market.returns.return_value = Returns(self.returns)
        
    def testStrategyReturnsMarketPricesForIndicator(self):
        self.strategy.market.close = "close"
        self.assertEqual(self.strategy.get_indicator_prices(), "close")
        self.strategy = ModelStrategy("OO", "O")
        self.strategy.market = Mock()
        self.strategy.market.open = "open"
        self.assertEqual(self.strategy.get_indicator_prices(), "open")
        
    def testStrategyReturns(self):
        self.strategy.indexer.positions = Mock()
        self.strategy.indexer.positions.return_value = self.positions
        
        strat_returns = self.strategy.returns
        
        expected_calls = [call(self.positions)]
        self.strategy.indexer.positions.assert_has_calls(expected_calls)
        self.assertIsInstance(strat_returns, Returns)
        self.assertEqual(len(strat_returns), len(self.positions.data))
        def expected_row(row):
            strat_rets = self.returns * self.position_series
            return log(strat_rets.iloc[row].sum() + 1)
        self.assertEqual(strat_returns.log(collapse_fun = "sum")[10], expected_row(10))


class TestTradeIndexing(unittest.TestCase):
    
    def setUp(self):
        self.trade_timing = "OO"
        self.ind_timing = "C"
        self.strategy = ModelStrategy(self.trade_timing, self.ind_timing)
        self.dummy_timeSeries = buildNumericDataFrame(["ASX", "BHP", "CBA"], 10)
        
    def testIndexerInstantiatedOnConstruction(self):
        self.assertIsInstance(self.strategy.indexer, Indexer)
        self.assertEqual(self.strategy.indexer.trade_timing, self.trade_timing)
    
    def testIndexerLagsIndicator(self):
        for trade_timing in ["OO", "CC", "OC", "CO"]:
            for ind_timing in ["O", "C"]:
                indexer = Indexer(trade_timing, ind_timing)
                if "OC" in ind_timing + trade_timing:
                    expected_series = self.dummy_timeSeries.shift(1)
                else:
                    expected_series = self.dummy_timeSeries.shift(2)
                calced_series = indexer.indicator(self.dummy_timeSeries)
                self.assertTrue(calced_series.equals(expected_series), 
                                trade_timing + ind_timing)

    def testIndexerLagsPositions(self):
        for trade_timing in ["OO", "CC", "OC", "CO"]:
            for ind_timing in ["O", "C"]:
                indexer = Indexer(trade_timing, ind_timing)
                if "OC" in ind_timing + trade_timing:
                    expected_series = self.dummy_timeSeries.shift(1)
                else:
                    expected_series = self.dummy_timeSeries.shift(2)
                calced_series = indexer.positions(self.dummy_timeSeries)
                self.assertTrue(calced_series.equals(expected_series), 
                                ind_timing + trade_timing)
                

class TestEnsembleStrategyQueries(unittest.TestCase):
    
    def setUp(self):
        ensemble_returns = Returns(buildNumericDataFrame(['all'], 10))
        returns1 = Returns(buildNumericDataFrame(['a'], 10))
        returns2 = Returns(buildNumericDataFrame(['b'], 10))
        returns3 = Returns(buildNumericDataFrame(['c'], 10))
        sub_returns = [returns1, returns2, returns3]
        self.strategy = EnsembleStrategy("CC", "C", [])
        self.strategy.indexer.positions = Mock()
        self.strategy.market = Mock()
        lagged_positions = Mock(spec = Position)
        lagged_positions.applied_to.return_value = ensemble_returns
        self.strategy.indexer.positions.return_value = lagged_positions
        sub_strat1 = Mock(spec = ModelStrategy)
        sub_strat1.returns = Mock(spec = Returns)
        sub_strat1.returns.collapse_by.return_value = returns1
        sub_strat2 = Mock(spec = ModelStrategy)
        sub_strat2.returns = Mock(spec = Returns)
        sub_strat2.returns.collapse_by.return_value = returns2
        sub_strat3 = Mock(spec = ModelStrategy)
        sub_strat3.returns = Mock(spec = Returns)
        sub_strat3.returns.collapse_by.return_value = returns3
        self.strategy.strategies = [sub_strat1, sub_strat2, sub_strat3]
        self.expected_result = ensemble_returns
        for returns in sub_returns:
            self.expected_result.data[returns.columns[0]] = returns.data
        
    
    def testEnsembleSummarisesAllReturns(self):
        returns = self.strategy.combined_returns("mean")
        for sub_strat in self.strategy.strategies:
            sub_strat.returns.collapse_by.assert_called_once_with("mean")
        self.assertEqual(returns.data.shape, self.expected_result.data.shape)


class TestStrategyMeasureEnsembleCreation(unittest.TestCase):
    
    def testMeasureEnsembleStratRequiresTradeTiming(self):
        self.assertRaisesRegex(StrategyException, 
                       "Trade timing must be one of: 'OO', 'CC', 'OC', 'CO'.", 
                       MeasureEnsembleStrategy, "XX", "C", [])
        
    def testStrategyChecksFieldsForecastWeight(self):
        strategy = MeasureEnsembleStrategy("OO", "C", [])
        strategy.market = Mock()
        strategy.measure = Mock()
        strategy.model = Mock()
        strategy.select_positions = Mock()
        self.assertRaisesRegex(StrategyException,
                          """Missing parameters: forecast_weight""", strategy.initialise)
    

class TestStrategyMeasureEnsembleInitialisation(unittest.TestCase):
    
    def setUp(self):
        self.parameter_set = [1, 2, 3, 4, 5]
        self.strategy = MeasureEnsembleStrategy("OO", "C", self.parameter_set)
        self.indicators = [Signal(p) for p in self.parameter_set]
        self.forecasts = [Forecast(p, 0) for p in self.parameter_set]
        self.positions = [Position(DataFrame([p])) for p in self.parameter_set]
        self.final_positions = Position(DataFrame(["final"]))
        self.positions += [self.final_positions]
        self.strategy.market = Mock(spec = Market)
        self.strategy.measure = Mock(spec = Crossover)
        self.strategy.measure.side_effect = self.indicators
        self.strategy.model = Mock()
        self.strategy.model.side_effect = self.forecasts
        self.strategy.select_positions = Mock()
        self.strategy.select_positions.side_effect = self.positions
        self.forecast_weight_output = "result"
        self.strategy.forecast_weight = Mock()
        self.strategy.forecast_weight.return_value = self.forecast_weight_output
        
        
    def testMeasureEnsembleCalculatesAllMeasures(self):
        self.strategy.initialise()
        expected_calls = [None] * len(self.parameter_set) * 2
        expected_calls[::2] = [call.update_param(param) for param in self.strategy.parameter_set]
        expected_calls[1::2] = [call(strategy) for strategy in self.strategy.strategies]
        self.assertEqual(self.strategy.measure.call_count, len(self.strategy.parameter_set))
        self.strategy.measure.assert_has_calls(expected_calls)
        
    def testMeasureEnsembleStoresAllMeasures(self):
        self.strategy.initialise()
        for i in bounds(len(self.parameter_set)):
            self.assertEqual(self.strategy._indicators[i], self.indicators[i])
        
    def testMeasureEnsembleForecastsEachMeasure(self):
        self.strategy.initialise()
        expected_calls = [call(strategy) for strategy in self.strategy.strategies]
        self.strategy.model.assert_has_calls(expected_calls)
        
    def testMeasureEnsembleStoresAllForecasts(self):
        self.strategy.initialise()
        for i in bounds(len(self.parameter_set)):
            self.assertEqual(self.strategy._forecasts[i], self.forecasts[i])
            
    def testMeasureEnsembleFeedsCorrectIndicatorToModel(self):
        expected_indicators = self.indicators.copy()
        def model_side_effect(strategy):
            expected_indicator = expected_indicators.pop(0)
            if not strategy.indicator == expected_indicator:
                raise Exception(
                    "Incorrect indicator, expected: {0}, received: {1}".format(
                                        expected_indicator.data, strategy.indicator.data))
            return self.forecasts.pop(0)
        self.strategy.model.side_effect = model_side_effect
        self.strategy.initialise()
    
    def testMeasureEnsembleSelectsAllPositions(self):
        self.strategy.initialise()
        expected_calls = [call(strategy) for strategy in self.strategy.strategies]
        self.strategy.select_positions.assert_has_calls(expected_calls)
        
    def testMeasureEnsembleStoresAllPositions(self):
        self.strategy.initialise()
        for i in bounds(len(self.parameter_set)):
            self.assertEqual(self.strategy._positions[i], self.positions[i])
            
    def testMeasureEnsembleFeedsCorrectForecastsToPosition(self):
        forecasts = self.forecasts + [self.forecast_weight_output]
        def position_side_effect(strategy):
            if not strategy.forecasts == forecasts.pop(0):
                raise Exception("Incorrect forecast")
            return self.positions.pop(0)
        self.strategy.select_positions.side_effect = position_side_effect
        self.strategy.initialise()
            
    def testMeasureEnsembleStoresWeightedForecasts(self):
        self.strategy.initialise()
        self.strategy.forecast_weight.assert_called_once_with(self.forecasts)
        self.assertEqual(self.strategy.forecasts, self.forecast_weight_output)
        
    def testMeasureEnsembleDeterminesPositionsFromMeanForecast(self):
        self.strategy.initialise()
        self.assertEqual(self.strategy.positions, self.final_positions)
        
   
class TestEnsembleMeanForecast(unittest.TestCase):
    
    def setUp(self):
        self.means = []
        self.stds = []
        for _ in bounds(3):
            self.means += [buildNumericPanel(['Forecast', 'a', 'b'], ["ASX", "BHP", "CBA"], 20)]
            self.stds += [buildNumericPanel(['Forecast', 'a', 'b'], ["ASX", "BHP", "CBA"], 20)]
        mean_sd = zip(self.means, self.stds)
        self.forecasts = [Forecast(mean, std) for mean, std in mean_sd]
        self.weight = MeanForecastWeighting()
    
    def testEnsembleForecastWeightCombinesForecasts(self):
        result = self.weight(self.forecasts)
        self.assertIsInstance(result, Forecast)
        for i in [0, 1, 5, 10, 19]:
            expected = [fcst.mean.iloc[i] for fcst in self.forecasts]
            expected = DataFrame(expected)
            self.assertTrue(result.mean.iloc[i].equals(expected.mean()))
    
    def testMeanForecastHandlesMissingValues(self):
        pass


class TestStrategyModelEnsembleInitialisation(unittest.TestCase):
    
    def setUp(self):
        self.parameter_set = [10, 20, 30, 40, 50]
        self.strategy = ModelEnsembleStrategy("OO", "C", self.parameter_set)
        self.indicator = Signal(10)
        self.forecasts = [Forecast(p, 0) for p in self.parameter_set]
        self.positions = [Position(DataFrame([p])) for p in self.parameter_set]
        self.final_positions = Position(DataFrame(["final"]))
        self.positions += [self.final_positions]
        self.strategy.market = Mock(spec = Market)
        self.strategy.measure = Crossover(0, 0)
        self.strategy.measure.execute = Mock()
        self.strategy.measure.execute.return_value = self.indicator
        self.strategy.model = Mock(spec = BlockForecaster)
        self.strategy.model.side_effect = self.forecasts
        self.strategy.select_positions = Mock()
        self.strategy.select_positions.side_effect = self.positions
        self.forecast_weight_output = "result"
        self.strategy.forecast_weight = Mock()
        self.strategy.forecast_weight.return_value = self.forecast_weight_output
        
    
    def testModelEnsembleCalculatesOneIndicatorOnly(self):
        self.strategy.initialise()
        self.strategy.measure.execute.assert_called_once_with(self.strategy)
        
    def testModelEnsembleForecastsIndicatorForEachParameter(self):
        self.strategy.initialise()
        expected_calls = [None] * len(self.parameter_set) * 2
        expected_calls[::2] = [call.update_param(param) for param in self.strategy.parameter_set]
        expected_calls[1::2] = [call(strategy) for strategy in self.strategy.strategies]
        self.assertEqual(self.strategy.model.call_count, len(self.strategy.parameter_set))
        self.strategy.model.assert_has_calls(expected_calls)
        
    def testModelEnsembleForecastFedCorrectIndicator(self):
        def model_side_effect(strategy):
            if not strategy.indicator == self.indicator:
                raise Exception("Incorrect indicator")
            return self.forecasts.pop(0)
        self.strategy.model.side_effect = model_side_effect
        self.strategy.initialise()
    
    def testModelEnsembleStoresAllForecasts(self):
        self.strategy.initialise()
        for i in bounds(len(self.parameter_set)):
            self.assertEqual(self.strategy._forecasts[i], self.forecasts[i])
            
    def testModelEnsembleSelectsAllPositions(self):
        self.strategy.initialise()
        expected_calls = [call(strategy) for strategy in self.strategy.strategies]
        self.strategy.select_positions.assert_has_calls(expected_calls)
    
    def testModelEnsembleStoresAllPositions(self):
        self.strategy.initialise()
        for i in bounds(len(self.parameter_set)):
            self.assertEqual(self.strategy._positions[i], self.positions[i])
            
    def testModelEnsembleFeedsCorrectForecastsToPosition(self):
        forecasts = self.forecasts + [self.forecast_weight_output]
        def position_side_effect(strategy):
            if not strategy.forecasts == forecasts.pop(0):
                raise Exception("Incorrect forecast")
            return self.positions.pop(0)
        self.strategy.select_positions.side_effect = position_side_effect
        self.strategy.initialise()
            
    def testModelEnsembleStoresWeightedForecasts(self):
        self.strategy.initialise()
        self.strategy.forecast_weight.assert_called_once_with(self.forecasts)
        self.assertEqual(self.strategy.forecasts, self.forecast_weight_output)
        
    def testModelEnsembleDeterminesPositionsFromMeanForecast(self):
        self.strategy.initialise()
        self.assertEqual(self.strategy.positions, self.final_positions)
        
        
class TestCompoundEnsembleCreation(unittest.TestCase):
    
    def setUp(self):
        self.model_parameters = [10, 20, 30]
        self.measure_parameters = [1, 2, 3, 4, 5]
        self.strategy = CompoundEnsembleStrategy("CC", "O", 
                                                 (self.model_parameters, self.measure_parameters))
        self.indicators = [Signal(p) for p in self.measure_parameters]
        self.forecasts = [Forecast(p, 0) for p in self.model_parameters]
        self.sub_positions = [Position(DataFrame([p])) for p in self.measure_parameters]
        self.super_positions = [Position(DataFrame([p])) for p in self.model_parameters]
        self.final_positions = Position(DataFrame(["final"]))
        self.strategy.market = Mock(spec = Market)
        self.measure = Mock(spec = Crossover)
        self.model = Mock(spec = BlockForecaster)
        self.strategy.measure = self.measure
        self.strategy.model = self.model
        self.strategy.select_positions = Mock()
        self.forecast_weight_output = "result"
        self.strategy.forecast_weight = Mock()
        self.strategy.forecast_weight.return_value = self.forecast_weight_output
        
    def testSubStratsAreEnsembles(self):
        self.strategy.initialise()
        for strategy in self.strategy.strategies:
            self.assertIsInstance(strategy, MeasureEnsembleStrategy)
        
    def testModelCalledForEachParamAndIndicator(self):
        self.strategy.initialise()
        num_models = len(self.model_parameters)
        num_measures = len(self.measure_parameters)
        self.assertEqual(len(self.strategy.strategies), num_models)
        expected_calls = []
        for p, sub_strat in enumerate(self.strategy.strategies):
            param = self.model_parameters[p]
            expected_calls += [call.update_param(param)]
            expected_calls += [call(strat) for strat in sub_strat.strategies]
        self.assertEqual(self.strategy.model.call_count, num_models * num_measures)
        self.strategy.model.assert_has_calls(expected_calls)


    
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()