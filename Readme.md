# Ensemble System Development

## About
Ensemble System Development is a framework for researching, building, and testing trading
strategies. The 'ensemble' in the name is a legacy as originally the packages main purpose
was to investigate the performance of groups of strategies. While this is still possible in
the framework, it is now more of a general purpose set of research tools.

## TODO
* Strategy position weighting (using trade modifier)
* Assess weighted vs unweighted positions
* Variable stop sizing (i.e. with position size) 
* ETD vs Max Drawdown, Volatility adjusted MAE plot
* Portfolio events for logging
* Review summary_report
* Fix Fuzzer multiprocessing

### Analysis
* Investigate drawdown profile for moves of a certain magnitude
* Neural network trial
* Getting more data
    + Extraction from pdfs
    + Using simplified fundamentals (i.e. CMC summary data)
* Define system testing process
    + In-sample exploration
        - Subset NYSE data
        - Parameter sensitivity
        - Filters
        - Use of stops
        - Position weighting
        - Cross-validation across market constituents
        - Performance report
        - Equity curve
    + Portfolio testing
        - Effect of starting capital
        - Cross-validation across market constituents
        - Different money management strategies
    + Out of sample testing
        - Extended NYSE data
        - ASX data
 
### Design and Refactoring
#### Architecture
- DONE - Define process for strategy execution
- DONE - Construct strategy with measure parameter set.
- DONE - Construct strategy with model parameter set
- DONE - Construct strategy with model and measure parameter sets
- DONE - Clean up modules names and associated classes
#### Refactoring
- DONE - Replace DataFrame and Panel builds with helper method constructions
- DONE - Weight for positions to be renamed to PositionSelector
- DONE - Move ind_timing from Measure to Strategy. Have strategy return required prices.
- DONE - Ensemble Strategy to contain sub-strategies instead of collections of forecasts etc.
*Speed up forecast window method.
*Maybe change strategy initialisation to require market input.
*Abstract out a data container object for use by market, models etc.

#### Interface definitions
- DONE - Indicator class definition
- DONE - Indicator implements __getitem__ method to allow index by ticker
- DONE - Supply indicator data when initiating
- DONE - PositionModel must return Position object
- DONE - Strategy to provide strat returns series.

### New Features
- DONE - Add ability to plot long only, and short only results
- DONE - Strategy does not redo existing components on initialise
*Plot long and short results together
*Calculate performance metrics
- DONE - Strategy to produce label for self

### Error proofing
*Ensemble forecast mean to handle missing values.
- DONE - Indicator levels are appropriate data type (e.g. string)
- DONE - Testing for IndicatorMeasure
- DONE - Measure object must implement ind_timing parameter
- DONE - Confirm Forecast data is same content (ticker and dates) as input

### FIXES
- DONE - Change BlockForecaster execute method not require ticker
- DONE - Change Crossover execute method not require ticker
- DONE - PositionModel expects Forecast, whereas Strategy provides itself when calling model.
- DONE - Strategy.indicator returns DataFrame: conflict between object and need for indicator lag.
- DONE - Forecast optF method is generating infinities
- DONE - Position model is not able to handle NaN values
- DONE - Indicator.levels doesn't handle NaNs
- DONE - Returns shouldn't be shifted after initial calculation within Market.
- DONE - Indicator date shifts are incorrect.
- DONE - Positions need to be shifted when calculating strat returns.

