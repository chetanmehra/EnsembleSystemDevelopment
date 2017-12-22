'''
The event_signals module handles creation of signals from disrete events rather than
continuous measures. This allows creation of trades from separate entry and exit setups.
'''

from pandas import Panel, DataFrame

from system.interfaces import SignalElement
from data_types.signals import Signal


class EventSignal(SignalElement):
    '''
    EventSignal is the basic class for generation of signals from event based entries and exits.
    '''
    def __init__(self, entries, exits):
        self.entries = entries
        self.exits = exits


    def execute(self, strategy):
        events_data = strategy.get_empty_dataframe(fill_data = '-')
        
        for entry_setup in self.entries:
            events_data = entry_setup.fill_events(strategy, events_data)

        for exit_setup in self.exits:
            events_data = exit_setup.fill_events(strategy, events_data)

