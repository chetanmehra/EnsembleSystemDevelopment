"""
The events module provides classes for dealing with events related to 
position changes, e.g. entries, exits, or adjustments.
"""
import pandas as pd

from data_types.constants import TradeSelected
from datetime import datetime

class EventCollection:

    def __init__(self, events, tickers, index):
        self.factory = EventFactory()
        self.events = events
        self.tickers = tickers
        self.index = index
        self._related_entries = {} # Dict of entries by ticker
        self._related_exits = {} # Dict of exits by ticker
        
        
    @staticmethod
    def from_position_data(pos_data):

        # Creation of events
        delta = pos_data - pos_data.shift(1)
        delta.iloc[0] = 0
        # Add entries occuring on first day
        non_zero_start = (pos_data.iloc[0] != 0)
        if any(non_zero_start):
            delta.iloc[0][non_zero_start] = pos_data.iloc[0][non_zero_start]
        # Identify locations of events
        event_keys = pd.DataFrame("-", index = pos_data.index, columns = pos_data.columns, dtype = str)
        # Note adjustment goes first and then gets overwritten by entry and exit
        event_keys[delta != 0] = AdjustmentEvent.Label
        # TODO: Event creation logic will not work if switching from long to short position
        #       in one day (i.e. from positive to negative size), or vice versa.
        event_keys[(delta != 0) & (delta == pos_data)] = EntryEvent.Label
        event_keys[(delta != 0) & (pos_data == 0)] = ExitEvent.Label
        
        factory = EventFactory()
        events = []
        tickers = event_keys.columns[pos_data.abs().sum() != 0]

        for ticker in tickers:
            ticker_events = event_keys[ticker]
            ticker_deltas = delta[ticker]
            event_dates = ticker_events.index[ticker_events != "-"]
            event_types = ticker_events[event_dates]
            event_sizes = ticker_deltas[event_dates]
            event_data = list(zip(event_types, [ticker] * len(event_types), event_dates, event_sizes))
            events.extend([factory.build(*event) for event in event_data])

        return EventCollection(events, tickers, delta.index)

    
    def find(self, condition):
        '''
        Accepts a callable condition object (e.g. lambda expression), which 
        must accept an Event. Returns a list of events which meet the condition.
        '''
        return [event for event in self.events if condition(event)]

    def __getitem__(self, key):

        if isinstance(key, datetime):
            return self.find(lambda e: e.date == key)
        elif isinstance(key, str):
            return self.find(lambda e: e.ticker == key)
        elif isinstance(key, int):
            return self.events[key]
        else:
            raise ValueError("key must be a date, ticker (string), or integer")

    @property
    def last_date(self):
        return self.index[-1]

    def related_entries(self, ticker):
        if ticker in self._related_entries:
            return self._related_entries[ticker]
        else:
            related_entries = self.find(lambda e: e.Label == EntryEvent.Label and e.ticker == ticker)
            self._related_entries[ticker] = related_entries
            return related_entries

    def related_exits(self, ticker):
        if ticker in self._related_exits:
            return self._related_exits[ticker]
        else:
            related_exits = self.find(lambda e: e.Label == ExitEvent.Label and e.ticker == ticker)
            self._related_exits[ticker] = related_exits
            return related_exits

    def next_exit(self, entry, exits = None):
        if exits is None:
            # Find all exits related to the entry.ticker and after the entry.date
            exits = [exit for exit in self.exits if exit.date > entry.date and exit.ticker == entry.ticker]
        else:
            # We assume the supplied exits are related to the entry.ticker
            exits = [exit for exit in exits if exit.date > entry.date]
        if len(exits):
            # sort exits by date ascending (i.e. earliest first).
            # and return the soonest
            exits.sort(key = lambda e: e.date)
            return exits[0]
        else:
            # create an exit event for on the last day
            exit_date = self.last_date
            return self.factory.build(ExitEvent.Label, entry.ticker, exit_date, -entry.size) 


class EventFactory:

    def __init__(self):
        self.event = {
            EntryEvent.Label : EntryEvent, 
            ExitEvent.Label : ExitEvent, 
            AdjustmentEvent.Label : AdjustmentEvent}

    def build(self, type, ticker, date, size):
        return self.event[type](ticker, date, size)


class TradeEvent:

    def __init__(self, ticker, date, size):
        self.ticker = ticker
        self.date = date
        self.size = size

    def __str__(self):
        first_line = '{0:^10}\n'.format(self.ticker)
        second_line = 'Date: {0:10}\n'.format(str(self.date))
        third_line = 'Size: {0:^10.1f}\n'.format(self.size)
        return first_line + second_line + third_line

    def update(self, positions):
        positions.type[self.ticker] = self.Label
        positions.selected[self.ticker] = TradeSelected.UNDECIDED


class EntryEvent(TradeEvent):
    Label = "entry"

    def update(self, positions):
        positions.target_size[self.ticker] = self.size
        super().update(positions)
        

class ExitEvent(TradeEvent):
    Label = "exit"

    def update(self, positions):
        positions.target_size[self.ticker] = 0
        super().update(positions)
        

class AdjustmentEvent(TradeEvent):
    Label = "adjustment"

    def update(self, positions):
        positions.target_size[self.ticker] += self.size
        positions.type[self.ticker] = self.Label
        if positions.applied_size[self.ticker] != 0:
            positions.selected[self.ticker] = TradeSelected.UNDECIDED
        else:
            positions.selected[self.ticker] = TradeSelected.NO
        
 