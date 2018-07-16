"""
The events module provides classes for dealing with events related to 
position changes, e.g. entries, exits, or adjustments.
"""
import pandas as pd

from data_types import Collection, CollectionItem
from data_types.constants import TradeSelected


class EventCollection(Collection):

    def __init__(self, events, tickers, index):
        super().__init__(events)
        self.factory = EventFactory()
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
            event_indexes = [pos_data.index] * len(event_dates)
            event_data = list(zip(event_types, [ticker] * len(event_types), event_dates, event_sizes, event_indexes))
            events.extend([factory.build(*event) for event in event_data])

        return EventCollection(events, tickers, delta.index)

    def copy(self, items = None, **kwargs):
        if items is None:
            items = [item.copy() for item in self.items]
        return EventCollection(items, self.tickers, self.index)
    
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
            exits = self.related_exits(entry.ticker)
        exits = [exit for exit in exits if exit.date > entry.date]
        if len(exits):
            # sort exits by date ascending (i.e. earliest first).
            # and return the soonest
            exits.sort(key = lambda e: e.date)
            return exits[0]
        else:
            # create an exit event for on the last day
            exit_date = self.last_date
            return self.factory.build(ExitEvent.Label, entry.ticker, exit_date, -entry.size, self.index) 


class EventFactory:

    def __init__(self):
        self.event = {
            EntryEvent.Label : EntryEvent, 
            ExitEvent.Label : ExitEvent, 
            AdjustmentEvent.Label : AdjustmentEvent}

    def build(self, type, ticker, date, size, index):
        return self.event[type](ticker, date, size, index)


class TradeEvent(CollectionItem):
    Label = ""

    def __init__(self, ticker, date, size, index):
        self.ticker = ticker
        self.date = date
        self.day = list(index).index(date)
        self.index = index
        self.size = size
        self.tuple_fields = ['ticker', 'Label', 'date', 'size']

    def __str__(self):
        first_line = '{0:^10}\n'.format(self.ticker)
        second_line = 'Date: {0:10}\n'.format(str(self.date))
        third_line = 'Size: {0:^10.1f}\n'.format(self.size)
        return first_line + second_line + third_line

    @property
    def previous_date(self):
        return self.index[self.day - 1]

    def offset(self, adjustment):
        '''
        The day_offset accepts an integer offset of the number of 
        days to move the event.
        '''
        return self.index[self.day + adjustment]

    def update(self, positions):
        positions.type.loc[self.ticker] = self.Label
        positions.selected.loc[self.ticker] = TradeSelected.UNDECIDED


class EntryEvent(TradeEvent):
    Label = "entry"

    def update(self, positions):
        positions.target_size.loc[self.ticker] = self.size
        super().update(positions)
        

class ExitEvent(TradeEvent):
    Label = "exit"

    def update(self, positions):
        positions.target_size.loc[self.ticker] = 0
        super().update(positions)
        

class AdjustmentEvent(TradeEvent):
    Label = "adjustment"

    def update(self, positions):
        positions.target_size.loc[self.ticker] += self.size
        positions.type.loc[self.ticker] = self.Label
        if positions.applied_size[self.ticker] != 0:
            positions.selected.loc[self.ticker] = TradeSelected.UNDECIDED
        else:
            positions.selected.loc[self.ticker] = TradeSelected.NO
        
 