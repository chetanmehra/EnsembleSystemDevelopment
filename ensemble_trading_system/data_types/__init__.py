
from pandas import DataFrame
from datetime import datetime

class Collection:
    '''
    The Collection class provides an interface for managing collections of
    specific items (e.g. trades, or events).
    It holds a list of 'items' and provides an interface for interacting with
    them.
    '''

    def __getitem__(self, key):
        if isinstance(key, datetime):
            return self.find(lambda e: e.date == key)
        elif isinstance(key, str):
            return self.find(lambda e: e.ticker == key)
        elif isinstance(key, int):
            return self.items[key]
        else:
            raise ValueError("key must be a date, ticker (string), or integer")            

    def __iter__(self):
        return self.items.__iter__()

    def __len__(self):
        return len(self.items)

    @property
    def count(self):
        return len(self.items)

    def as_list(self):
        return self.items

    def as_dataframe(self):
        data = [item.as_tuple() for item in self.items]
        return DataFrame(data, columns = self.items[0].tuple_fields)

    def index(self, item):
        return self.items.index(item)

    def find(self, condition):
        '''
        Accepts a callable condition object (e.g. lambda expression), which 
        must accept a CollectionItem. Returns a new Collection which meet the condition.
        '''
        new_items = [item for item in self.items if condition(item)]
        return self.copy_with(new_items)

    def apply(self, modifier):
        modified_trades = []
        for trade in self.items:
            new_trade = modifier(trade)
            if new_trade is not None:
                modified_trades.append(new_trade)
        return self.copy_with(modified_trades)

    def subset(self, selection):
        '''
        Returns a new collection containing only items in the selection.
        selection is typically a list of integers representing the index
        of the item to select, although a list of tickers will also work.
        '''
        new_items = [self[s] for s in selection]
        return self.copy_with(new_items)

    def copy_with(self, items):
        '''
        Collection classes need to implement copy_with to create new
        instances with revised items.
        '''
        raise NotImplementedError



class CollectionItem:

    def as_tuple(self):
        return tuple(getattr(self, name) for name in self.tuple_fields)
