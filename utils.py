import json
import sqlite3
import pickle
from time import time

import dlib


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def Instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


# FacesModel.Instance()
@Singleton
class FacesModel(object):
    def __init__(self):
        self.db = sqlite3.connect('data/model.sqlite', check_same_thread=False)
        # The row factory class sqlite3.Row is used to access the columns of a query by name instead of by index
        self.db.row_factory = sqlite3.Row
        self.cursor = self.db.cursor()

        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS vectors (
              label  TEXT,
              vector TEXT,
              timestamp INTEGER);''')

        self.db.commit()

    def save_single_vector(self, label, vector):
        self.cursor.execute('INSERT INTO vectors (label, vector, timestamp) VALUES(?,?,?)',
                            (label, json.dumps(list(vector)), int(time())))

        self.db.commit()

    def save_multiple_vectors(self, label, vectors):
        for vec in vectors:
            self.save_single_vector(label, vec)

    def load(self):
        self.cursor.execute('SELECT label, vector FROM vectors')
        result = []
        for row in self.cursor:
            result.append([int(row['label']), dlib.vector(json.loads(row['vector']))])

        return result
