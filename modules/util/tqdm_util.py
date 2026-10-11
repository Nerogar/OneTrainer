from tqdm import tqdm as _tqdm

#progress bars created through the tqdm below, innermost last
_bars = []


class tqdm(_tqdm):
    _status = None

    @classmethod
    def get_lock(cls):
        #tqdm caches the terminal write lock on the class that first asks for it, so a subclass
        #would get one of its own and stop serializing against bars drawn by tqdm itself.
        return _tqdm.get_lock()

    @classmethod
    def show_status(cls, message: str):
        #status of long-running work - a compile, an autotune sweep - goes into the innermost bar's
        #postfix rather than on a line of its own, and stands until the next postfix write replaces it.
        bar = next((bar for bar in reversed(_bars) if not bar.disable), None)
        if bar is None:
            cls.write(message)
        else:
            bar.set_postfix_str(message)
            bar._status = message

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _bars.append(self)

    def _clear_status(self, refresh=True):
        #anything written over the status - the training loop's loss - is left standing.
        if self._status is not None and self.postfix == self._status:
            self.set_postfix_str("", refresh=refresh)
        self._status = None

    def update(self, n=1):
        #the status describes work that was running while the bar stood still, so the step that
        #follows it is the point where it stops being current.
        self._clear_status(refresh=False)
        return super().update(n)

    def close(self):
        self._clear_status()
        super().close()
        #compared by identity: tqdm's __eq__ is by screen position, so a bar closed late by __del__
        #would drop whichever live bar has taken over its line.
        global _bars
        _bars = [bar for bar in _bars if bar is not self]
