import faulthandler


class ProfilingWindowController:
    def __init__(self):
        self.view = None

    def create_window(self, parent, view_cls):
        self.view = view_cls(parent, self)
        return self.view

    def dump_stack(self):
        with open('stacks.txt', 'w') as f:
            faulthandler.dump_traceback(f)
        self.view.set_message('Stack dumped to stacks.txt')

    def start_profiler(self):
        from scalene import scalene_profiler
        scalene_profiler.start()
        self.view.set_profiling_active(True)

    def end_profiler(self):
        from scalene import scalene_profiler
        scalene_profiler.stop()
        self.view.set_profiling_active(False)
