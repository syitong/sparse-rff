import time

class log:
    """
    This module is used to track the progress of events
    and write it into a log file.
    """
    def __init__(self,filepath,init_message):
        self._message = ''
        self._filepath = filepath
        self._progress = {'task':[init_message],'time':[time.process_time()]}
        print(self._progress['task'][-1]
              + ': {:.4f}'.format(self._progress['time'][-1]))

    @property
    def progress(self):
        return self._progress

    def time_event(self,message):
        self._progress['task'].append(message)
        self._progress['time'].append(time.process_time())
        print(self._progress['task'][-1]
              + ': {:.4f}'.format(self._progress['time'][-1]
                              - self._progress['time'][-2]))

    def record(self,message):
        self._message = message

    def save(self):
        progress = self._progress
        with open(self._filepath,'w') as logfile:
            for idx in range(1,len(progress['task'])):
                logfile.write(progress['task'][idx]
                              + ': {:.4f}\n'.format(progress['time'][idx]
                                                    - progress['time'][idx - 1]))
            logfile.write(self._message)
