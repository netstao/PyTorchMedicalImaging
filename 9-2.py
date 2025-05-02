import datetime
from util.util import importstr
from util.logconf import logging
log = logging.getLogger('nb')


class LunaTrainingApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]
            
        parser = argparse.ArgumentParser()
        parser.add_argument('--num-workers',
                           help='Number of worker processes  for background data loading',
                           default=8,
                           type=int,)
        
        self.cli_args = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S')
        

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

def run(app, *argv):
    argv = list(argv)
    argv.insert(0, '--num-workers=4')
    log.info("Running: {}({!r}).main()".format(app,argv))
    
    app_cls = importstr(*app.rsplit('.',1))
    app_cls(argv).main()
    
    log.info("Finished: {}({!r}).main()".format(app, argv))

run('LunaTrainingApp','--num-workers=4')
if __name__ == '__main__':
    LunaTrainingApp.main()