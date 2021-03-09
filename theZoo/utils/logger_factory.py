import logging


def get_logger(logger_name: str):
    logger = logging.getLogger(logger_name)
    return logger


class LoggerFactory:
    def __init__(self,
                 level=logging.DEBUG,
                 format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                 datefmt='%m-%d %H:%M',
                 filename='log/app.log',
                 filemode='w'):
        logging.basicConfig(level=level,
                            format=format,
                            datefmt=datefmt,
                            filename=filename,
                            filemode=filemode)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
