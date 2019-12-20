'''
Written by Whalechen
'''

import logging

logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG, 
    filename= './loss_result.log',
    filemode= 'w')

log = logging.getLogger()