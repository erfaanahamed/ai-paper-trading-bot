import time,logging
from .runner import scan_all_markets
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
def main():
 while True:
  start=time.monotonic()
  for event in scan_all_markets():logging.info('%s',event)
  time.sleep(max(1,60-(time.monotonic()-start)))
if __name__=='__main__':main()
