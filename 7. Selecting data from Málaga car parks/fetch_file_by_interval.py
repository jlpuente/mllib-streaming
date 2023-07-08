import sys
import time
import urllib.request

if len(sys.argv) != 4:
    print("Usage: python fetch_file_by_interval "
          "<URL to fetch> <streaming directory> <number of seconds>", file=sys.stderr)
    exit(-1)

url_to_fetch = sys.argv[1]
streaming_directory = sys.argv[2]
seconds_to_sleep = int(sys.argv[3])

while True:
    time_stamp = str(time.time())

    # Note: streaming_directory already ends with a slash ("\" in Windows or "/" in Unix)
    # The resulting new_file hence is something like "./streaming_directory/12345678.csv"
    new_file = streaming_directory + str(time_stamp).replace(".", "") + ".csv"
    urllib.request.urlretrieve(url_to_fetch, new_file)

    time.sleep(seconds_to_sleep)
