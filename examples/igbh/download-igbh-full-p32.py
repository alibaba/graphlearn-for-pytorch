import os
import subprocess
from urllib.parse import urlparse, urlunparse

def download_url(input_url):
    input_url = input_url.strip()
    if input_url.endswith("/"):
        print(f"Skipping folder URL: {input_url}")
    else:
        # Replace "oss://graphlearn" with "https://graphlearn.oss-cn-hangzhou.aliyuncs.com"
        download_url = input_url.replace("oss://graphlearn", "https://graphlearn.oss-cn-hangzhou.aliyuncs.com")
        print(f"Download: {download_url}")

        parsed_url = urlparse(download_url)
        url_path = parsed_url.path
        start_index = url_path.find("igbh-full-partition-32")

        if start_index != -1:
            folder_path = url_path[start_index:]
            # Split the path by "/" and exclude the last part
            folder_parts = folder_path.split("/")[:-1]
            folder_path = "/".join(folder_parts)
        else:
            folder_path = None

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Use wget to download the URL
        wget_command = f"wget {download_url} -P {folder_path}"
        subprocess.call(wget_command, shell=True)

def process_line(line):
    # Find the position of "oss://"
    oss_start = line.find("oss://")

    # Extract the string starting with "oss://"
    if oss_start != -1:
        oss_url = line[oss_start:]
        download_url(oss_url)
        return 1
    else:
        print(f"No 'oss://' found in: {line}")
        return 0

if __name__ == "__main__":
    url = "https://graphlearn.oss-cn-hangzhou.aliyuncs.com/data/igbh/igbh-full-partition-32/igbh-full-p32-oss-url.txt"
    destination_path = "igbh-full-p32-oss-url.txt"
    # Run wget command to download the file
    wget_command = f"wget {url} -O {destination_path}"
    try:
        subprocess.run(wget_command, shell=True, check=True)
        print(f"Download file list {destination_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")

    with open(destination_path, "r") as input_file:
        processed = 0
        for line in input_file:
            print(f"Download file {processed+1}/{2198}")
            processed += process_line(line)
        assert processed == 2198 # 2198 objects including the directories
    print("Download completes!")
    




