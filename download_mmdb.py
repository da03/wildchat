import os
import re
import requests
import gzip
import shutil
from io import BytesIO

# Constants
REPO = "wp-statistics/GeoLite2-City"
API_COMMITS_URL = f"https://api.github.com/repos/{REPO}/commits"
DOWNLOAD_DIR = os.path.expanduser("./geocity")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Fetch commit list from GitHub API
commits = requests.get(API_COMMITS_URL, params={"per_page": 100}).json()

# Regex to match commit messages like "Update DB to YYYY-MM-DD"
pattern = re.compile(r"Update DB to (\d{4}-\d{2}-\d{2})")

for commit in commits:
    message = commit['commit']['message']
    match = pattern.match(message)

    if match:
        date = match.group(1)
        commit_sha = commit['sha']
        print(f"Processing DB from date: {date}, commit: {commit_sha}")

        # Construct URL to raw mmdb.gz file at specific commit
        raw_url = f"https://raw.githubusercontent.com/{REPO}/{commit_sha}/GeoLite2-City.mmdb.gz"

        response = requests.get(raw_url)

        if response.status_code == 200:
            # Decompress mmdb.gz file
            gz_file = BytesIO(response.content)
            with gzip.open(gz_file, 'rb') as f_in:
                mmdb_filename = f"GeoLite2-City-{date}.mmdb"
                mmdb_path = os.path.join(DOWNLOAD_DIR, mmdb_filename)
                with open(mmdb_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

            print(f"Saved: {mmdb_filename}")
        else:
            print(f"Failed to download DB for date {date}. Status: {response.status_code}")

