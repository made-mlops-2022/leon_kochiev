import argparse
import requests

parser = argparse.ArgumentParser(
    prog="sender", description="send post request to listener"
)
parser.add_argument("url")
parser.add_argument("-d", "--data", required=False)

if __name__ == "__main__":
    args = parser.parse_args()
    resp = requests.post(args.url, data=args.data)
    print(resp.content.decode("utf-8"))
