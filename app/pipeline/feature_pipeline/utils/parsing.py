import requests

token = "eyJ0eXBlIjoiSldUIiwiYWxnIjoiSFM1MTIifQ.eyJqdGkiOiI1MDE0ODk4Iiwicm9sIjoiUk9MRV9SRUdJU1RFUiIsImlzcyI6Ik9wZW5YTGFiIiwiaWF0IjoxNzYyNTA0MzA4LCJjbGllbnRJZCI6ImxremR4NTdudnkyMmprcHE5eDJ3IiwicGhvbmUiOiIiLCJvcGVuSWQiOm51bGwsInV1aWQiOiI4NDIwOTg4MS05Y2M3LTRkMGUtODA0MC0zZmU2ODc3N2MxMzIiLCJlbWFpbCI6IiIsImV4cCI6MTc2MzcxMzkwOH0.syDlP81g1e8xlchoI_KDYLynv5nLiT-oFzxG2ev1n5gmaF34u4D-9WfzYcykf5LEbimzd3JJN1HFY-pooqGPjw"
url = "https://mineru.net/api/v4/extract/task"
header = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}"
}
data = {
    "url": "https://cdn-mineru.openxlab.org.cn/demo/example.pdf",
    "model_version": "vlm"
}

res = requests.post(url,headers=header,json=data)
print(res.status_code)
print(res.json())
print(res.json()["data"])