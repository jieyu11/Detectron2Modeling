import json
f = open("example_anno.json")
data = json.load(f)
for key, value in data.items():
    print(key)

for itm in data["images"]:
    print(itm)
    break

print("----------------------")
for itm in data["annotations"]:
    print(itm)
    break

print("----------------------")
print(data["categories"])
