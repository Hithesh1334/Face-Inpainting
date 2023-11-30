import os

#Code to rename the files accordingly
path = "E:/CSE/finalDataset/femasked9/"
s = os.listdir(path)
print(len(s))
count = 1
for i in os.listdir(path):
    # innerPath = path + i + "/"
    # for j in os.listdir(innerPath):
    oldFileName = path + i
    newFileName = path  + "100" + str(count) +".jpg"
    count += 1
    os.rename(oldFileName, newFileName)


