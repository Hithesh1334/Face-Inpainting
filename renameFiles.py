import os

#Code to rename the files accordingly
path = "E:/CSE/finalDataset/"
# s = os.listdir(path)
# print(len(s))
count = 1
for i in os.listdir(path):
    innerPath = path + i + "/"
    for j in os.listdir(innerPath):
        oldFileName = innerPath + j
        newFileName = path + "mergedDataset/" + "00" + str(count) +".jpg"
        count += 1
        os.rename(oldFileName, newFileName)


