import os

markDirPath = "../../data/mark/"
items = os.listdir(markDirPath)
filenameList = []
for names in items:
    if names.endswith(".mark"):
        filenameList.append(names)
print(filenameList)

sampleRate = 20000
for fileName in filenameList:
    file = open(markDirPath + fileName)
    while 1:
        lines = file.readlines(100000)
        if not lines:
            break
        for line in lines:
            # TODO do something
            pass