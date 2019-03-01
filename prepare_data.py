import glob

def scandir(directory):
    i, t = 0, {}
    pfile = glob.glob(directory)
    for filename in pfile:
        i = i + 1
        t[i] = filename
    return t


clothes = scandir('E:/work/ML/infinity/lookbook/data/*CLEAN1*')
models = scandir('E:/work/ML/infinity/lookbook/data/*CLEAN0*')
nmodels = len(models)
nclothes = len(clothes)
i, j = 1, 1
print(nclothes)
print(nmodels)

match = {}

while((i <= nclothes) and (j <= nmodels)):
    #print(i,nclothes,j,nmodels)
    pos = clothes[i].find('PID')
    pid = clothes[i][pos:pos + 9]
    #print(clothes[i],pos,pid)
    k = 0
    match[i] = {}
    print(models[j], pid, models[j].find(pid))
    while((j <= nmodels) and (models[j].find(pid) != -1)):
        #print("inside :D")
        match[i][k] = models[j]
        j = j + 1
        k = k + 1
    i = i + 1

torch.save('cloth_table.t7', clothes)
torch.save('models_table.t7', match)