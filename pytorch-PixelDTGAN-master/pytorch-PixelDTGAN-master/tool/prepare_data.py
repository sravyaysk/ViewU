import six.moves.cPickle as Pickle
import os


dataset_dir = '/home/pramati/sravya/ViewU/lookbook/data/'
models = []
clothes = []
users = []

for filename in os.listdir(dataset_dir):
    if filename.endswith('.jpg'):
        if filename.split('_')[1].endswith('0'):
            models.append(filename)
        elif filename.split('_')[1].endswith('2'):
            users.append(filename)
        else:
            clothes.append(filename)

print(len(models))
print(len(clothes))
print(len(users))

i = 0
match = []
user_match = []
while i < len(clothes):
    pid = clothes[i][3:9]

    match_i = []
    j = 0
    while j < len(models):
        if models[j][3:9] == pid:
            match_i.append(models[j])
        j += 1
    match.append(match_i)

    user_match_i = []
    k = 0
    while k < len(users):
        if users[k][3:9] == pid:
            user_match_i.append(users[k])
        k += 1
    user_match.append(user_match_i)

    i += 1
# print("model table: ",sorted(match)[0:10])
# print("user table: ",sorted(user_match)[0:10])
# print("cloth_table: ",sorted(clothes)[0:10])

# for i in user_match:
#     if(len(i)>0):
#         print(i)

with open('cloth_table.pkl', 'wb') as cloth_table:
    Pickle.dump(clothes, cloth_table)
with open('model_table.pkl', 'wb') as model_table:
    Pickle.dump(match, model_table)
with open('user_table.pkl', 'wb') as user_table:
    Pickle.dump(user_match,user_table)

print('done')
