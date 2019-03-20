import six.moves.cPickle as Pickle
data_dir='/home/pramati/sravya/ViewU/lookbook/data/'
index_dir='../tool/'
with open(index_dir + 'cloth_table.pkl', 'rb') as cloth:
    cloth_table = Pickle.load(cloth)
with open(index_dir + 'model_table.pkl', 'rb') as model:
    model_table = Pickle.load(model)
with open(index_dir + 'user_table.pkl', 'rb') as user:
    user_table = Pickle.load(user)
# with open('tables.txt','w') as f:
#     f.write("Clothes_table: "+ " ".join(str(i) for i in cloth_table) + "\n")
#     f.write("model_table: "+ " ".join(str(i) for i in model_table) + "\n")
#     f.write("user_table: "+ " ".join(str(i) for i in user_table) + "\n")
print(sorted(cloth_table)[0:10])
print(sorted(model_table)[0:10])
user_table1 = [x for x in user_table if x]
user_table1 = (sorted(user_table1))
for i in range(0,len(user_table1)):
    print(user_table1[i])