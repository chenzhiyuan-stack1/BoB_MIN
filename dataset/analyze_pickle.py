import pickle
import numpy as np

# Shape of each element:
#   - observations: (6534914, 150)
#   - actions: (6534914, 1)
#   - next_observations: (6534914, 150)
#   - rewards: (6534914,)
#   - terminals: (6534914,)

# pickle_path = "/home/min414/data1/Schaferct/training_dataset_pickle/v8.pickle"
pickle_path = 'BoB_012.pickle'
with open(pickle_path, 'rb') as f:
    data = pickle.load(f)

print("Keys in the pickle file:", data.keys())
print("\nShape of each element:")

for key, value in data.items():
    try:
        # 将每个元素转换为numpy数组并打印其形状
        np_array = np.array(value)
        print(f"  - {key}: {np_array.shape}")
    except Exception as e:
        print(f"  - {key}: Could not convert to numpy array. Type: {type(value)}, Error: {e}")
        
# print(data["observations"][1][44])  # 打印第一个观测值以进行检查