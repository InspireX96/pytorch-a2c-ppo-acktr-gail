import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
import seaborn as sns


fetch_data = {}
reacher_data = {}

# load PPO
folder_path = './PPO avg reward'
fetch_data['Fetch PPO'] = np.load(os.path.join(folder_path, 'FetchReach-v1_avg_reward.npy'))
reacher_data['Reacher PPO'] = np.load(os.path.join(folder_path, 'Reacher-v2_avg_reward.npy'))

# load GAIL
root_folder_path = './GAIL avg reward'

print(os.listdir(root_folder_path))

dir_list = ['Fetch GAIL',
            'Fetch GAIL plus env rwd (prob 0.5)',
            'Fetch GAIL plus env rwd (prob decay from 0.5)',
            'Reacher GAIL',
            'Reacher GAIL plus env rwd (prob 0.5)',
            'Reacher GAIL plus env rwd (prob decay from 0.5)']

for folder_path in dir_list:
    if 'Fetch' in folder_path:
        fetch_data[folder_path] = np.load(os.path.join(root_folder_path, folder_path, 'traj 500', 'FetchReach-v1_avg_reward.npy'))
    elif 'Reacher' in folder_path:
        reacher_data[folder_path] = np.load(os.path.join(root_folder_path, folder_path, 'traj 10', 'Reacher-v2_avg_reward.npy'))
    else:
        raise IOError

print('Data loaded!!!')

# plot
# mlp.style.use("seaborn-darkgrid")
legend = ['Expert', 'PPO', 'GAIL', 'GAIL w/ RS ($\epsilon=0.5$)', 'GAIL w/ RS ($\epsilon=0.5$ w/ decay)']
plt.figure(figsize=(12, 5))
# Fetch
# NOTE: expert -3
plt.subplot(122)
plt.rcParams['image.cmap']=plt.cm.get_cmap('Dark2')

plt.plot(-3 * np.ones(len(list(fetch_data.values())[0])), '--')
for data in fetch_data.values():
    plt.plot(data)

assert list(fetch_data.keys()) == ['Fetch PPO',
                                   'Fetch GAIL',
                                   'Fetch GAIL plus env rwd (prob 0.5)',
                                   'Fetch GAIL plus env rwd (prob decay from 0.5)']

plt.xlabel('number of updates')
plt.ylabel('average reward')
plt.legend(legend, loc='lower right')
plt.title('FetchReach-v1')
plt.grid()
# plt.savefig('FetchReach-v1_result.png')

# plt.show()

print('Fig saved for Fetch!!!')

# Reacher
# NOTE: expert -3.68
plt.subplot(121)

plt.plot(-3.68 * np.ones(len(list(reacher_data.values())[0])), '--')
for data in reacher_data.values():
    plt.plot(data)

assert list(reacher_data.keys()) == ['Reacher PPO',
                                     'Reacher GAIL',
                                     'Reacher GAIL plus env rwd (prob 0.5)',
                                     'Reacher GAIL plus env rwd (prob decay from 0.5)']

plt.xlabel('number of updates')
plt.ylabel('average reward')
plt.legend(legend)
plt.title('Reacher-v2')
plt.grid()
# plt.savefig('Reacher-v2_result.png')
plt.savefig('All_result.png')

plt.show()

print('Fig saved for Reacher!!!')