import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp


def load_results(path, num_results=8):
    """
    Load N npy results and convert to N x K np array
    :NOTE: N is number of results, K is number of updates
    Example path: '.imgs/PPO avg reward/FetchReach-v1_avg_reward' (without .npy)
    
    :param num_results: int, N
    :param path: str, load path
    :return: N x K np array of results
    """
    results = []
    for i in range(num_results):
        load_path = path + '_{}.npy'.format(i)
        results.append(np.load(load_path))
    return np.array(results)


fetch_data = {}
reacher_data = {}

# load PPO
folder_path = './imgs/PPO avg reward'
fetch_data['Fetch PPO'] = load_results(os.path.join(folder_path, 'FetchReach-v1_avg_reward'))
reacher_data['Reacher PPO'] = load_results(os.path.join(folder_path, 'Reacher-v2_avg_reward'))

# load GAIL
root_folder_path = './imgs/GAIL avg reward'

print(os.listdir(root_folder_path))

dir_list = ['Fetch GAIL',
            'Fetch GAIL plus env rwd (prob 0.5)',
            'Fetch GAIL plus env rwd (prob decay from 0.5)',
            'Reacher GAIL',
            'Reacher GAIL plus env rwd (prob 0.5)',
            'Reacher GAIL plus env rwd (prob decay from 0.5)']

for folder_path in dir_list:
    if 'Fetch' in folder_path:
        fetch_data[folder_path] = load_results(os.path.join(root_folder_path, folder_path, 'traj 500', 'FetchReach-v1_avg_reward'))
    elif 'Reacher' in folder_path:
        reacher_data[folder_path] = load_results(os.path.join(root_folder_path, folder_path, 'traj 10', 'Reacher-v2_avg_reward'))
    else:
        raise IOError

print('Data loaded!!!')

# plot settings
legend = ['Expert', 'PPO', 'GAIL', 'GAIL w/ RS ($\epsilon=0.5$)', 'GAIL w/ RS ($\epsilon=0.5$ w/ decay)']
color_list = ['orange', 'tomato', 'mediumseagreen', 'dodgerblue']
plt.figure(figsize=(12, 5))

# Fetch
assert list(fetch_data.keys()) == ['Fetch PPO',
                                   'Fetch GAIL',
                                   'Fetch GAIL plus env rwd (prob 0.5)',
                                   'Fetch GAIL plus env rwd (prob decay from 0.5)']
# NOTE: expert -3
plt.subplot(122)

plt.plot(-3 * np.ones(list(fetch_data.values())[0].shape[1]), '--', c='steelblue')     # expert
for i, data in enumerate(fetch_data.values()):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    plt.plot(mean, c=color_list[i])
    plt.fill_between(np.arange(len(mean)), mean + std, mean - std, facecolor=color_list[i], alpha=0.35)


plt.xlabel('number of updates')
plt.ylabel('average reward')
plt.legend(legend, loc='lower right')
plt.title('FetchReach-v1')
plt.grid()

print('Fig saved for Fetch!!!')

# Reacher
assert list(reacher_data.keys()) == ['Reacher PPO',
                                     'Reacher GAIL',
                                     'Reacher GAIL plus env rwd (prob 0.5)',
                                     'Reacher GAIL plus env rwd (prob decay from 0.5)']
# NOTE: expert -3.68
plt.subplot(121)

plt.plot(-3.68 * np.ones(list(fetch_data.values())[0].shape[1]), '--', c='steelblue')     # expert
for i, data in enumerate(reacher_data.values()):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    plt.plot(mean, c=color_list[i])
    plt.fill_between(np.arange(len(mean)), mean + std, mean - std, facecolor=color_list[i], alpha=0.35)

plt.xlabel('number of updates')
plt.ylabel('average reward')
plt.legend(legend)
plt.title('Reacher-v2')
plt.grid()
plt.savefig('./imgs/All_result.png')

plt.show()

print('Fig saved for Reacher!!!')