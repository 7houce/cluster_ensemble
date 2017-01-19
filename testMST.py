from MSTClustering import MSTClustering
import numpy as np
import os
import matplotlib.pyplot as plt

colors = ['dodgerblue', 'black', 'darkorange', 'magenta', 'darkcyan', 'goldenrod', 'mediumslateblue', 'khaki', 'saddlebrown', 'crimson']


def autoClustering(nidpath, pospath, savepath, distype='nid', cutoff_upper=0.6, cutoff_lower=0.1, interval=0.05):
    nidpath = os.path.expanduser(nidpath)
    for f in os.listdir(nidpath):
        fullpath = os.path.join(nidpath, f)
        if os.path.isfile(fullpath):
            fname = os.path.splitext(f)
            filename = fname[0].split('_' + distype)[0]
            dataset_name = filename.split('_')[0]
            if not os.path.isdir(savepath + dataset_name):
                os.mkdir(savepath + dataset_name)
            distanceMatrix = np.loadtxt(fullpath, delimiter=',')
            pos = np.loadtxt(pospath + filename + '_mds2d.txt', delimiter=',')
            cur_cutoff = cutoff_lower
            while cur_cutoff <= cutoff_upper:
                mstmodel = MSTClustering(cutoff_scale=cur_cutoff, min_cluster_size=5, metric='precomputed')
                mstmodel.fit(distanceMatrix[0:-4, 0:-4])
                clusters = np.unique(mstmodel.labels_)
                fig = plt.figure(1)
                plt.clf()
                for i in clusters:
                    xs = pos[0:-4][mstmodel.labels_ == i, 0]
                    ys = pos[0:-4][mstmodel.labels_ == i, 1]
                    ax = plt.axes([0., 0., 1., 1.])
                    if i != -1:
                        plt.scatter(xs, ys, c=colors[((int(i) + 1) % len(colors))], label='Clusters-' + str(i))
                    else:
                        plt.scatter(xs, ys, c=colors[((int(i) + 1) % len(colors))], label='Outliers')
                plt.scatter(pos[-4:-1, 0], pos[-4:-1, 1], c='blue', marker='D', label='Consensus')
                plt.scatter(pos[-1:, 0], pos[-1:, 1], c='red', marker='D', label='Real')
                plt.legend(loc='best', shadow=True)
                plt.savefig(savepath + dataset_name + '/' + filename + '_afterMST' + '_' + str(cur_cutoff) + '.png', format='png', dpi=240)
                cur_cutoff += interval
    return

autoClustering('Results/MST/nid/', 'Results/MST/pos/', 'Results/MST/mst/')
