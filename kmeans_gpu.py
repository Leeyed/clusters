import numpy as np
import torch


def get_closest_dist(X:torch.tensor, centers:list):
    ans = []
    for center in centers:
        tmp = X-center
        ans.append(torch.norm(tmp, p='fro', dim=1).unsqueeze(0))
    # torch 1.8+
    # ans = torch.vstack(ans)
    ans = torch.cat(ans, dim=0)
    ret = torch.min(ans, dim=0)
    return ret.values, ret.indices


class kMeans_gpu:
    def __init__(self, n_clusters=8, init='kmeans++', max_iter=300,
                 verbose=False, random_state=777):
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.verbose = verbose
        self.random_state = random_state
        np.random.seed(random_state)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # temporary parameters
        self._centers = None
        self._dists = None
        self._labels = None
        self._validation = None
        self.n_iter_ = 0

        # output attrs
        self.cluster_centers_ = None
        self.labels_ = None
        # self.inertia_ = None

    def init_centers(self, X):
        """
        each point will be selected by the percent: P=(|V|^2)/{∑|V|^2}
        """
        np.random.seed(self.random_state)
        centers = [X[np.random.choice(range(0, len(X)))]]
        for index in range(1, self.n_clusters):
            # 每个点与最近一个聚类中心的距离
            distances, _ = get_closest_dist(X, centers)
            # update v0.0.1
            cumsum = torch.cumsum(distances**2, dim=0)
            pick = torch.sum(distances ** 2) * np.random.random()
            i = torch.where(cumsum - pick > 0)[0][0]
            centers.append(X[i])
            if self.verbose:
                print(f'NO.{index + 1}: centers[:5]:{X[i][:5]}')
            # speed up 速度较慢
            # for i, di in enumerate(distances):  # 轮盘法选出下一个聚类中心；
            #     total -= di ** 2
            #     if total > 0: continue
            #     centers.append(X[i])
            #     if self.verbose:
            #         print(f'NO.{index + 1}: centers[:5]:{X[i][:5]}')
            #     break
        return centers

    def X_check(self, X):
        if isinstance(X, np.ndarray):
            return torch.from_numpy(X.astype(np.float32)).to(self.device)
        elif isinstance(X, torch.Tensor):
            return X
        else:
            print(type(X), "input error")
            exit()

    def nearest_center(self, X):
        distances, indices = get_closest_dist(X, self._centers)
        # if self.verbose:
        #     print('centers', self._centers)
        # print(distances, indices)
        self._labels = indices
        if self._dists is not None:
            self._validation = torch.abs(torch.sum(distances) - torch.sum(self._dists))
        else:
            self._validation = torch.abs(torch.sum(distances))
        self._dists = distances

    def update_center(self, X):
        centers = []
        for label in range(self.n_clusters):
            filtered_index = torch.where(self._labels==label)
            cluster_samples = X[filtered_index]
            centers.append(torch.mean(cluster_samples, dim=0))
        self._centers = centers

    def fit(self, X):
        X = self.X_check(X)
        # 只有kmeans++
        self._centers = self.init_centers(X)
        while True:
            # 聚类标记
            self.nearest_center(X)
            # 更新中心点
            self.update_center(X)
            if self.verbose:
                print(f'iteration:{self.n_iter_}', self._validation)
            if torch.abs(self._validation) < 1e-3:
                break
            elif self.n_iter_ == self.max_iter:
                break
            self.n_iter_ += 1
        if self.verbose:
            print('calculation complete!')
        self.representative_sample()

    def representative_sample(self):
        self.cluster_centers_ = [center.cpu().numpy() for center in self._centers]
        self.labels_ = [label.numpy() for label in self._labels.cpu()]
        # self.inertia_ = None


if __name__ == '__main__':
    import yaml

    with open('cfg.yaml', 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

