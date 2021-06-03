"""
clustering script, metrics, strategy, save to dir
"""
import os
import yaml
import torch
import pickle
import shutil
import numpy as np
import pandas as pd
import random
import sklearn.cluster as clst
import sklearn.metrics as mtrs
import torch.nn.functional as F
import time
import datetime

from kmeans_gpu import kMeans_gpu


def get_pkls(dir: str, norm: bool, num=99999):
    pcd_pkl = os.path.join(dir, 'data.pkl')
    if os.path.exists(pcd_pkl) and norm:
        ### normalized data
        f = open(pcd_pkl, 'rb')
        ret_dict = pickle.load(f)
        return np.array(ret_dict.keys()), np.array(list(ret_dict.values()), dtype=np.float32)

    folders = os.listdir(dir)
    feature_urls = []
    for i, folder in enumerate(folders[:num]):
        f = os.listdir(os.path.join(dir, folder))
        f = list(filter(lambda x: x.endswith('.pkl'), f))
        for _f in f:
            feature_urls.append(os.path.join(dir, folder, _f))
    res_dict = {}
    for i, furl in enumerate(feature_urls):
        print(i + 1, 'before: read', furl)
        f = open(furl, 'rb')
        data = pickle.load(f)
        res_dict.update(data)
    keys, values = res_dict.keys(), res_dict.values()
    keys, values = np.array(list(keys)), np.array(list(values), dtype=np.float32)
    if not norm:
        return keys, values
    values = torch.from_numpy(values)
    values = F.normalize(values, p=2, dim=1)
    # print(values[0])
    return keys, np.array(values, dtype=np.float32)


def gen_params(name: str, config: dict):
    if name == 'KMeans':
        KMeans_ks = config['KMeans_k']
        if isinstance(KMeans_ks, int):
            KMeans_ks = [KMeans_ks]
            return KMeans_ks
        return range(KMeans_ks[0], KMeans_ks[1], 1)

    elif name == 'AffinityPropagation':
        return [["None", "None"]]

    elif name == 'AgglomerativeClustering':
        AgglomerativeClustering_ks = config['AgglomerativeClustering_k']
        AgglomerativeClustering_types = config['AgglomerativeClustering_type']
        if isinstance(AgglomerativeClustering_ks, str):
            AgglomerativeClustering_ks = [AgglomerativeClustering_ks]
        else:
            AgglomerativeClustering_ks = range(AgglomerativeClustering_ks[0], AgglomerativeClustering_ks[1], 1)

        if isinstance(AgglomerativeClustering_types, str):
            AgglomerativeClustering_types = [AgglomerativeClustering_types]

        k_types = []
        for type in AgglomerativeClustering_types:
            for k in AgglomerativeClustering_ks:
                k_types.append([k, type])
        return k_types
    else:
        print('metrics error')
        exit()


def cluster(name: str, k: int, keys: np.array, values: np.array):
    if name == 'KMeans':
        if not isinstance(k, int):
            print('param error')
            exit()
        return kMeans(k, keys, values)

    elif name == 'AffinityPropagation':
        return affinityPropagation(keys, values)

    else:
        print('cluster name error')
        exit()
    # elif name=='AgglomerativeClustering':
    #     pass
    # pass


def kMeans(k: int, keys: np.array, values: np.array):
    # model = clst.KMeans(n_clusters=k, init='k-means++', random_state=777)
    model = kMeans_gpu(n_clusters=k, init='k-means++', verbose=True, random_state=777)
    model.fit(values)

    # metrics: silhouette_score,
    # CH分数（Calinski Harabasz Score ）,
    # 戴维森堡丁指数(DBI)——davies_bouldin_score

    pred_labels = np.array(model.labels_)
    pred_centers = np.array(model.cluster_centers_)
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d')} silhouette_scores calculation! ")
    silhouette_scores = mtrs.silhouette_samples(values, pred_labels, metric='euclidean')
    silhouette_scores = np.array(silhouette_scores)
    ss_dict = {}
    for label in set(pred_labels):
        filtered_index = np.where(pred_labels == label)
        if len(np.array(filtered_index).reshape(-1)) < 50: continue
        filtered_sscores = silhouette_scores[filtered_index]
        filtered_keys = keys[filtered_index]

        filtered_values = values[filtered_index]
        distances = np.sqrt(np.sum(np.asarray(filtered_values - pred_centers[label]) ** 2, axis=1))
        top80_keys = filtered_keys[np.argpartition(distances, int(.8 * len(distances)))]
        # score, imgs, top80imgs
        ss_dict[label] = [np.mean(filtered_sscores), filtered_keys, top80_keys]

    # default: 升序
    ss_dict = dict(sorted(ss_dict.items(), key=lambda kv: kv[1][0], reverse=True))

    print(f"{datetime.datetime.now().strftime('%Y-%m-%d')} ch_score calculation! ")
    ch_score = mtrs.calinski_harabasz_score(values, pred_labels)
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d')} dbi_score calculation! ")
    dbi_score = mtrs.davies_bouldin_score(values, pred_labels)

    return ss_dict, ch_score, dbi_score


def affinityPropagation(keys: np.array, values: np.array):
    model = clst.AffinityPropagation().fit(values)
    pred_labels = np.array(model.labels_)
    pred_centers = np.array(model.cluster_centers_)
    silhouette_scores = mtrs.silhouette_samples(values, pred_labels, metric='euclidean')
    silhouette_scores = np.array(silhouette_scores)
    ss_dict = {}
    for label in set(pred_labels):
        filtered_index = np.where(pred_labels == label)
        if len(np.array(filtered_index).reshape(-1)) < 50: continue
        filtered_sscores = silhouette_scores[filtered_index]
        filtered_keys = keys[filtered_index]

        filtered_values = values[filtered_index]
        distances = np.sqrt(np.sum(np.asarray(filtered_values - pred_centers[label]) ** 2, axis=1))
        top80_keys = filtered_keys[np.argpartition(distances, int(.8 * len(distances)))]
        # score, imgs, top80imgs
        ss_dict[label] = [np.mean(filtered_sscores), filtered_keys, top80_keys]

    # default: 升序
    ss_dict = sorted(ss_dict.items(), key=lambda kv: kv[1][0], reverse=True)

    ch_score = mtrs.calinski_harabasz_score(values, pred_labels)
    dbi_score = mtrs.davies_bouldin_score(values, pred_labels)

    return ss_dict, ch_score, dbi_score


def dataProcess(data:dict, imgs_dir:str, config:dict, fun_name:str, param:str, interval:str):
    ss_dict, ch_score, dbi_score = data
    scores = np.array(list(map(lambda x:x[0], ss_dict.values())))
    top20 = np.mean(scores[:int(.2*len(scores))])
    top50 = np.mean(scores[:int(.5*len(scores))])
    top80 = np.mean(scores[:int(.8*len(scores))])
    top100 = np.mean(scores)

    save_data = {
        "pickle_dir": [config['pickle_dir']],
        "normalize": [config['norm']],
        "function":[fun_name],
        "parameters":[param],
        "ss_top20p":[top20],
        "ss_top50p":[top50],
        "ss_top80p":[top80],
        "ss_top100p":[top100],
        "ch_score":[ch_score],
        "dbi_score":[dbi_score],
        "interval":[str(interval)],
    }
    df_data = pd.DataFrame(save_data)

    if not os.path.exists(os.path.join(config['base_dir'], 'cluster')):
        os.makedirs(os.path.join(config['base_dir'], 'cluster'))

    # result.csv
    res_dir = os.path.join(config['base_dir'], 'cluster', config['result_csv'])
    print(res_dir)
    if os.path.exists(res_dir):
        df_data.to_csv(res_dir, mode='a', header=False, index=False)
    else:
        df_data.to_csv(res_dir, mode='w', header=True, index=False)

    # detail
    mode = 'a' if os.path.exists(res_dir) else 'w'
    scores4txt = list(map(lambda x:str(x), scores))
    txt = ",".join(scores4txt)
    with open(os.path.join(config['base_dir'], 'cluster',config['detail_txt']), mode) as f:
        f.write(txt+'\n')

    # save images
    for k,v in ss_dict.items():
        imgs, filtered_imgs = v[1], v[2]
        save_dir = os.path.join(imgs_dir, 'origin', str(v[0])+'_'+str(k))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for img in random.sample(list(imgs), min(200, len(imgs))):
            shutil.copy(img, save_dir)

        save_dir = os.path.join(imgs_dir, 'processed', str(v[0]) + '_' + str(k))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for img in random.sample(list(imgs), min(200, len(filtered_imgs))):
            shutil.copy(img, save_dir)

    print(fun_name, param, 'finish!')


def run(config: dict):
    keys, values = get_pkls(config['pickle_dir'], config['norm'])
    cluster_funcs = config['cluster_funcs']

    if isinstance(cluster_funcs, str):
        cluster_funcs = [cluster_funcs]

    for func in cluster_funcs:
        for param in gen_params(func, config):
            param_name = "_".join(param) if isinstance(param, list) else str(param)
            img_dir = os.path.join(config['base_dir'], 'cluster', func, param_name)
            if os.path.exists(img_dir):
                continue
            else:
                print(img_dir)
                os.makedirs(img_dir)
            start = time.time()
            ans = cluster(func, param, keys, values)
            interval = time.time() - start
            print('interval:', interval)
            exit()
            dataProcess(ans, img_dir, config, func, param_name, interval)


if __name__ == '__main__':
    with open('cfg.yaml', 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    run(config)
