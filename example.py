# -*- coding: utf-8 -*-
# 引入相关模块
import utils.load_dataset as ld
import member_generation.library_generation as lg
import utils.io_func as io
import utils.settings as settings
import ensemble.Cluster_Ensembles as ce
import evaluation.Metrics as metrics

# 导入数据集，目前我有使用过的数据集都在utils.load_dataset模块中封装成函数了
# 调用时返回两个量，其一是特征矩阵，尺寸是(#Instances * #Features)
# 部分数据集内置了一些参数，如进行0-1规范化等，自行查看
name = 'Iris'
d, t = ld.load_iris()

# 生成聚类成员集的函数我都包装在member_generation.library_generation中
# 主要函数是generate_library，它可以提供random-subspace的聚类成员集生成
# 以及半监督的聚类成员集(目前有E2CP和COP_KMEANS两种半监督聚类方法)的生成
# 该函数返回的是library的名字，实际library保存在Results/[数据集名字]/
# 具体参数，请见函数注释说明，可能写的比较乱，如果不太明白再问
# p.s. 如果使用random-subspace的方式生成聚类成员，其生成方法主要是对样本or特征的随机采样
# 具体函数封装在member_generation.subspace中，library_generation进行调用
lib_name = lg.generate_library(d, t, name, 10, 3)

# 根据名字读入，这里读进来的是一个(#members * #instances)的矩阵
lib = io.read_matrix(settings.default_library_path + name + '/' + lib_name)

# 进行ensemble，这里ensemble返回的是集成之后的簇标签
ensemble_result = ce.cluster_ensembles_CSPAONLY(lib, N_clusters_max=3)

# print出来看看
print ensemble_result
print metrics.normalized_max_mutual_info_score(t, ensemble_result)
