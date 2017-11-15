import utils.load_dataset as ld
import member_generation.library_generation as lg
import utils.io_func as io
import utils.settings as settings
import ensemble.Cluster_Ensembles as ce
import evaluation.Metrics as metrics

name = 'Iris'
d, t = ld.load_iris()
lib_name = lg.generate_library(d, t, name, 10, 3)
lib = io.read_matrix(settings.default_library_path + name + '/' + lib_name)
ensemble_result = ce.cluster_ensembles_CSPAONLY(lib, N_clusters_max=3)
print ensemble_result
print metrics.normalized_max_mutual_info_score(t, ensemble_result)
