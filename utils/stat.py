import csv
import os


def do_performance_stat_in_folder(folder, stat_file_name):
    """
    merge all performances in specific folder into a csv file.

    :param folder:
    :param stat_file_name:
    """
    if not os.path.isdir(folder):
        raise Exception("first argument should be a folder")
    with open(folder + stat_file_name, 'wb') as stat_file:
        writer = csv.writer(stat_file)
        writer.writerow(['library_name', 'CSPA_Performance', 'HGPA_Performance',
                         'MCLA_Performance', 'Single-KMeans_Performance'])
        for f in os.listdir(folder):
            fname = os.path.splitext(f)
            if not fname[0].endswith('performance'):
                continue
            fullpath = os.path.join(folder, f)
            if os.path.isfile(fullpath):
                print 'doing stat for '+f
                line_holder = [fname[0]]
                with open(fullpath) as file_obj:
                    for line in file_obj:
                        line_holder.append(line)
                writer.writerow(line_holder)
    return


def write_to_file(array, folder, filename, col_names=None):
    """

    :param array:
    :param folder:
    :param filename:
    :param col_names:
    :return:
    """
    with open(folder + filename, 'wb') as out_file:
        writer = csv.writer(out_file)
        if col_names is not None:
            writer.writerow(col_names)
        for row in array:
            writer.writerow(row)
    return
