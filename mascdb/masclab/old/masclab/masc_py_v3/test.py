from masc_py.core import LabelParams, ProcessParams, MASC_process

label = LabelParams(
    campaigndir=r"C:\Users\monne\Documents\Travail\EPFL\LTE\masclab\masc_py_v3\Raw_data",
    output_dir=r"C:\Users\monne\Documents\Travail\EPFL\LTE\masclab\masc_py_v3\output",
    starthr_vec=(2020,10,1,0,0,0),
    endhr_vec=(2021,2,2,0,0,0),
)
process = ProcessParams(use_triplet_algo=True)
results = MASC_process(label, process)
print(len(results), "images traitees")
