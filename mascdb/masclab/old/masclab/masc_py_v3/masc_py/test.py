from masc_py.core import LabelParams, ProcessParams, MASC_process

label = LabelParams(
    campaigndir="/Raw_data",
    output_dir="/output",
    starthr_vec=(2017,12,1,0,0,0),
    endhr_vec=(2017,12,2,0,0,0),
)
process = ProcessParams(use_triplet_algo=True)
results = MASC_process(label, process)
print(len(results), "images traitées")
