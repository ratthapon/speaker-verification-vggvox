import glob
import re
import shutil
from pathlib import Path

import fire
import pandas
from pandas import DataFrame


def sample_voxceleb2(
        N_SPEAKER: int,
        N_SAMPLE_PER_SPEAKER: int,
        SOURCE_DIRECTORY: str,
        DEST_DIRECTORY: str,
        DEST: str = None,
        EXTENDING: bool = False,
        RANDOM: bool = False,
        SORTING: str = "asc",
        GLOB_PATTERN: str = "*/*/*.m4a",
        DRY: bool = True,
        FLAT_DIRECTORY: bool = True,
        COLUMNS: list = None,
        bootstrap_replace: bool = False,
        id_pattern: str = r"(id\d+)"
):
    """
    Sampling N_SPEAKER and N_SAMPLE_PER_SPEAKER data from VoxCeleb2 dataset's SOURCE_DIRECTORY and copy to DEST_DIRECTORY.
    :param N_SPEAKER:
    :param N_SAMPLE_PER_SPEAKER:
    :param SOURCE_DIRECTORY:
    :param DEST_DIRECTORY:
    :param DEST:
    :param EXTENDING:
    :param RANDOM:
    :param GLOB_PATTERN:
    :param DRY:
    :param FLAT_DIRECTORY:
    :param COLUMNS:
    :param bootstrap_replace:
    :param id_pattern:
    :return:
    """
    # default kwargs
    if COLUMNS is None:
        COLUMNS = ["id", "filename", "origin_filepath"]

    # list all files in directory and extract meta info to DataFrame
    print(SOURCE_DIRECTORY + GLOB_PATTERN)
    path_list = glob.glob(SOURCE_DIRECTORY + GLOB_PATTERN)
    if not RANDOM and SORTING == "asc":
        path_list.sort()
    elif not RANDOM and SORTING == "desc":
        path_list.sort(reverse=True)

    id_list = []
    filename_list = []
    origin_filename_list = []
    for filename in path_list:
        id = re.search(id_pattern, filename).group(1)
        preprocessed_filename = ""
        if FLAT_DIRECTORY:
            preprocessed_filename = filename.replace(SOURCE_DIRECTORY, "").replace("/", "_")
        else:
            preprocessed_filename = filename.replace(SOURCE_DIRECTORY, "")

        id_list += [id]
        filename_list += [preprocessed_filename]
        origin_filename_list += [filename]

    # preload existing dataset if exists to append
    dataset = None
    if EXTENDING and Path(DEST).exists():
        # test dest exists, reuse dest file
        dataset = pandas.read_csv(DEST)
        extending_dataset = DataFrame({
            COLUMNS[0]: id_list,
            COLUMNS[1]: filename_list,
            COLUMNS[2]: origin_filename_list
        })
        dataset = dataset.append(extending_dataset)
    else:
        # dest not exists, create new dest file
        dataset = DataFrame({
            COLUMNS[0]: id_list,
            COLUMNS[1]: filename_list,
            COLUMNS[2]: origin_filename_list
        })

    # sampling speaker and speech
    unique_ids = list(dataset[COLUMNS[0]].unique())
    unique_ids.sort()
    selected_ids = unique_ids[0:N_SPEAKER]
    dataset = dataset[dataset[COLUMNS[0]].isin(selected_ids)]
    dataset = dataset.groupby(COLUMNS[0]).apply(lambda x: x.sample(N_SAMPLE_PER_SPEAKER))

    # Copy file if not a DRY run
    if not DRY:
        for idx, row in dataset.iterrows():
            id = row[COLUMNS[0]]
            src = row[COLUMNS[2]]
            dest = Path(DEST_DIRECTORY, id, row[COLUMNS[1]])
            dest.parent.mkdir(exist_ok=True, parents=True)
            shutil.copy2(src, dest)

    # return result to DEST file or stdout
    if DEST is not None:
        dataset.to_csv(DEST, columns=COLUMNS[0:2], index=False)
    else:
        # this may pipe in terminal directly
        # python3 cmd/prepare_voxceleb ${PARAMS} > ${DEST}
        return dataset.to_csv(columns=COLUMNS[0:2], index=False)


def voxceleb2_303ids_5samples():
    # sample training set
    import os

    HOME = os.getenv("HOME")
    out1 = sample_voxceleb2(
        N_SPEAKER=303,
        N_SAMPLE_PER_SPEAKER=5,
        SOURCE_DIRECTORY=f"{HOME}/dataset/VoxCeleb2_simple10/dev/aac/",
        DEST_DIRECTORY=f"{HOME}/dataset/VoxCeleb2_303id5sample/ref/",
        FLAT_DIRECTORY=True,
        DRY=False
    )

    out2 = sample_voxceleb2(
        N_SPEAKER=303,
        N_SAMPLE_PER_SPEAKER=5,
        SOURCE_DIRECTORY=f"{HOME}/dataset/VoxCeleb2_simple10/dev/aac/",
        DEST_DIRECTORY=f"{HOME}/dataset/VoxCeleb2_303id5sample/eval/",
        SORTING="desc",
        FLAT_DIRECTORY=True,
        DRY=False
    )
    print(out1)
    print(out2)

if __name__ == "__main__":
    fire.Fire({
        "sample_voxceleb2": sample_voxceleb2,
        "voxceleb2_303ids_5samples": voxceleb2_303ids_5samples
    })
