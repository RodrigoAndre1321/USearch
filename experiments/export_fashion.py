import h5py
import numpy as np
from pathlib import Path

# Archivo original de ANN-Benchmarks
origen = Path(
    r"D:\PROYECTOEDAD\ann_datasets\fashion-mnist-784-euclidean.hdf5"
)

# Carpeta donde guardaremos los binarios
destino = Path(r"D:\PROYECTOEDAD\ann_datasets\fashion_bin")

destino.mkdir(parents=True, exist_ok=True)

Q = 1000
K = 10

with h5py.File(origen, "r") as archivo:

    # Vectores utilizados para construir HNSW
    train = np.ascontiguousarray(
        archivo["train"][:],
        dtype="<f4"
    )

    # Primeras 100 consultas
    test = np.ascontiguousarray(
        archivo["test"][:Q],
        dtype="<f4"
    )

    # Ground truth de las primeras 100 consultas
    neighbors = np.ascontiguousarray(
        archivo["neighbors"][:Q, :K],
        dtype="<i4"
    )

    train.tofile(destino / "train.bin")
    test.tofile(destino / "test.bin")
    neighbors.tofile(destino / "neighbors.bin")

    print("Conversion terminada")
    print("Train:", train.shape)
    print("Test:", test.shape)
    print("Neighbors:", neighbors.shape)