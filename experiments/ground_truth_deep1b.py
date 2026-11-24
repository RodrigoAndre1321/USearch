import numpy as np
import sys
from pathlib import Path

carpeta = Path(
    r"D:\PROYECTOEDAD\ann_datasets\deep1b_bin"
)

N_TOTAL = 9990000

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000000

if not 10 <= N <= N_TOTAL:
    raise ValueError(
        f"N debe estar entre 10 y {N_TOTAL}"
    )
Q = 100
DIM = 96
K = 10
BLOQUE = 10000

print("Cargando consultas...")

consultas = np.fromfile(
    carpeta / "test.bin",
    dtype="<f4",
    count=Q * DIM
).reshape(Q, DIM)
# Leer train.bin sin cargarlo completamente en RAM.
train = np.memmap(
    carpeta / "train.bin",
    dtype="<f4",
    mode="r",
    shape=(9990000, DIM)
)

# Mejores distancias e identificadores encontrados.
mejores_distancias = np.full(
    (Q, K),
    np.inf,
    dtype=np.float32
)

mejores_ids = np.full(
    (Q, K),
    -1,
    dtype=np.int32
)

# Normas de las consultas
normas_consultas = np.sum(
    consultas * consultas,
    axis=1
)

print("Calculando vecinos exactos...")

for inicio in range(0, N, BLOQUE):

    fin = min(inicio + BLOQUE, N)

    datos = np.asarray(
        train[inicio:fin]
    )

    normas_datos = np.sum(
        datos * datos,
        axis=1
    )

    # Distancias L2 al cuadrado
    distancias = (
        normas_datos[:, None]
        + normas_consultas[None, :]
        - 2.0 * (datos @ consultas.T)
    )

    distancias = np.maximum(distancias, 0.0)

    ids_bloque = np.arange(
        inicio,
        fin,
        dtype=np.int32
    )

    for q in range(Q):

        # K mejores candidatos del bloque actual
        candidatos = np.argpartition(
            distancias[:, q],
            K - 1
        )[:K]

        nuevos_ids = ids_bloque[candidatos]

        nuevas_distancias = distancias[
            candidatos,
            q
        ]

        # Combinar mejores anteriores y nuevos
        todos_ids = np.concatenate((
            mejores_ids[q],
            nuevos_ids
        ))

        todas_distancias = np.concatenate((
            mejores_distancias[q],
            nuevas_distancias
        ))

        # Ordenar por distancia; ID como desempate
        orden = np.lexsort((
            todos_ids,
            todas_distancias
        ))[:K]

        mejores_ids[q] = todos_ids[orden]

        mejores_distancias[q] = todas_distancias[orden]

    print(f"Procesados: {fin}/{N}")

archivo_salida = carpeta / f"neighbors_{N}.bin"

mejores_ids.astype("<i4").tofile(
    archivo_salida
)

print("Archivo generado:", archivo_salida)

print("\nGround truth generado")
print("Forma:", mejores_ids.shape)
print("Primera consulta:", mejores_ids[0])