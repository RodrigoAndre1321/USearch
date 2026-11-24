#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>

using namespace std;
using namespace unum::usearch;
constexpr size_t HILOS = 4;
constexpr size_t N_TOTAL = 9990000;
constexpr size_t Q = 100;
constexpr size_t DIM = 96;
constexpr size_t K = 10;
constexpr size_t BLOQUE = 10000;

using Clock = chrono::steady_clock;

const string CARPETA =
    R"(D:\PROYECTOEDAD\ann_datasets\deep1b_bin)";

struct Resultado {
    string algoritmo;
    double parametro;
    double recall;
    double latencia_us;
    double distancias;
};

// Leer archivos binarios pequenos:
// consultas y vecinos exactos.
template <typename T>
bool leerBinario(
    const string& ruta,
    vector<T>& datos,
    size_t cantidad
) {
    ifstream archivo(ruta, ios::binary | ios::ate);

    if (!archivo)
        return false;

    const auto bytes =
        static_cast<streamsize>(cantidad * sizeof(T));

    if (archivo.tellg() != bytes)
        return false;

    archivo.seekg(0);
    datos.resize(cantidad);

    archivo.read(
        reinterpret_cast<char*>(datos.data()),
        bytes
    );

    return static_cast<bool>(archivo);
}

double calcularRecall(
    const vector<size_t>& aproximados,
    const int32_t* exactos
) {
    size_t aciertos = 0;

    for (size_t id : aproximados) {
        for (size_t j = 0; j < K; j++) {
            if (id == static_cast<size_t>(exactos[j])) {
                aciertos++;
                break;
            }
        }
    }

    return static_cast<double>(aciertos) / K;
}


bool construirIndice(
    index_dense_t& index,
    size_t cantidad
) {
    ifstream archivo(
        CARPETA + "\\train.bin",
        ios::binary
    );

    if (!archivo) {
        cerr << "No se pudo abrir train.bin\n";
        return false;
    }

    vector<float> bloque(BLOQUE * DIM);

    size_t insertados = 0;

    while (insertados < cantidad) {

        size_t restantes = cantidad - insertados;
        size_t leer = min(BLOQUE, restantes);

        const streamsize bytes =
            static_cast<streamsize>(
                leer * DIM * sizeof(float)
            );

        archivo.read(
            reinterpret_cast<char*>(bloque.data()),
            bytes
        );

        if (!archivo) {
            cerr << "Error leyendo train.bin\n";
            return false;
        }

        // Control de errores entre hilos
        std::atomic<bool> errorInsercion{false};
        std::atomic<size_t> idError{0};

        // Crear los trabajadores
        std::vector<std::thread> trabajadores;
        trabajadores.reserve(HILOS);

        for (size_t hilo = 0; hilo < HILOS; ++hilo) {

            trabajadores.emplace_back([&, hilo]() {

                const size_t comienzo =
                    (leer * hilo) / HILOS;

                const size_t final =
                    (leer * (hilo + 1)) / HILOS;

                // Cada hilo procesa su rango
                for (size_t j = comienzo; j < final; ++j) {

                    if (errorInsercion.load())
                        break;

                    const float* vector =
                        bloque.data() + j * DIM;

                    const size_t id = insertados + j;

                    // Tercer argumento: identificador del hilo
                    auto resultado = index.add(
                        id,
                        vector,
                        hilo
                    );

                    if (!resultado) {
                        idError.store(id);
                        errorInsercion.store(true);
                        break;
                    }
                }
            });
        }

        // Esperar a que terminen todos los hilos
        for (auto& trabajador : trabajadores) {
            trabajador.join();
        }

        if (errorInsercion.load()) {
            cerr << "Error insertando ID "
                 << idError.load() << "\n";
            return false;
        }

        insertados += leer;

        if (insertados % 100000 == 0 ||
            insertados == cantidad) {

            cerr << "Insertados: "
                 << insertados << "/"
                 << cantidad << "\n";
        }
    }

    return true;
}

// Evaluar una configuracion.
Resultado evaluar(
    index_dense_t& index,
    const vector<float>& consultas,
    const vector<int32_t>& vecinosExactos,
    bool usarABS,
    double parametro
) {
    double sumaRecall = 0;
    double sumaLatencia = 0;
    double sumaDistancias = 0;

    if (!usarABS) {
        index.change_expansion_search(
            static_cast<size_t>(parametro)
        );
    }

    for (size_t i = 0; i < Q; i++) {

        const float* consulta =
            consultas.data() + i * DIM;

        auto inicio = Clock::now();

        auto resultado = usarABS
            ? index.search_abs(consulta, K, parametro)
            : index.search(consulta, K);

        auto fin = Clock::now();

        if (!resultado || resultado.size() != K) {
            throw runtime_error(
                "Busqueda fallida en consulta " +
                to_string(i)
            );
        }

        vector<size_t> aproximados;
        aproximados.reserve(K);

        for (size_t j = 0; j < resultado.size(); j++) {
            aproximados.push_back(
                static_cast<size_t>(
                    resultado[j].member.key
                )
            );
        }

        const int32_t* exactos =
            vecinosExactos.data() + i * K;

        sumaRecall += calcularRecall(
            aproximados,
            exactos
        );

        sumaLatencia +=
            chrono::duration<double, micro>(
                fin - inicio
            ).count();

        sumaDistancias += resultado.computed_distances;
    }

    return {
        usarABS ? "ABS" : "HNSW",
        parametro,
        sumaRecall / Q,
        sumaLatencia / Q,
        sumaDistancias / Q
    };
}

int main(int argc, char** argv) {

    if (argc != 2) {
        cerr << "Uso: benchmark_deep1b.exe cantidad\n";
        return 1;
    }

    size_t cantidad = 0;

    try {
        cantidad = stoull(argv[1]);
    }
    catch (...) {
        cerr << "Cantidad invalida\n";
        return 1;
    }

    if (cantidad == 0 || cantidad > N_TOTAL) {
        cerr << "Cantidad fuera de rango\n";
        return 1;
    }

    cerr << "DEEP1B: " << cantidad
         << " vectores, " << DIM
         << " dimensiones\n";

    // Construccion del indice
    metric_punned_t metric(
        DIM,
        metric_kind_t::l2sq_k,
        scalar_kind_t::f32_k
    );

    index_dense_t index = index_dense_t::make(metric);

    if (!(bool)index.try_reserve(index_limits_t(cantidad, HILOS))) {
        cerr << "No se pudo reservar memoria\n";
        return 1;
    }

    auto inicioConstruccion = Clock::now();

    if (!construirIndice(index, cantidad))
        return 1;

    auto finConstruccion = Clock::now();

    double segundos =
        chrono::duration<double>(
            finConstruccion - inicioConstruccion
        ).count();

    cerr << "Indice construido: "
         << index.size() << " vectores\n";
         size_t presentes = 0;

    for (size_t id = 0; id < cantidad; ++id) {
        if (index.contains(id)) {
            ++presentes;
        }
    }

    cerr << "IDs presentes: "
        << presentes << "/"
        << cantidad << "\n";
    cerr << "Tiempo de construccion: "
         << segundos << " segundos\n";

    // Prueba parcial: solo verificar que busca.
    const bool tieneGroundTruth =
    cantidad == 200000 ||
    cantidad == 1000000 ||
    cantidad == 4000000 ||
    cantidad == N_TOTAL;
    if (!tieneGroundTruth) {

        vector<float> consultas;

        if (!leerBinario(
                CARPETA + "\\test.bin",
                consultas,
                Q * DIM
            )) {
            cerr << "Error leyendo consultas\n";
            return 1;
        }

        auto resultado = index.search_abs(
            consultas.data(), K, 0.20
        );

        if (!resultado || resultado.size() != K) {
            cerr << "La busqueda parcial fallo\n";
            return 1;
        }

        cerr << "Busqueda ABS parcial correcta.\n";
        cerr << "Recall no calculado: "
             << "ground truth corresponde al indice completo.\n";

        return 0;
    }

    // Solo para el indice completo:
    // cargar consultas y ground truth.
    vector<float> consultas;
    vector<int32_t> vecinosExactos;
    string archivoVecinos =
    CARPETA + "\\neighbors_" +
    to_string(cantidad) + ".bin";

    // Compatibilidad con los archivos anteriores
    if (cantidad == 200000) {
        archivoVecinos =
            CARPETA + "\\neighbors_200k.bin";
    }
    else if (cantidad == 1000000) {
        archivoVecinos =
            CARPETA + "\\neighbors_1m.bin";
    }
    if (!leerBinario(
            CARPETA + "\\test.bin",
            consultas,
            Q * DIM
        ) ||
        !leerBinario(
            archivoVecinos,
            vecinosExactos,
            Q * K
        )) {

        cerr << "Error cargando consultas o vecinos\n";
        return 1;
    }

    const vector<size_t> expansiones = {
        10, 40, 100
    };

    const vector<double> gammasNormales = {
        0.0, 0.05, 0.10, 0.15, 0.20
    };

    const vector<double> gammasAltos = {
        0.25, 0.30, 0.40, 0.50
    };

    ofstream csv(
        "deep1b_k10_N" + to_string(cantidad) + ".csv"
    );

    if (!csv) {
        cerr << "No se pudo crear el CSV\n";
        return 1;
    }

    csv << "Algoritmo,N,K,Parametro,Recall,"
        << "Latencia_us,Distancias\n";

    csv << fixed << setprecision(6);

    auto guardar = [&](const Resultado& r) {

        csv << r.algoritmo << ","
            << cantidad << ","
            << K << ","
            << r.parametro << ","
            << r.recall << ","
            << r.latencia_us << ","
            << r.distancias << "\n";

        csv.flush();

        cerr << r.algoritmo
             << " parametro=" << r.parametro
             << " recall=" << r.recall
             << " distancias=" << r.distancias
             << "\n";
    };

    try {

        // HNSW original
        for (size_t expansion : expansiones) {

            guardar(evaluar(
                index,
                consultas,
                vecinosExactos,
                false,
                static_cast<double>(expansion)
            ));
        }

        // ABS con gamma normales
        for (double gamma : gammasNormales) {

            guardar(evaluar(
                index,
                consultas,
                vecinosExactos,
                true,
                gamma
            ));
        }

        // ABS con gamma altos
        for (double gamma : gammasAltos) {

            guardar(evaluar(
                index,
                consultas,
                vecinosExactos,
                true,
                gamma
            ));
        }
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    cerr << "Benchmark DEEP1B terminado.\n";

    return 0;
}