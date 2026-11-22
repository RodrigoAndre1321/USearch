#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <cstdint>
#include <cmath>
#include <numeric>
#include <stdexcept>

#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>

using namespace std;
using namespace unum::usearch;

constexpr size_t N = 60000;
constexpr size_t Q = 1000;
constexpr size_t DIM = 784;
constexpr size_t KMAX = 10;

using Clock = chrono::steady_clock;

template <typename T>
bool leerBinario(const string& ruta, vector<T>& datos, size_t cantidad) {
    ifstream archivo(ruta, ios::binary | ios::ate);
    if (!archivo) { cerr << "No se pudo abrir: " << ruta << "\n"; return false; }
    const auto bytesEsperados = static_cast<streamsize>(cantidad * sizeof(T));
    if (archivo.tellg() != bytesEsperados) { cerr << "Tamano incorrecto: " << ruta << "\n"; return false; }
    archivo.seekg(0);
    datos.resize(cantidad);
    archivo.read(reinterpret_cast<char*>(datos.data()), bytesEsperados);
    return static_cast<bool>(archivo);
}

double calcularRecall(const vector<size_t>& aproximados, const int32_t* exactos, size_t k) {
    size_t aciertos = 0;
    for (size_t id : aproximados) {
        for (size_t j = 0; j < k; j++) {
            if (id == static_cast<size_t>(exactos[j])) { aciertos++; break; }
        }
    }
    return static_cast<double>(aciertos) / k;
}

// Estructura para acumular resultados de una configuracion
struct Resultado {
    string algoritmo;
    size_t k;
    string parametro_nombre;  // "expansion" o "gamma"
    double parametro_valor;
    double recall;
    double latencia_us;
    double distancias;
};

// Ejecuta una configuracion HNSW y devuelve el resultado
Resultado ejecutarHNSW(index_dense_t& index, const vector<float>& test,
                       const vector<int32_t>& neighbors, size_t k, size_t expansion) {
    index.change_expansion_search(expansion);

    double sumaRecall = 0.0, sumaLatencia = 0.0, sumaDistancias = 0.0;

    for (size_t i = 0; i < Q; i++) {
        const float* consulta = test.data() + i * DIM;
        auto inicio = Clock::now();
        auto resultado = index.search(consulta, k);
        auto fin = Clock::now();
        if (!resultado || resultado.size() != k) {
            cerr << "Error HNSW: "
                << "K=" << k
                << " Expansion=" << expansion
                << " Consulta=" << i << "\n";

            throw runtime_error("Fallo en benchmark HNSW");
        }
        vector<size_t> aproximados;
        aproximados.reserve(k);
        for (size_t j = 0; j < resultado.size(); j++)
            aproximados.push_back(static_cast<size_t>(resultado[j].member.key));

        const int32_t* exactos = neighbors.data() + i * KMAX;
        sumaRecall += calcularRecall(aproximados, exactos, k);
        sumaLatencia += chrono::duration<double, micro>(fin - inicio).count();
        sumaDistancias += resultado.computed_distances;
    }

    return {"HNSW", k, "expansion", static_cast<double>(expansion),
            sumaRecall / Q, sumaLatencia / Q, sumaDistancias / Q};
}

// Ejecuta una configuracion ABS
Resultado ejecutarABS(index_dense_t& index, const vector<float>& test,
                      const vector<int32_t>& neighbors, size_t k, double gamma) {
    double sumaRecall = 0.0, sumaLatencia = 0.0, sumaDistancias = 0.0;

    for (size_t i = 0; i < Q; i++) {
        const float* consulta = test.data() + i * DIM;
        auto inicio = Clock::now();
        auto resultado = index.search_abs(consulta, k, gamma);
        auto fin = Clock::now();
        if (!resultado || resultado.size() != k) {
            cerr << "Error ABS: "
                << "K=" << k
                << " Gamma=" << gamma
                << " Consulta=" << i << "\n";

            throw runtime_error("Fallo en benchmark ABS");
        }
        vector<size_t> aproximados;
        aproximados.reserve(k);
        for (size_t j = 0; j < resultado.size(); j++)
            aproximados.push_back(static_cast<size_t>(resultado[j].member.key));

        const int32_t* exactos = neighbors.data() + i * KMAX;
        sumaRecall += calcularRecall(aproximados, exactos, k);
        sumaLatencia += chrono::duration<double, micro>(fin - inicio).count();
        sumaDistancias += resultado.computed_distances;
    }

    return {"ABS", k, "gamma", gamma,
            sumaRecall / Q, sumaLatencia / Q, sumaDistancias / Q};
}

int main() {
    const string carpeta = R"(D:\PROYECTOEDAD\ann_datasets\fashion_bin)";

    vector<float> train, test;
    vector<int32_t> neighbors;

    cerr << "Cargando Fashion-MNIST...\n";
    if (!leerBinario(carpeta + "\\train.bin", train, N * DIM) ||
        !leerBinario(carpeta + "\\test.bin", test, Q * DIM) ||
        !leerBinario(carpeta + "\\neighbors.bin", neighbors, Q * KMAX)) {
        return 1;
    }
    cerr << "Dataset cargado.\n";

    // Construir HNSW
    metric_punned_t metric(DIM, metric_kind_t::l2sq_k, scalar_kind_t::f32_k);
    index_dense_t index = index_dense_t::make(metric);
    if (!(bool)index.try_reserve(N)) { cerr << "Error reservando\n"; return 1; }

    cerr << "Construyendo HNSW...\n";
    auto inicioC = Clock::now();
    for (size_t i = 0; i < N; i++) {
        const float* vector = train.data() + i * DIM;
        if (!index.add(i, vector)) { cerr << "Error insertando " << i << "\n"; return 1; }
    }
    auto finC = Clock::now();
    cerr << "Construccion: " << chrono::duration<double>(finC - inicioC).count() << " s\n";

    // ============ BARRIDOS FINOS ============
    // HNSW: expansions finos para trazar curva completa
    const vector<size_t> expansiones = {
        10, 20, 30, 40, 60, 80, 100
    };

    // ABS: gammas finos
    const vector<double> gammas = {
        0.0, 0.05, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30
    };
    const vector<size_t> valoresK = {1, 5, 10};

    vector<Resultado> resultados;

    // Barrido HNSW
    for (size_t k : valoresK) {
        for (size_t exp : expansiones) {
            auto r = ejecutarHNSW(index, test, neighbors, k, exp);
            resultados.push_back(r);
            cerr << "HNSW K=" << k << " exp=" << exp
                 << " Recall=" << r.recall
                 << " Lat=" << r.latencia_us << "us\n";
        }
    }

    // Barrido ABS
    for (size_t k : valoresK) {
        for (double g : gammas) {
            auto r = ejecutarABS(index, test, neighbors, k, g);
            resultados.push_back(r);
            cerr << "ABS  K=" << k << " gamma=" << g
                 << " Recall=" << r.recall
                 << " Lat=" << r.latencia_us << "us\n";
        }
    }

    // ============ EXPORTAR CSV ============
    // CSV 1: formato largo (para pandas/seaborn)
    {
        ofstream f("resultados_largo.csv");
        f << "Algoritmo,K,Parametro,Valor,Recall,Latencia_us,Distancias\n";
        f << fixed << setprecision(4);
        for (auto& r : resultados) {
            f << r.algoritmo << "," << r.k << ","
              << r.parametro_nombre << "," << r.parametro_valor << ","
              << r.recall << "," << r.latencia_us << ","
              << r.distancias << "\n";
        }
    }

    // CSV 2: por K, formato ancho para graficos rapidos
    for (size_t k : valoresK) {
        ofstream f("resultados_K" + to_string(k) + ".csv");
        f << "Algoritmo,Parametro,Valor,Recall,Latencia_us,Distancias\n";
        f << fixed << setprecision(4);
        for (auto& r : resultados) {
            if (r.k != k) continue;
            f << r.algoritmo << "," << r.parametro_nombre << ","
              << r.parametro_valor << ","
              << r.recall << "," << r.latencia_us << ","
              << r.distancias << "\n";
        }
    }

    // ============ TABLA RESUMEN EN CONSOLA ============
    cout << fixed << setprecision(4);
    cout << "\n=== RESUMEN ===\n";
    cout << "Algoritmo,K,Parametro,Valor,Recall,Latencia_us,Distancias\n";
    for (auto& r : resultados) {
        cout << r.algoritmo << "," << r.k << ","
             << r.parametro_nombre << "," << r.parametro_valor << ","
             << r.recall << "," << r.latencia_us << ","
             << r.distancias << "\n";
    }

    cerr << "\nCSV generados: resultados_largo.csv, resultados_K{1,5,10}.csv\n";
    return 0;
}