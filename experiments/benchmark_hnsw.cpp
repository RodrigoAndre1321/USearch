
#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <numeric>
#include <cmath>
#include <stdexcept>

#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>

using namespace std;
using namespace unum::usearch;

constexpr size_t N = 10000;
constexpr size_t Q = 100;
constexpr size_t DIM = 32;
constexpr unsigned SEED = 42;

using Vector = array<float, DIM>;
using Clock = chrono::steady_clock;

vector<Vector> generarVectores(size_t cantidad, mt19937& rng) {
    uniform_real_distribution<float> dist(0.0f, 1.0f);

    vector<Vector> datos(cantidad);

    for (auto& v : datos) {
        for (float& componente : v) {
            componente = dist(rng);
        }
    }

    return datos;
}

float distanciaL2(const Vector& a, const Vector& b) {
    float suma = 0.0f;

    for (size_t i = 0; i < DIM; i++) {
        float diferencia = a[i] - b[i];
        suma += diferencia * diferencia;
    }

    return suma;
}

vector<size_t> busquedaExacta(
    const vector<Vector>& datos,
    const Vector& query,
    size_t k
) {
    vector<pair<float, size_t>> distancias;
    distancias.reserve(datos.size());

    for (size_t i = 0; i < datos.size(); i++) {
        float d = distanciaL2(datos[i], query);
        distancias.push_back({d, i});
    }

    sort(distancias.begin(), distancias.end());

    vector<size_t> resultado;
    for (size_t i = 0; i < k; i++) {
        resultado.push_back(distancias[i].second);
    }

    return resultado;
}

double calcularRecall(
    const vector<size_t>& aproximados,
    const vector<size_t>& exactos
) {
    size_t aciertos = 0;

    for (size_t id : aproximados) {
        if (find(exactos.begin(), exactos.end(), id)
            != exactos.end()) {
            aciertos++;
        }
    }

    return static_cast<double>(aciertos) / exactos.size();
}

int main() {
    mt19937 rng(SEED);

    auto datos = generarVectores(N, rng);
    auto consultas = generarVectores(Q, rng);

    // 1. Crear indice HNSW
    metric_punned_t metric(
        DIM,
        metric_kind_t::l2sq_k,
        scalar_kind_t::f32_k
    );

    index_dense_t index = index_dense_t::make(metric);

    if (!(bool)index.try_reserve(N)) {
        cerr << "No se pudo reservar memoria\n";
        return 1;
    }

    // 2. Insertar los vectores
    std::cout << "Construyendo HNSW..." << endl;

    for (size_t i = 0; i < N; i++) {
        auto added = index.add(i, datos[i].data());

        if (!added) {
            cerr << "Error al insertar vector " << i << "\n";
            return 1;
        }
    }

    std::cout << "HNSW construido con "
         << index.size() << " vectores.\n";

    // 3. Preparar los vecinos exactos
    const vector<size_t> valoresK = {1, 5, 10};
    const vector<size_t> expansiones = {10, 40, 100};

    // Exactos para K=10. Sus prefijos sirven para K=1 y K=5.
    vector<vector<size_t>> exactos(Q);

    for (size_t i = 0; i < Q; i++) {
        exactos[i] = busquedaExacta(datos, consultas[i], 10);
    }

    // 4. Evaluar HNSW
    std::cout << fixed << setprecision(4);
    std::cout << "\nK,Expansion,Recall,Latencia_us\n";

    for (size_t k : valoresK) {
        for (size_t expansion : expansiones) {

            index.change_expansion_search(expansion);

            double sumaRecall = 0.0;
            double sumaLatencia = 0.0;

            for (size_t i = 0; i < Q; i++) {

                auto inicio = Clock::now();

                auto resultado = index.search(
                    consultas[i].data(), k
                );

                auto fin = Clock::now();

                if (!resultado || resultado.size() != k) {
                    cerr << "Error en consulta " << i << "\n";
                    return 1;
                }

                // Copiar IDs antes de la siguiente busqueda
                vector<size_t> aproximados;
                aproximados.reserve(k);

                for (size_t j = 0; j < resultado.size(); j++) {
                    aproximados.push_back(
                        static_cast<size_t>(resultado[j].member.key)
                    );
                }

                vector<size_t> exactosK(
                    exactos[i].begin(),
                    exactos[i].begin() + k
                );

                sumaRecall += calcularRecall(
                    aproximados, exactosK
                );

                double micros = chrono::duration<double, micro>(
                    fin - inicio
                ).count();

                sumaLatencia += micros;
            }

            double recall = sumaRecall / Q;
            double latencia = sumaLatencia / Q;

            std::cout << k << ","
                 << expansion << ","
                 << recall << ","
                 << latencia << "\n";
        }
    }

    return 0;
}
