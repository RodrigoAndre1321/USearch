
#include <iostream>
#include <vector>
#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>
#include <cmath>
#include <limits>

using namespace unum::usearch;
using namespace std;

int main() {

    // 1. Crear el indice con distancia L2 al cuadrado.
    metric_punned_t metric =
        metric_punned_t::builtin(2, metric_kind_t::l2sq_k);

    index_dense_t index = index_dense_t::make(metric);
    metric_punned_t metrica_vacia(
        2,
        metric_kind_t::l2sq_k,
        scalar_kind_t::f32_k
    );

    index_dense_t indice_vacio = index_dense_t::make(metrica_vacia);
    indice_vacio.reserve(10);
    index.reserve(10);

    // 2. Insertar cinco vectores.
    vector<vector<float>> puntos = {
        {1.0f, 1.0f},  // ID 0
        {1.0f, 1.0f},  // ID 1
        {1.0f, 1.0f},  // ID 2
        {3.0f, 3.0f},  // ID 3
        {5.0f, 5.0f}   // ID 4
    };
    for (size_t i = 0; i < puntos.size(); ++i) {
        auto resultado = index.add(i, puntos[i].data());

        if (!resultado) {
            cerr << "Error al insertar ID " << i << "\n";
            return 1;
        }
    }

    // 3. Definir consulta y parametros.
    float consulta[] = {1.0f, 1.0f};

    size_t k = 3;
    auto resultado_hnsw = index.search(consulta, k);

    if (!resultado_hnsw)
        return 1;
    /*
    cout << "=== HNSW ORIGINAL ===\n";

    for (size_t i = 0; i < resultado_hnsw.size(); ++i) {
        cout << "ID: " << resultado_hnsw[i].member.key
            << " | Distancia: " << resultado_hnsw[i].distance
            << "\n";
    }

    // Ahora ejecutamos ABS.
    auto resultado_abs = index.search_abs(consulta, k, gamma);

    cout << "\n=== HNSW + ABS ===\n";
    for (size_t i = 0; i < resultado_abs.size(); ++i) {
        cout << "ID: " << resultado_abs[i].member.key
             << " | Distancia: " << resultado_abs[i].distance
             << "\n";
    }
    auto resultado = indice_vacio.search_abs(
        consulta,
        3,
        0.5
    );

    cout << "CP-08 Indice vacio"
        << " | Exito: " << (resultado ? "SI" : "NO")
        << " | Resultados: " << resultado.size()
        << "\n";
    */
    // CP-09: Busqueda despues de eliminar un vector

    auto eliminado = index.remove(1);

    if (!eliminado || eliminado.completed != 1) {
        cerr << "CP-09: Error al eliminar el vector\n";
        return 1;
    }
    cout << "\n=== CP-09: VECTOR ELIMINADO ===\n";

    cout << "Vectores eliminados: "
        << eliminado.completed << "\n";
    auto resultado_original = index.search(consulta, 3);

    bool original_correcto = static_cast<bool>(resultado_original) && resultado_original.size() == 3;

    for (size_t i = 0; i < resultado_original.size(); i++) {

        auto id = resultado_original[i].member.key;

        if (id == 1) {
            original_correcto = false;
        }
    }

    cout << "HNSW original sin ID eliminado: "
        << (original_correcto ? "SI" : "NO") << "\n";
    
    auto resultado_abs = index.search_abs(consulta, 3, 0.2);

    bool abs_correcto = static_cast<bool>(resultado_abs) && resultado_abs.size() == 3;

    for (size_t i = 0; i < resultado_abs.size(); i++) {

        auto id = resultado_abs[i].member.key;

        if (id == 1) {
            abs_correcto = false;
        }
    }

    cout << "ABS sin ID eliminado: "
        << (abs_correcto ? "SI" : "NO") << "\n";

    cout << "CP-09: "
        << (original_correcto && abs_correcto ? "APROBADO" : "FALLO")
        << "\n";
    return original_correcto && abs_correcto ? 0 : 1;
}
