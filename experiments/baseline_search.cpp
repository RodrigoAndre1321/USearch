
#include <iostream>
#include <usearch/index.hpp>
#include <usearch/index_dense.hpp>

using namespace unum::usearch;

int main() {
    metric_punned_t metric(
        2,
        metric_kind_t::l2sq_k,
        scalar_kind_t::f32_k
    );

    index_dense_t index = index_dense_t::make(metric);
    index.reserve(10);

    float vectors[5][2] = {
        {1.0f, 1.0f},
        {2.0f, 2.0f},
        {5.0f, 5.0f},
        {8.0f, 8.0f},
        {1.0f, 2.0f}
    };

    for (int i = 0; i < 5; i++) {
        auto added = index.add(i, vectors[i]);
        if (!added) {
            std::cerr << "Error al insertar\n";
            return 1;
        }
    }

    float query[2] = {1.2f, 1.4f};

    auto results = index.search(query, 3);
    if (!results) {
        std::cerr << "Error en busqueda\n";
        return 1;
    }

    std::cout << "Vecinos encontrados:\n";

    for (std::size_t i = 0; i < results.size(); i++) {
        std::cout << "ID: " << results[i].member.key
                  << " | Distancia: " << results[i].distance
                  << "\n";
    }

    return 0;
}
