#ifndef FILTER_SOURCE_H
#define FILTER_SOURCE_H

#include <stdio.h>
#include <stdlib.h>
#include "img_source.h"

// Função de filtro sequencial
void aplicar_filtro_gaussiano_sequencial(Imagem *img, int iteracoes);

// Função para validar a corretude entre a imagem processada na CPU (Sequencial) e a processada na GPU (CUDA).
void validar_corretude_imagens(Imagem *img_cpu, Imagem *img_gpu, int tolerancia);

#endif // FILTER_SOURCE_H