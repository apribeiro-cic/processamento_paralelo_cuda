#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h> 

// Incluindo os headers em C para a leitura e escrita de imagens
extern "C" {
    #include "img_source.h"
    #include "filter_source.h"
}

// Declaração função Cuda
extern "C" void aplicar_filtro_gaussiano_cuda(Imagem *img, int iteracoes);

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Uso: %s <entrada.p?m> <saida.p?m> <iteracoes>\n", argv[0]);
        return 1;
    }

    const char *entrada = argv[1];
    const char *saida = argv[2];
    int iteracoes = atoi(argv[3]);

    Imagem img;
    
    if (strstr(entrada, ".pgm")) {
        img = ler_imagem_pgm_p5(entrada);
    } else if (strstr(entrada, ".ppm")) {
        img = ler_imagem_ppm_p6(entrada);
    } else {
        printf("Formato não suportado.\n");
        return 1;
    }
    printf("Imagem carregada: %dx%d (%d canais)\n", img.largura, img.altura, img.canais);

    Imagem img_cpu;
    img_cpu.largura = img.largura;
    img_cpu.altura = img.altura;
    img_cpu.canais = img.canais;
    int total_bytes = img.largura * img.altura * img.canais;
    img_cpu.pixels = (unsigned char *)malloc(total_bytes);
    memcpy(img_cpu.pixels, img.pixels, total_bytes); 

    // Cronometragem da CPU
    printf("\nExecutando filtro Sequencial na CPU para referencia...\n");
    clock_t inicio_cpu = clock();
    
    aplicar_filtro_gaussiano_sequencial(&img_cpu, iteracoes);
    
    clock_t fim_cpu = clock();
    double tempo_cpu_ms = ((double)(fim_cpu - inicio_cpu)) / CLOCKS_PER_SEC * 1000.0;
    printf("Tempo de Execucao (Sequencial CPU): %.3f ms\n", tempo_cpu_ms);

    printf("\nExecutando filtro Paralelo na GPU...\n");
    aplicar_filtro_gaussiano_cuda(&img, iteracoes); 

    validar_corretude_imagens(&img_cpu, &img, 2);
    
    if (img.canais == 3) {
        salvar_imagem_ppm_p6(saida, &img);
    } else {
        salvar_imagem_pgm_p5(saida, &img);
    }

    free(img.pixels);
    free(img_cpu.pixels);
    printf("Processamento finalizado. Imagem da GPU salva em %s\n", saida);
    return 0;
}