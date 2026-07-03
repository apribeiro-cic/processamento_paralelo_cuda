#include "filter_source.h"
#include <math.h>

// Função para validar a corretude entre a imagem processada na CPU (Sequencial) e a processada na GPU (CUDA).
void validar_corretude_imagens(Imagem *img_cpu, Imagem *img_gpu, int tolerancia) {
    if (img_cpu->largura != img_gpu->largura || img_cpu->altura != img_gpu->altura || img_cpu->canais != img_gpu->canais) {
        printf("Erro: As imagens tem dimensoes diferentes e nao podem ser comparadas.\n");
        return;
    }

    int total_bytes = img_cpu->largura * img_cpu->altura * img_cpu->canais;
    int erro_maximo = 0;
    double soma_erros = 0.0;
    int pixels_fora_tolerancia = 0;

    // Compara pixel a pixel
    for (int i = 0; i < total_bytes; i++) {
        // Calcula a diferença absoluta entre o cálculo da CPU e da GPU
        int diff = abs((int)img_cpu->pixels[i] - (int)img_gpu->pixels[i]);

        if (diff > erro_maximo) {
            erro_maximo = diff;
        }

        soma_erros += diff;

        if (diff > tolerancia) {
            pixels_fora_tolerancia++;
        }
    }

    double erro_medio = soma_erros / total_bytes;

    printf("\n=== RESULTADO DO TESTE DE CORRETUDE ===\n");
    printf("Tolerancia adotada: %d\n", tolerancia);
    printf("Erro Maximo Absoluto: %d\n", erro_maximo);
    printf("Erro Relativo Medio: %.4f\n", erro_medio);
    printf("Pixels fora da tolerancia: %d de %d\n", pixels_fora_tolerancia, total_bytes);

    if (pixels_fora_tolerancia == 0) {
        printf("Status: SUCESSO! A implementacao CUDA e numericamente equivalente a CPU.\n");
    } else {
        printf("Status: FALHA! Diferencas numericas excedem a tolerancia permitida.\n");
    }
    printf("=======================================\n\n");
}

// Função de filtro sequencial
void aplicar_filtro_gaussiano_sequencial(Imagem *img, int iteracoes) {
    int largura = img->largura;
    int altura = img->altura;
    int canais = img->canais;
    int total_bytes = largura * altura * canais;

    // Aloca um buffer temporário para salvar os cálculos da iteração atual
    unsigned char *buffer = (unsigned char *)malloc(total_bytes * sizeof(unsigned char));
    if (!buffer) {
        printf("Erro ao alocar buffer temporário.\n");
        exit(1);
    }

    // Kernel 5x5 normalizado retirado do output do código Python disponibilizado pelo professor
    const float kernel_5x5[5][5] = {
        {0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902},
        {0.01330621, 0.05963430, 0.09832033, 0.05963430, 0.01330621},
        {0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823},
        {0.01330621, 0.05963430, 0.09832033, 0.05963430, 0.01330621},
        {0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902}
    };
    
    // Para kernel 5x5, o padding necessário é de 2 pixels em cada direção
    int pad = 2; 

    // Ponteiros para a técnica de alteração de ponteiros (duas matrizes, uma para leitura e outra para escrita, alternando a cada iteração)
    unsigned char *leitura = img->pixels;
    unsigned char *escrita = buffer;

    for (int iter = 0; iter < iteracoes; iter++) {
        
        // Varre a imagem pixel por pixel
        for (int y = 0; y < altura; y++) {
            for (int x = 0; x < largura; x++) {
                
                // Trata cada canal de cor independentemente (funciona para PGM e PPM)
                for (int c = 0; c < canais; c++) {
                    float soma = 0.0;

                    // Percorre a máscara/kernel 5x5
                    for (int ky = -pad; ky <= pad; ky++) {
                        for (int kx = -pad; kx <= pad; kx++) {
                            
                            // Calcula as coordenadas reais do vizinho
                            int py = y + ky;
                            int px = x + kx;

                            // Lógica de borda, como se fosse o mode='edge' do np.pad do Python
                            if (py < 0) py = 0;
                            else if (py >= altura) py = altura - 1;
                            
                            if (px < 0) px = 0;
                            else if (px >= largura) px = largura - 1;

                            // Mapeia o 2D para o índice 1D contíguo
                            int indice_pixel = (py * largura + px) * canais + c;
                            
                            // Multiplica e soma o valor do pixel pelo valor correspondente do kernel
                            soma += leitura[indice_pixel] * kernel_5x5[ky + pad][kx + pad];
                        }
                    }
                    
                    // Salva o resultado no buffer de escrita
                    int indice_saida = (y * largura + x) * canais + c;
                    escrita[indice_saida] = (unsigned char)soma;
                }
            }
        }
        
        // Troca os ponteiros para a próxima iteração
        unsigned char *temp = leitura;
        leitura = escrita;
        escrita = temp;
    }

    // Se o número de iterações for ímpar, o resultado final terá parado no 'buffer'.
    if (iteracoes % 2 != 0) {
        for (int i = 0; i < total_bytes; i++) {
            img->pixels[i] = buffer[i];
        }
    }

    // Libera a memória que foi alocada dinamicamente nesta função
    free(buffer);
}