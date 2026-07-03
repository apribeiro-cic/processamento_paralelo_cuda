#include <stdio.h>
#include "img_source.h"

// Kernel Gaussiano 5x5 armazenado na memória constante da GPU
__constant__ float d_kernel_5x5[5][5] = {
    {0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902},
    {0.01330621, 0.05963430, 0.09832033, 0.05963430, 0.01330621},
    {0.02193823, 0.09832033, 0.16210282, 0.09832033, 0.02193823},
    {0.01330621, 0.05963430, 0.09832033, 0.05963430, 0.01330621},
    {0.00296902, 0.01330621, 0.02193823, 0.01330621, 0.00296902}
};

// O Kernel Executado na GPU
__global__ void kernel_gaussiano_5x5(unsigned char *leitura, unsigned char *escrita, int largura, int altura, int canais) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < largura && y < altura) {
        int pad = 2;

        for (int c = 0; c < canais; c++) {
            float soma = 0.0;

            for (int ky = -pad; ky <= pad; ky++) {
                for (int kx = -pad; kx <= pad; kx++) {
                    int py = y + ky;
                    int px = x + kx;

                    if (py < 0) py = 0;
                    else if (py >= altura) py = altura - 1;

                    if (px < 0) px = 0;
                    else if (px >= largura) px = largura - 1;

                    int indice_pixel = (py * largura + px) * canais + c;
                    soma += leitura[indice_pixel] * d_kernel_5x5[ky + pad][kx + pad];
                }
            }

            int indice_saida = (y * largura + x) * canais + c;
            escrita[indice_saida] = (unsigned char)soma;
        }
    }
}

// A Função Wrapper chamada pelo CPU (Host)
extern "C" void aplicar_filtro_gaussiano_cuda(Imagem *img, int iteracoes) {
    int total_bytes = img->largura * img->altura * img->canais * sizeof(unsigned char);

    // Prepara eventos CUDA para medir o tempo de execução
    cudaEvent_t start_total, stop_total;
    cudaEvent_t start_kernel, stop_kernel;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);

    // Marca o início do TEMPO TOTAL (inclui alocação e cópia de memória)
    cudaEventRecord(start_total);

    unsigned char *d_buffer_A, *d_buffer_B;
    cudaMalloc((void**)&d_buffer_A, total_bytes);
    cudaMalloc((void**)&d_buffer_B, total_bytes);

    // Envia a imagem original para a GPU
    cudaMemcpy(d_buffer_A, img->pixels, total_bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((img->largura + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (img->altura + threadsPerBlock.y - 1) / threadsPerBlock.y);

    unsigned char *d_leitura = d_buffer_A;
    unsigned char *d_escrita = d_buffer_B;

    // Marca o início da execução dos kernels (apenas o tempo de processamento)
    cudaEventRecord(start_kernel);

    for (int i = 0; i < iteracoes; i++) {
        kernel_gaussiano_5x5<<<numBlocks, threadsPerBlock>>>(d_leitura, d_escrita, img->largura, img->altura, img->canais);
        
        unsigned char *temp = d_leitura;
        d_leitura = d_escrita;
        d_escrita = temp;
    }
    
    // Marca o fim da execução dos kernels
    cudaEventRecord(stop_kernel);
    cudaEventSynchronize(stop_kernel); // Espera a GPU terminar o último bloco

    // Copia o resultado de volta para o Host
    cudaMemcpy(img->pixels, d_leitura, total_bytes, cudaMemcpyDeviceToHost);

    // Marca o fim do tempo total
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    // Exibe o relatório de desempenho
    float ms_total = 0, ms_kernel = 0;
    cudaEventElapsedTime(&ms_total, start_total, stop_total);
    cudaEventElapsedTime(&ms_kernel, start_kernel, stop_kernel);

    printf("\n--- Relatorio de Desempenho CUDA ---\n");
    printf("Tempo dos Kernels (Processamento puro): %.3f ms\n", ms_kernel);
    printf("Tempo Total (Kernels + Transferencias): %.3f ms\n", ms_total);
    printf("Overhead de Memoria (Host <-> Device): %.3f ms\n", ms_total - ms_kernel);
    printf("------------------------------------\n");

    cudaFree(d_buffer_A);
    cudaFree(d_buffer_B);
    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
}