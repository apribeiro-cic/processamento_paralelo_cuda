# Benchmark de Suavização/Desfoque Gaussiano com CUDA

Universidade Estadual de Santa Cruz — UESC  
Curso de Ciência da Computação  
Disciplina: DEC107 — Processamento Paralelo  

---

## 📌 Descrição

Este projeto implementa e avalia o desempenho do filtro Gaussiano aplicado em imagens digitais, explorando diferentes modelos de paralelismo: execução sequencial (CPU) e execução paralela em GPU utilizando CUDA.

O objetivo principal é analisar o ganho de desempenho obtido com o uso de GPUs em comparação com a versão sequencial, além de avaliar aspectos como escalabilidade e corretude numérica.

O filtro Gaussiano é amplamente utilizado em processamento de imagens para suavização e redução de ruídos, sendo altamente adequado para paralelização devido à independência no processamento de pixels.

---

## ⚙️ Implementação

A versão CUDA foi desenvolvida com as seguintes características:

- **Mapeamento:** Cada thread processa um pixel da imagem.
- **Filtro:** Kernel Gaussiano 5x5 aplicado em paralelo.
- **Memória Global:** Utilizada para o armazenamento e transferência da imagem.
- **Memória Constante:** Utilizada para o armazenamento do kernel Gaussiano (otimização de leitura).
- **Dimensão dos Blocos:** Configuração de blocos de `16x16` threads.
- **Gerenciamento:** Alocação e liberação explícita através de `cudaMalloc`, `cudaMemcpy` e `cudaFree`.
- **Métricas de Tempo:** Medição precisa do tempo de processamento na GPU utilizando `cudaEvent_t`.

---

## 🧪 Teste de Corretude

A corretude da implementação foi verificada comparando pixel a pixel a saída gerada pela GPU com o resultado obtido na versão sequencial executada pela CPU.

As métricas calculadas foram:
- Erro máximo absoluto
- Erro médio relativo
- Número de pixels fora da tolerância

> **Nota:** A tolerância numérica é configurável no código (valor padrão: `10`), permitindo acomodar pequenas divergências provenientes da aritmética de ponto flutuante entre as arquiteturas da CPU e da GPU.

---

## 📊 Resultados

### Desempenho
- A execução na CPU sequencial foi utilizada como *baseline* (referência padrão).
- A versão desenvolvida em CUDA apresentou uma redução drástica no tempo de execução.
- Nos cenários testados, o ganho de desempenho (*Speedup*) ultrapassou a marca de **400x**.

### Escalabilidade
O tempo de execução na GPU apresentou um crescimento significativamente mais controlado e linear em relação à CPU à medida que a resolução das imagens aumentava, comprovando excelente escalabilidade para grandes volumes de dados.

---

## 🖥️ Hardware Utilizado

- **CPU:** AMD Ryzen 7 5800H (8 cores / 16 threads)  
- **GPU:** NVIDIA GeForce GTX 1650  
- **RAM:** 16 GB  
- **Sistema Operacional:** Ubuntu 24.04 LTS  
- **Compilador:** CUDA Toolkit (compatível com `nvcc`)  

---

## 🚀 Como Compilar

Para compilar o projeto, execute o comando `nvcc` linkando todos os módulos necessários:

```bash
nvcc main_cuda.cu kernel_cuda.cu img_source.c filter_source.c -o filtro_gauss
```

## ▶️ Como Executar


O binário gerado aceita o caminho da imagem de entrada (no formato `.pgm`), o caminho de destino para a imagem resultante e a quantidade de iterações para o benchmark:

```bash
./filtro_gauss <imagem_entrada.pgm> <imagem_saida.pgm> <iteracoes>
```
## ⚠️ Nota de Transparência sobre o Uso de IA

Declaro que este projeto contou com o auxílio das ferramentas de IA ChatGPT e Gemini exclusivamente para as tarefas de estruturação de código, auxílio na implementação, geração de scripts de teste e apoio na redação técnica do relatório.

Como autor(a), atesto que revisei, testei e validei criticamente todo o conteúdo gerado, assumindo total e exclusiva responsabilidade pela correção lógica do código, precisão dos resultados e integridade acadêmica do material entregue.

- **Ariel Pina Ribeiro** — 04/07/2026  
- **Henrique Sousa Brito** — 04/07/2026