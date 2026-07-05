import matplotlib.pyplot as plt

imagens = ["512x512", "1024x1024", "4096x4096"]

cpu = [2656.789, 10592.498, 170436.582]
cuda = [5.973, 24.160, 351.512]
speedup = [444.7, 438.5, 484.9]

plt.figure()
plt.plot(imagens, cpu, marker='o', label="CPU")
plt.plot(imagens, cuda, marker='o', label="CUDA")

plt.title("Tempo de Execução: CPU vs CUDA")
plt.xlabel("Imagem")
plt.ylabel("Tempo (ms)")
plt.legend()
plt.grid(True)

plt.savefig("tempo_cpu_vs_cuda.png", dpi=300, bbox_inches='tight')
plt.show()


plt.figure()
plt.plot(imagens, speedup, marker='o', color='green')

plt.title("Speedup (CPU / CUDA)")
plt.xlabel("Imagem")
plt.ylabel("Speedup (x)")
plt.grid(True)

plt.savefig("speedup.png", dpi=300, bbox_inches='tight')
plt.show()


pixels = [512*512, 1024*1024, 4096*4096]

plt.figure()
plt.plot(pixels, cpu, marker='o', label="CPU")
plt.plot(pixels, cuda, marker='o', label="CUDA")

plt.title("Escalabilidade do Algoritmo")
plt.xlabel("Pixels")
plt.ylabel("Tempo (ms)")
plt.legend()
plt.grid(True)

plt.savefig("escalabilidade.png", dpi=300, bbox_inches='tight')
plt.show()