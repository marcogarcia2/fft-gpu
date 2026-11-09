# fft-gpu

Trabalho de Conclusão de Curso em Engenharia de Computação: **otimização progressiva da FFT em GPU**, medindo tempo **e energia** de cada etapa de otimização e comparando com a cuFFT.

O projeto implementa a FFT de Cooley-Tukey (decimação no tempo, radix-2 e radix-4) em CUDA em uma sequência de versões, cada uma acrescentando uma única melhoria à anterior. Todas são medidas com o mesmo *harness* (tempo por eventos CUDA, energia pelo contador do NVML), perfiladas com o Nsight Compute e comparadas em gráficos prontos para o texto.

## Kernels

| Versão | O que acrescenta |
|---|---|
| `FFT_Naive` | *In place*: *bit reversal* + um kernel global por estágio |
| `FFT_Stockham` | *Out of place*, alternando entre dois vetores (*ping-pong*), sem *bit reversal* |
| `FFT_Shared` | Stockham com memória compartilhada: vários estágios por passada, menos idas à memória global |
| `FFT_BCF` | Shared com *padding* na memória compartilhada (uma posição vazia de 32 B a cada 4) para reduzir os conflitos de banco (*bank-conflict-free*) |
| `FFT_Radix4` | BCF com os estágios locais fundidos dois a dois (borboletas radix-4) |
| `cuFFT` | Referência (biblioteca da NVIDIA) |

Cada versão própria tem uma gêmea `*_Tabled`, que lê os *twiddle factors* de uma tabela pré-calculada em vez de calculá-los com `sin`/`cos`. Todas suportam `float` (FP32) e `double` (FP64).

## Como funciona

A GPU fica numa máquina remota. Tudo é disparado da máquina local pelo `run.sh`, que:

1. copia os fontes para a máquina remota via `scp`;
2. compila lá com `nvcc` (`make cuda-compile`);
3. roda o benchmark (`-e`) e/ou o *profiler* (`-p`);
4. traz os resultados de volta para `results/<timestamp>/`;
5. gera os gráficos (`plots.py`) e um resumo da execução (`summary.py`).

### Metodologia de medição

- **Tamanhos:** N = 2¹⁰, 2¹², …, 2²⁴, em FP32 e FP64.
- **Repetições:** cada combinação kernel × N × precisão roda `RPT = 30` vezes. Cada repetição executa o kernel `ACC` vezes seguidas, com `ACC` escolhido por kernel, precisão e N (tabela `ACC_list` em [`src/main.cu`](src/main.cu)) para a janela de energia ficar em torno de 1 s.
- **Energia:** lida pelo contador acumulado do NVML (`nvmlDeviceGetTotalEnergyConsumption`) no início e no fim de cada janela.
- **Overflow:** como a FFT não é normalizada, os dados são restaurados a cada `SRI` execuções (cópia D2D). Um experimento de controle mede o tempo e a energia dessa cópia, que são descontados das medições.
- **Estatísticas:** média, mediana, desvio padrão e IC 95% (t de Student, 29 graus de liberdade) de tempo e energia; a partir das médias, GFLOP/s (5N log₂N operações), desempenho por watt e EDP; temperatura e *clock* do SM no início e no fim de cada configuração.
- **Correção:** antes de ser medido, cada kernel é comparado com a cuFFT (erro relativo máximo de 10⁻³, `THRESHOLD` em [`src/util.cuh`](src/util.cuh)). Uma divergência é reportada no `stderr`, mas não interrompe a execução.

## Requisitos

**Máquina remota (com a GPU)**

- GPU NVIDIA (os resultados do trabalho são de uma RTX 3060) e CUDA Toolkit com `nvcc`, cuFFT e NVML (`-lcufft -lnvidia-ml`)
- Nsight Compute em `/usr/bin/ncu`, para o *profiler*
- Permissão de `sudo` para o usuário: o `ncu` precisa dela para ler os contadores de desempenho

**Máquina local**

- `bash`, `ssh`, `scp` e [`sshpass`](https://sourceforge.net/projects/sshpass/)
- Python 3 com `numpy`, `pandas` e `matplotlib`
- Opcional: TeX Live, para os gráficos usarem a fonte Latin Modern (a mesma do LaTeX). Sem ela, o `plots.py` usa DejaVu Serif. O caminho está em `LATIN_MODERN_DIR`, no topo do [`plots.py`](plots.py).

```bash
pip install numpy pandas matplotlib
```

## Configuração (`.env`)

Crie um arquivo `.env` na raiz do repositório (ele está no `.gitignore`):

```bash
HOST=gpu.exemplo.edu.br          # endereço da máquina remota
USER=seu_usuario                 # usuário SSH na máquina remota
PORT=22                          # porta SSH
PASSWD=sua_senha                 # senha SSH, também usada no sudo do ncu
REMOTE_PATH=/home/seu_usuario/fft-gpu   # pasta de trabalho na máquina remota (precisa existir)
LOCAL_PATH=/caminho/local/fft-gpu       # raiz deste repositório (usada pelo script do roofline)
```

> A senha fica em texto puro no `.env`. Não faça *commit* desse arquivo.

## Como rodar

### 1. Parâmetros do roofline (uma vez por GPU)

```bash
./roofline/get_roofline.sh
```

Roda um *microbenchmark* de banda de memória (cópia entre dois vetores) e de pico de GFLOP/s (FMA, em FP32 e FP64) na GPU remota e salva `results/roofline.json`. O `plots.py` depende desse arquivo.

### 2. Experimentos e *profiling*

```bash
./run.sh -e        # benchmark completo: tempo e energia -> results/<timestamp>/Results.csv
./run.sh -p        # profiler NCU de cada kernel -> results/<timestamp>/*.ncu-rep e *.csv
./run.sh -e -p     # os dois, e no fim gera os gráficos
```

- Sem flags, o `run.sh` só copia e compila.
- O benchmark roda todos os kernels de `kernel_list` em [`src/main.cu`](src/main.cu), para todos os N de `NX_list`. Uma rodada completa leva cerca de 1h30.
- O *profiler* roda uma FFT de cada combinação definida no topo do [`run.sh`](run.sh) (`kernel_list`, `nx_list`, `precision_list`; por padrão N = 2²⁴), com as seções e métricas de `NCU_FLAGS` no [`Makefile`](Makefile).

### 3. Gráficos e resumo

```bash
python3 plots.py results/<timestamp>          # gera os PDFs em images/<timestamp>/
python3 plots.py results/<timestamp> --show   # também abre cada gráfico numa janela
python3 summary.py results/<timestamp>        # resumo da execução (ver abaixo)
```

Rode os scripts a partir da raiz do repositório: os caminhos `results/` e `images/` são relativos. Sem o caminho, os dois scripts usam a pasta mais recente de `results/`. O `plots.py` precisa do `Results.csv` e dos CSVs do NCU na mesma pasta, além de `results/roofline.json`.

O `summary.py` mostra, para cada configuração da cuFFT, a duração estimada, a janela de energia, o *clock* e a temperatura, e no fim as faixas de *clock* e de temperatura dessas configurações.

| Arquivo | Gráfico |
|---|---|
| `plot_00_*` | Modelo roofline (N = 2²⁴), FP32 e FP64 |
| `plot_01_*` | Tempo × N, FP32 e FP64 |
| `plot_02_Pareto_N24` | Pareto tempo × energia |
| `plot_03_GFLOPS_W` | Desempenho por watt × N (grade FP32/FP64 × direto/tabelado) |
| `plot_04_Razao_Tabelado` | Razão tempo tabelado / cálculo direto × N, com o limite do L2 (2,25 MB, fixo no código) |
| `plot_05_Gargalo_N24` | Mapa de calor de utilização das unidades (Nsight) |
| `plot_06_Ganho_Etapas_*` | Speedup acumulado por etapa de otimização |

Os gráficos 00, 01 e 06 têm uma versão `*_TABLED`, com as versões de *twiddle factors* tabelados.

### Rodando um kernel isolado

Na máquina remota, depois de compilar, o binário sem argumentos roda o benchmark completo (`make cuda-run`). Com um kernel, N e precisão, ele roda uma única FFT, que é o modo usado pelo *profiler*:

```bash
./main FFT_Radix4 16777216 double
```

## Estrutura

```
src/
  main.cu          lista de kernels, tabela de ACC e laços do experimento
  kernels.cuh      implementações da FFT
  harness.cuh      medição de tempo e energia, experimento de controle
  util.cuh         contexto da FFT, verificação contra a cuFFT, estatísticas, CSV
  profiler.cuh     modo de execução única, para o ncu
roofline/          microbenchmark dos tetos do roofline
cpu_version/       versão em CPU (OpenMP), compilada localmente com `make`
run.sh             orquestra a execução remota
Makefile           alvos locais (versão em CPU) e remotos (compilação, execução, ncu, roofline)
plots.py           gráficos do trabalho
summary.py         resumo de uma execução
arcane.py          funções auxiliares de leitura dos resultados
```

As pastas `results/` e `images/` não são versionadas (estão no `.gitignore`).

### Adicionando um kernel

1. Implemente a função em [`src/kernels.cuh`](src/kernels.cuh), com a assinatura `void K(T2* d_X, int N, FFTContext& ctx)`.
2. Registre em `kernel_list` no [`src/main.cu`](src/main.cu), indicando se ele precisa de vetor auxiliar, de tabela ou de plano.
3. Adicione uma linha em `ACC_list` com um `ACC` por N e precisão. O programa se recusa a rodar se faltar a linha.
4. Para perfilar, inclua o nome em `kernel_list` no [`run.sh`](run.sh).
5. Para aparecer nos gráficos, inclua o nome nas listas de kernels (com a cor) de cada gráfico no [`plots.py`](plots.py).

## Licença

[MIT](LICENSE)
