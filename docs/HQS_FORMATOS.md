# Formatos HQS: cuáles valen

Revisión del 2026-11-02, medida sobre pesos reales de Gemma 4 26B y contrastada con
Unsloth UD-Q4_K_M. Error: L2 relativo de los pesos frente al BF16 original, en seis
tensores (atención q y o, down de la MLP compartida, gate/up y down de expertos, filas
del embedding), con el relleno a 256 de GM4X.

## Escalones vigentes

| Escalón | Formato | Bytes/256 | Bits/peso | Error | Equivalente llama.cpp |
|---|---|---:|---:|---:|---|
| 4 bits | HQ44 (`hq44k_g16`, HQS v4) | 152 | 4,75 | ~6 % | Q4_K: 7,1 % con matriz de importancia |
| 5 bits | HQ52K | 184 | 5,75 | 3,46 % | Q5_1: 3,5 % |
| 8 bits | **HQ8K**, nuevo | 272 | 8,50 | 0,56 % | Q8_0: 0,555 % |
| Sin pérdida | FP16 | 512 | 16 | 0 % para valores BF16 | — |

En el modelo entero, los expertos HQ44 rinden igual que los de Unsloth a igual tamaño:
KL frente al original de 0,544 los nuestros y 0,541 los suyos, con la parte densa sin
cuantizar en ambos. El error no venía del formato sino del reparto: atención y MLP
compartida son pequeñas y muy sensibles, y no deben bajar de 8 bits.

## HQ8K

`src/hqs/hq8.rs`. Superbloque de 256 pesos, 272 bytes, alineado a 16:

| Bytes | Campo |
|---|---|
| 0–1 | `d`, f16: paso máximo del superbloque, redondeado hacia arriba |
| 2–9 | `q[8]`, u8: paso de cada grupo de 32 = `d * q / 255`, con `q` hacia arriba |
| 10–15 | Reservado, cero |
| 16–271 | 256 códigos i8 simétricos en [−127, 127]; peso = código × paso |

Al redondear hacia arriba, ningún grupo recorta su máximo. En pesos reales, el
codificador de Rust da 0,543–0,613 %. Variante descartada: subescalas de 6 bits en 264 B
(8,25 bits, 0,57 %); ahorraba unos 70 MB en el 26B a cambio de leer alineado solo a 8.

Estado: el motor lo carga (`kernels/hq8.cu`, GEMV en `hq8_gemv.cuh`); es el formato de
atención, MLP compartida y embedding en el HNF final del 26B. Texto de la revisión del
2026-11-02, traído del conversor v9.1.

## Descartados

- **HQ6.2K** (2026-11-02): cada grupo de 8 pesos lleva escala y mínimo de 8 bits, que
  son 2 bits/peso de metadatos, y solo deja 6 bits para el código. Da 1,11 % con 8,25
  bits, el doble que HQ8K. `--embedding-hq62` ya falla con un mensaje; el decodificador
  se conserva para leer HNF antiguos. Si hiciera falta un escalón de 6 bits, HQ52K con
  un bit más da 1,79 % con 6,75 bits.

## Sin revisar

HQ31K, HQ41K, HQ51K, HQ42K y las variantes x.3 (`hqN3k_gG`) no se han medido en esta
revisión. Antes de retirarlos, comprobar qué HNF en uso los contienen.
