# HNFv10: especificación del contenedor

Borrador 1, 2026-11-02. Parte del diseño acordado en
`helios_convert_v9.1/docs/GEMMA4_MOE_CONTAINER_AUDIT.md`: lista de componentes, sin JSON
dentro, dos cabeceras, estados «en curso» y «completo», memoria como fórmula con sus límites,
y para cada hash lo que protege. No es compatible con HNFv9: el motor no lee v9; solo lo lee el
migrador.

Todo es little-endian. Los offsets son siempre absolutos desde el principio del fichero.

## 1. Disposición

| Zona | Offset | Tamaño |
|---|---|---|
| Cabecera A | 0 | 512 |
| Cabecera B | 512 | 512 |
| Payloads de los componentes | desde 4096 | cada uno alineado según declare, al menos 4096 |
| Metadatos | detrás de los payloads, alineados a 4096 | variable |

Las dos cabeceras ocupan un sector cada una. Los metadatos nuevos se escriben siempre detrás
de lo último escrito, nunca encima: así una escritura a medias no estropea el estado válido.

## 2. Cabecera (512 bytes)

| Offset | Tipo | Campo |
|---:|---|---|
| 0 | u8[8] | magic `HNFv10\0\0` |
| 8 | u16 | major = 10 |
| 10 | u16 | minor = 0 |
| 12 | u32 | tamaño de la cabecera = 512 |
| 16 | u64 | secuencia; la mayor válida manda |
| 24 | u32 | estado: 1 completo, 2 en curso |
| 28 | u32 | reservado, 0 |
| 32 | u64 | offset de los metadatos |
| 40 | u64 | tamaño de los metadatos |
| 48 | u8[16] | xxh3-128 de los metadatos |
| 64 | u64 | tamaño del fichero que esta cabecera da por bueno |
| 72 | u8[16] | identidad del fichero (§6) |
| 88 | u8[420] | reservado, 0 |
| 508 | u32 | CRC32 de los bytes 0–507 |

### Lectura

1. Se leen las dos cabeceras. Una es válida si su magic, su versión y su CRC cuadran, el
   fichero mide al menos `tamaño del fichero` y el xxh3-128 de sus metadatos coincide.
2. Se elige la válida con mayor secuencia. Si ninguna es válida, el fichero se rechaza.
3. Si la elegida dice «en curso», el fichero se rechaza aunque la otra diga «completo»: los
   pesos pueden haber cambiado ya. Solo una herramienta de reparación puede tocarlo (§7).

### Escritura

- **Solo metadatos** (sellar, añadir procedencia): metadatos nuevos al final, `fsync`;
  cabecera inactiva con secuencia + 1, estado completo, `fsync`.
- **Pesos en el sitio** (refinar sobre el propio fichero): metadatos con el paso «en curso» y la
  lista de tensores afectados, `fsync`; cabecera inactiva, estado en curso, `fsync`; pesos,
  `fsync`; metadatos con el paso cerrado y los hashes nuevos, `fsync`; cabecera inactiva,
  estado completo, `fsync`.

## 3. Metadatos

Empiezan con una tabla de secciones:

| Offset | Tipo | Campo |
|---:|---|---|
| 0 | u8[4] | magic `HNFM` |
| 4 | u32 | versión = 1 |
| 8 | u32 | número de secciones |
| 12 | u32 | reservado |
| 16 | {u32 tipo, u32 reservado, u64 offset, u64 tamaño}[n] | offsets relativos al inicio de los metadatos |

Tipos de sección:

| Tipo | Nombre | Contenido |
|---:|---|---|
| 1 | STRINGS | cadenas UTF-8 seguidas; se citan como {u32 offset, u32 longitud} dentro de esta sección |
| 2 | COMPONENTS | tabla de componentes (§4) |
| 3 | TENSORS | índice de tensores (§5) |
| 4 | MEMORY | fórmulas de memoria (§8) |
| 5 | PROVENANCE | pasos de procedencia (§9) |
| 6 | PROPERTIES | propiedades tipadas por componente (§10) |

Una sección de tipo desconocido se ignora si su bit de opcional (31) está activo; si no, el
fichero se rechaza.

## 4. Componentes

Cada entrada (100 bytes):

| Campo | Tipo |
|---|---|
| id | u32, único en el fichero |
| tipo | u32: 1 texto, 2 visión, 3 audio, 4 vídeo, 16 tokenizer, 17 configuración, 18 plantilla |
| flags | u32: bit 0 obligatorio |
| nombre | cadena |
| payload | u64 offset, u64 tamaño, u32 alineación |
| bytes de pesos en GPU | u64 |
| fórmula de memoria | u32, índice en MEMORY o 0xFFFFFFFF |
| primer tensor, número de tensores | u32, u32 |
| depende de | u32, id de otro componente o 0xFFFFFFFF (la visión depende del texto) |
| compuesto | u8[16], xxh3-128 de sus entradas de TENSORS en orden (§6) |
| hash de bytes | u8[16], xxh3-128 del payload si no tiene tensores (tokenizer, plantilla, configuración) |

El tokenizer es HTF3 tal cual. La configuración es el bloque binario de hints actual (magic
`TNIH`, GM4X y GM4V) tal cual. La plantilla son los bytes UTF-8 exactos del `chat_template.jinja`.

## 5. Tensores

Cada entrada (128 bytes):

| Campo | Tipo |
|---|---|
| nombre canónico | cadena |
| componente | u32 |
| capa | i32, −1 si no es de una capa |
| dtype | cadena (`hq44k_g16`, `hq8k`, `fp16`…) |
| rango | u32 |
| forma física | u64[4] |
| columnas lógicas de la última dimensión | u64 (relleno de GM4X) |
| offset, tamaño | u64, u64, absolutos |
| expertos | u32: 0 si no es MoE; si no, la primera dimensión, con salto = tamaño / expertos |
| hash de bytes | u8[16], xxh3-128 |
| origen | cadena: nombre en el checkpoint |
| hash del origen | u64, xxh3-64 de los bytes de origen (como hoy en `source_tensors`) |

La capa y el componente son campos: nadie los deduce del nombre.

## 6. Hashes

| Hash | Sobre qué | Se comprueba |
|---|---|---|
| CRC32 de cabecera | bytes 0–507 | siempre |
| xxh3-128 de metadatos | toda la región de metadatos | siempre |
| Compuesto por componente | xxh3-128 de la concatenación, en orden, de (offset, tamaño, hash de bytes) de cada tensor | siempre: sale de los metadatos |
| Hash de bytes por tensor | sus bytes | bajo demanda (`--verify`), y siempre en las herramientas que los modifican |
| Identidad del fichero | xxh3-128 de (hash de metadatos sin la procedencia, compuestos de todos los componentes) | entrada y salida de cada paso de procedencia |

La identidad excluye la procedencia: añadir un paso de sellado no cambia qué es el modelo.
Detectan corrupción, no manipulación.

## 7. Reparación

Si la cabecera más reciente dice «en curso», `hnf10 repair` recalcula los hashes de los tensores
afectados. Si todos coinciden con los antiguos, la escritura no llegó a empezar: se cierra el
paso como cancelado. Si todos coinciden con los nuevos, terminó: se cierra como completo. En
cualquier otro caso el fichero no se puede reparar y hay que regenerarlo.

## 8. Memoria de trabajo

Cada fórmula: bytes fijos y una lista de términos {variable u32, exponente u32, coeficiente
f64}, más el máximo admitido de cada variable que use. Variables: 1 tokens de contexto, 2
imágenes, 3 parches por imagen, 4 tokens por bloque de prefill. Ejemplo: la atención de la torre
visual lleva un término en parches² (cabezas × parches² × 2 bytes por imagen). El cargador evalúa
la fórmula con sus parámetros y rechaza lo que pase de los máximos. En lugar de términos, un
componente puede declarar una cota fija calculada en el máximo admitido.

## 9. Procedencia

Lista ordenada de pasos. Cada paso:

- tipo: convertir, capturar importancia, refinar, sellar, migrar;
- estado: completo, en curso o cancelado;
- herramienta, versión y commit;
- identidad de entrada y de salida;
- huella de la calibración;
- tensores afectados;
- parámetros como propiedades tipadas (§10).

Para el 26B refinado, el migrador escribe cuatro pasos: conversión, captura de importancia,
refinado y migración. Los valores salen del `refinement` de `hnf-seal` y del manifiesto v9.

## 10. Propiedades tipadas

Pares {clave: cadena, tipo: u8, valor}. Tipos: 1 bool (u8), 2 entero (i64), 3 real (f64), 4
cadena, 5 bytes (u32 longitud + datos). Sirven para parámetros de procedencia y para los campos
de configuración que todavía no tienen sitio en `TNIH`. Cada clave tiene una sola autoridad: si
está en `TNIH`, no puede repetirse aquí.

## 11. Inventario de la configuración del 26B (2026-11-02)

Héctor lee primero los hints binarios y después aplica encima el objeto `text` del JSON: no es
un plan de reserva. De las 37 claves de `text`:

- **15** las trae también el binario y el JSON las sobrescribe con el mismo nombre: `arch`,
  `attention_type`, `dtype`, `head_dim`, `hidden_size`, `intermediate_size`,
  `mlp_activation`, `mlp_type`, `norm_type`, `num_attention_heads`, `num_hidden_layers`,
  `num_key_value_heads`, `rms_norm_eps`, `tie_word_embeddings`, `vocab_size`.
- **1** solo existe en el JSON: `final_logit_softcapping`. Hoy funciona porque el motor usa 30
  por defecto, que es el valor del 26B. En HNFv10 va como propiedad tipada de la configuración
  hasta que `TNIH` tenga su campo.
- **21** no las lee el motor: son descripción, ya presente en GM4X, o procedencia.

Antes de que el migrador dé por buena la configuración hay que comprobar que, para las 15
claves repetidas, el valor del binario coincide con el del JSON. Si no coinciden, hoy manda el
JSON, y esa es la configuración efectiva que el HNFv10 tiene que reproducir.

## 12. Primera migración: Gemma 4 26B refinado (2026-11-03)

Migrado con `hnf10 migrate`, desde el HNFv9 sellado que cargaba HEXOS
(`gemma4-26b-moe-dense-hq8-hector-refined`), en 1,5 minutos.

- **Configuración:** la configuración de texto que regenera `TextModelConfigBin::from_json`
  desde el JSON coincide byte a byte con la de `TNIH`, y los flags de modalidad coinciden.
  `final_logit_softcapping` pasa a propiedad tipada de la configuración (30).
- **Pesos:** los 657 tensores, el tokenizer y la configuración son idénticos byte a byte al v9.
  `hnf10 verify` recalcula todos los hashes en 14 s.
- **Plantilla:** `chat_template.jinja` entra como componente opcional, con sus 18.683 bytes.
- **Procedencia:** cinco pasos. Conversión, captura de importancia, refinado y sellado salen
  del manifiesto v9; el último es la migración.
- **Motor:** Héctor lee HNFv10 (`src/hnf10.cpp`).
  - `helios_model_probe` da la misma salida con el v9 y con el v10.
  - `hnf_config_dump` muestra la configuración efectiva. Solo difieren:
    - las claves que solo traía el JSON y que el motor no lee;
    - `mlp_activation`: `gelu_new` en v9, porque el JSON se aplicaba encima, y `gelu` en v10,
      porque `TNIH` no distingue las dos. El motor envía las dos al mismo kernel `add_gelu`.
      Si algún modelo necesita la `gelu` exacta, hará falta un campo en `TNIH`.
  - **Regresión de logits:** 5 casos del conjunto final, unos 646 MB de logits, idénticos bit a
    bit entre v9 y v10 con el mismo `tune.cache`.
- HEXOS carga el HNFv10 desde el 2026-11-03:
  `helios_convert_v10/output/gemma4-26b-moe-dense-hq8-gptq.v10.hnf`, con su `.sha256`.

## 13. Cadena completa en HNFv10 (2026-11-03)

Desde safetensors, sin el v9.1 ni scripts:

1. `hnf10 convert --dense-hq8 --importance importancia-hector-v1`: frente al HNFv10 de HEXOS,
   597 tensores iguales, 60 tensores HQ44 de expertos con la misma rejilla (cabeceras de bloque
   idénticas; solo cambian los códigos, que en aquel están refinados), y tokenizer,
   configuración y plantilla iguales.
2. `hnf10 begin-edit` sobre esa conversión, `hnf_refine` de las 30 capas en 41 minutos y
   `hnf10 seal --record`: 60 tensores cambiados.
3. `hnf10 compare` con el HNFv10 de HEXOS, recalculando antes los bytes de los dos: **657
   tensores iguales**, mismos descriptores e inventario, y tokenizer, configuración y plantilla
   iguales.

La cadena reproduce el modelo bit a bit.

## 14. Independiente del v9.1 (2026-11-03)

`helios_convert_v10` ya no depende de `helios_convert_v9.1`. Los módulos que usa (codificadores
`hqs`, `mapping`, `safetensor`, el tokenizer `htf`, la configuración binaria `hints`,
`dictionary`, `base_id` y el plan `moe_experimental`) se trajeron tal como estaban en el commit
`79835e4` de la rama `gemma4-26b-moe`, sin el escritor HNFv9 ni sus binarios. Desde aquí
evolucionan en el v10. El v9.1 queda como pieza histórica y no se vuelve a tocar. Cuando el conversor v10 funcione de punta a punta y Héctor
esté integrado y probado solo con HNFv10 (incluidos los HNF de prueba del motor regenerados en v10), se
quitará toda referencia al v9.1: la ruta v9 del cargador de Héctor y el uso de HNFv9 en las
herramientas. Decisión de Andrés, 2026-11-03. Pruebas: 62
de los módulos traídos y 6 del contenedor.

Pendiente:
- Revisión de GPT (2026-11-03) corregida: las confirmaciones van en una `Session` con cerrojo exclusivo
  y comprueban en disco la secuencia vigente; cada tensor debe estar en exactamente un rango de
  componente y dentro de la zona de payloads; las fórmulas de memoria se validan (variables, exponentes,
  coeficientes y máximos). Lo mismo en el lector del motor.
- Segunda revisión de GPT (2026-11-03) corregida:
  - `seal` y `repair` comprueban también el hash del tokenizer, la configuración y la plantilla
    antes de cerrar;
  - el índice rechaza nombres repetidos, dtypes desconocidos, tamaños que no cuadran con dtype y
    forma, y más columnas lógicas que físicas, también en el lector del motor;
  - `compare` exige el mismo inventario en los dos sentidos y los mismos descriptores, y antes de
    comparar recalcula los hashes de los bytes de los dos ficheros.
- Ya no: `hnf_refine` sobre HNFv10 con `hnf10 begin-edit` y `hnf10 seal` funciona; con las capas 0
  y 1 del 26B da 0 tensores cambiados.
- `hnf_refine` y `hnf-seal` siguen escribiendo HNFv9; falta pasarlos a v10 con los pasos
  «en curso» y «completo».
- El conversor de safetensors a HNFv10 directo.
- Las fórmulas de memoria y el reparto de GPUs con visión.
- Regenerar los HNF de prueba del motor.
