# Almacenamiento de modelos en el cluster

Estado a 2026-10-24 y plan para centralizar los modelos en el disco de 10 TB del Studio, cargarlos
de ahí directamente a RAM sin escribir en los SSD, y que siga valiendo cuando entren más Macs.

## 1. Situación medida

| Mac | SSD | Libre | Grandes consumidores |
|---|---|---|---|
| studio | 3.6 TB | **23 GB (100 %)** | `espetero-work/kld` 1130 GB (mlx 470, gguf 390, nemo 82, eqcache 82), HF cache 1264 GB (DSV4.1-Flash 475), `modelos` 405, `espetero-work/dense` 324, `espetero-verify` 186, `espetero-work/quant` 89 |
| macbook | 1.8 TB | **59 GB (97 %)** | HF cache 527 GB, `models` 300 (jev 197, clef 80), `modelos` 198 (q35-9b 158), `.cache/ultra-metal/runs` 82, `espetero-work` 60 |

Disco: **WD Elements 25A3, 10 TB, disco mecánico por USB a 5 Gb/s**, sin SMART por el puente USB.
Venía en NTFS (sólo lectura en macOS); formateado APFS `Almacen` el 2026-10-24.

Medido en el Studio (2026-10-24, lecturas F_NOCACHE en zonas no cacheadas):

| Patrón | MB/s |
|---|---|
| escritura secuencial (`dd` 16 MiB) | 276 |
| lectura secuencial, 1 hilo | 264-278 |
| lectura en trozos de 64 MB, 8 ó 16 hilos (patrón del loader de Espetero) | **176** |

Un modelo de 100 GB se carga en ~6 min leyendo secuencial y ~9.5 min con el patrón paralelo.

## 2. Principio: el SSD no se escribe con modelos

El desgaste de un SSD lo causan las **escrituras**; leer no lo gasta. Así que:

- **Todo modelo vive en el almacén y se carga de ahí a RAM directamente.** No se copia al SSD.
- **Las salidas de los jobs se escriben directamente en el almacén** (EQ quantize, calibración,
  dumps KLD, conversiones, descargas de HF). Son escrituras secuenciales a 276 MB/s.
- **Excepción: modelos que se ejecutan haciendo streaming de disco** (DSV4.1-Flash 475 GB,
  GLM-5.3-Flash FP8 328 GB y los que vengan). Leen expertos aleatorios durante la inferencia, que
  en un HDD sería inusable. Se escriben una vez en el SSD y se quedan ahí; ejecutarlos sólo lee.
  Tienen copia también en el almacén.
- El SSD queda para código, builds, venvs, logs pequeños y esos modelos en streaming.

```
       ┌──────────────────── Almacen (HDD 10 TB, studio) ────────────────────┐
       │ modelos HF y propios, EQ, calib, dumps KLD, runs                    │
       └───────┬──────────────────────────────────────────────▲─────────────┘
   carga secuencial a RAM                              salidas de jobs
   (studio local; macbook por red)                     (escritura directa)
       ┌───────▼──────────┐                         ┌──────────┴──────────┐
       │ RAM del Mac:     │                         │ job                 │
       │ pesos + caché de │────────────────────────▶│                     │
       │ ficheros (warm)  │                         └─────────────────────┘
       └──────────────────┘
       SSD: sólo modelos en streaming (escritos una vez), código, venvs
```

Tras la primera carga, un modelo cargado por mmap sigue en la caché de ficheros de la RAM: una
segunda carga del mismo modelo no toca el HDD mientras la memoria lo permita (gputurn ya agrupa
jobs por modelo "warm").

## 3. Lo que hace falta para que la carga desde HDD sea rápida

1. **Lectura secuencial en el loader.** El loader de Espetero lee en paralelo (trozos de 64 MB
   con `parallel_for`, `pread_uncached` y `parallel_copy` del mmap): en HDD eso son seeks y baja
   de 264 a 176 MB/s. Cambio: si el fichero está en un disco rotacional (o con
   `ESPETERO_LOAD_THREADS=1`), leer con un solo hilo y en orden de fichero, en trozos grandes.
   llama.cpp: `--no-mmap` lee secuencial; MLX: medir, o precalentar (abajo).
2. **Precalentado genérico** para herramientas que no controlamos (MLX, PyTorch, scripts): un
   paso previo `vault warm PATH` que lee los ficheros de principio a fin (1 hilo) para dejarlos
   en la caché de ficheros; luego el loader real lee de RAM. Sirve sólo si el Mac tiene RAM para
   pesos + caché; si no, mejor carga directa secuencial.
3. **Una sola carga a la vez por disco.** Dos jobs cargando a la vez del mismo HDD se reparten
   mal y van más lentos que uno detrás de otro. gputurn serializa las cargas del almacén
   (recurso `almacen` en la cola, independiente de la GPU).
4. **Swap.** Cargar modelos grandes a RAM con poca memoria libre hace swap al SSD, que sí es
   escritura. gputurn ya admite por presupuesto de memoria (`--mem`); mantenerlo honesto es lo
   que protege el SSD aquí.

### 3.1 Medido: carga desde el HDD (rama `perf/hdd-sequential-load`, 2026-10-25)

Modelo `mlx-community/Qwen3.6-35B-A3B-4bit` (MLX 4 bits, 20.4 GB, 4 shards) en
`/Volumes/Almacen/models/`, cargado en studio (M4 Max) con `bench_native --load-only` a través de
gputurn (#525), con F1 pausado. Caché fría antes de cada carga: `mmap` + `msync(MS_INVALIDATE)` sobre
cada shard (sin root; `mincore` 1.0 → 0.0). Dos repeticiones por brazo, intercaladas:

| Lectura | Carga (s) | MB/s efectivos |
|---|---|---|
| Antes: paralelo, trozos de 8/64 MB (`ESPETERO_LOAD_THREADS=0`) | 143.2 / 136.5 | ~146 |
| 1 hilo, orden de fichero, 64 MB por tensor (`ESPETERO_LOAD_WARM=0`) | 92.0 / 91.9 | ~222 |
| Precalentado: shards de principio a fin y luego carga desde la caché (por defecto) | 78.5 / 78.3 | ~261 |

El precalentado lee los 20.4 GB en 73.5 s (278 MB/s, el máximo secuencial del disco); la carga
posterior desde la caché tarda ~5 s. Sin él, el loader lee en orden de capas, y los shards están en
orden de nombre (capas 0, 1, 10, 11, …), así que el cabezal salta entre tensores.

Qué hace el código (`src/model/impl/common.h`, `tensors.h`):
- `load_readers(fd)` decide una vez por volumen: 1 lector si el disco es rotacional o USB (IOKit,
  probado: el HDD del almacén da rotacional/USB y el SSD interno no), un recurso de red (smbfs, nfs,
  afpfs, webdav) o una raíz del almacén (`ESPETERO_VAULT_ROOTS`, por defecto `/Volumes/Almacen`). En
  cualquier otro caso, el camino paralelo de siempre. `ESPETERO_LOAD_THREADS=N` lo fija.
- Con 1 lector, `parallel_copy`, `prefetch_mapped`, `pread_uncached`, `tensor_uncached` (GGUF) y la
  tabla n-gram en GPU leen en orden de fichero en preads de 64 MB.
- Precalentado de safetensors: sólo en volúmenes lentos, si los ficheros caben en la mitad de la RAM
  (si no, carga directa secuencial); `ESPETERO_LOAD_WARM=0` lo apaga. Llena la caché de ficheros (RAM,
  liberable), nunca el SSD. Los GGUF no se precalientan.
- Lectura de expertos en inferencia (DSV4, Qwen4-Exp) sin cambios: esos modelos viven en el SSD.

Pendiente: no he medido el acceso por SMB desde el MacBook ni GGUF desde el HDD. La tabla mide el
precalentado con `ESPETERO_LOAD_WARM=1`; como valor por defecto (mismo código) no se ha vuelto a
cargar desde el HDD.

## 4. Acceso desde otros Macs

El MacBook (y Macs futuros) cargan desde el almacén del Studio por red:

- **Compartir `/Volumes/Almacen` por SMB** en el Studio (Ajustes > General > Compartir > Compartir
  archivos) y montarlo en el MacBook como `/Volumes/Almacen` (mismo path en todos los Macs: los
  `--model PATH` valen en cualquiera).
- Por el enlace Thunderbolt (~10 Gbit/s) el cuello sigue siendo el HDD (~2.2 Gbit/s). Por WiFi o
  1 GbE la carga baja a ~110 MB/s: hay que usar el cable.
- Las cargas remotas compiten con las locales por el mismo disco: la cola de §3.3 es del disco,
  no del Mac.
- Escrituras de salida desde el MacBook: también al almacén por SMB.

## 5. Formato y montaje del disco (hecho)

```bash
diskutil eraseDisk APFS Almacen GPT disk4
```

```bash
sudo mdutil -i off /Volumes/Almacen
```

- Hecho 2026-10-24: APFS `Almacen` (disk5s1), `.almacen-id` en la raíz, Spotlight desactivado.
- Opcional: `sudo pmset -a disksleep 0` si el arranque del disco tras dormirse (~10 s) molesta.

## 6. Estructura del almacén

```
/Volumes/Almacen/
  hf/hub/                    caché HF (HF_HUB_CACHE de los jobs apunta aquí)
  models/<familia>/<nombre>/ modelos propios o convertidos (jev/, clef/, lfm2/, nemotron/…)
  eq/<modelo>/<variante>-<commit>/   salidas de EQ quantize
  calib/<modelo>/            ficheros de calibración
  kld/<modelo>/<formato>/    dumps de logits/KLD de referencia
  runs/<fecha>-<tema>/       resultados de experimentos que se quieren guardar
  tools/                     vault.py, vault-f1.sh, readpat.py
  MANIFEST/                  un JSON por artefacto
```

En la migración inicial cada tanda conserva su árbol de origen bajo la categoría
(`kld/studio-kld/` = `studio:~/espetero-work/kld/`); reordenar por modelo queda para cuando se
sepa qué es cada cosa. `tools/vault.py put|verify|status` copia con rsync y compara el sha256 de
cada fichero a ambos lados; `tools/vault-f1.sh` son las tandas de F1 (log en `studio:~/vault-f1.log`).

Cada artefacto lleva `MANIFEST/<ruta-aplanada>.json`: origen (host y path, o repo HF@rev, o
comando + commit), bytes, sha256 por fichero, symlinks, `verified`, y `reproducible: bool` para
marcar lo que no se puede volver a generar barato (§9).

## 7. Integración en gputurn (generaliza al cluster)

Config por cluster:

```json
"vaults": [{"name": "almacen", "host": "studio", "path": "/Volumes/Almacen", "kind": "hdd", "load_slots": 1}]
```

Config por Mac: `"model_roots": ["/Volumes/Almacen/models", "/Volumes/Almacen/hf/hub", "~/models"]`,
`"pinned_ssd": ["deepseek-ai/DeepSeek-V4.1-Flash-1017", "glm5-fp8"]`.

- `gputurn models`: el almacén aparece como una ubicación compartida por todos los Macs que lo
  montan; los modelos en streaming, como copias en SSD.
- Un job con `--model NAME` cuyo modelo está en el almacén toma el slot de carga del disco
  mientras carga (hasta que el job lo suelta o pasa un tiempo según el tamaño); los demás jobs
  que necesiten cargar del mismo disco esperan; los que ya tienen el modelo warm, no.
- Preferir el Mac que ya tiene el modelo warm en RAM (ahorra la carga entera).
- Antes de admitir un job que lee o escribe en el almacén: comprobar que está montado (`.almacen-id`);
  si no, fallar claro en vez de escribir en un `/Volumes/Almacen` vacío del SSD.
- `gputurn fetch REPO[@rev]`: descarga de HF directamente a `hf/hub` del almacén.
- `--out` / `$VAULT_OUT` para jobs que producen artefactos: directorio en el almacén y MANIFEST
  al acabar.

Crecer el cluster:

- Un Mac nuevo monta el almacén por SMB en el mismo path y declara sus `model_roots`.
- Más almacenes = más entradas en `vaults`, cada uno con su cola de carga. Repartir modelos entre
  discos según uso multiplica el ancho de banda de carga.
- Un NAS rápido (RAID o SSD por 10GbE/Thunderbolt, >1 GB/s) cambia `kind` y `load_slots`; el
  esquema sigue igual.

## 8. Reglas para agentes (CLAUDE.md global)

- No descargar ni escribir modelos ni artefactos grandes en el SSD: `HF_HUB_CACHE=/Volumes/Almacen/hf/hub`,
  salidas a `/Volumes/Almacen/{eq,calib,kld,runs}/…`.
- Sólo los modelos en streaming de la lista `pinned_ssd` viven en el SSD.
- Cargar modelos con `--model NAME`: gputurn pone en cola la carga del disco.
- El código sigue en el MacBook y se sincroniza con `--sync-tree` como hasta ahora.

## 9. Riesgos

- **Una sola copia en un HDD USB.** Lo descargable de HF y lo regenerable (dumps KLD) se puede
  rehacer; salidas EQ publicables, calibraciones y modelos propios (`reproducible: false`)
  necesitan segunda copia: un segundo disco o un bucket (B2/R2) sólo para ese subconjunto.
- El disco se desmonta si el Studio duerme o se desconecta el USB; con el MacBook montándolo por
  SMB, el Studio no debe dormir (`pmset sleep 0`).
- Cargas lentas frente al SSD: 100 GB son ~6 min. Compensa reutilizar modelos warm y no
  alternar modelos grandes en jobs seguidos.

## 10. Migración por fases

| Fase | Qué | Estado |
|---|---|---|
| F0 | Formatear APFS, Spotlight fuera, medir | hecho 2026-10-24 |
| F1 | Studio → almacén: `espetero-work/{kld,kld2,dense,quant,hesstest}`, cada carpeta de `modelos`, `espetero-verify/scratch`. Copiar, verificar sha256, borrar del SSD por tandas con OK de Fran | copiando |
| F1b | HF cache del Studio (formato con `hub/blobs` compartido; inspeccionar antes) salvo DSV4 | pendiente |
| F2 | MacBook → almacén por TB: HF cache, `models`, `modelos`, `ultra-metal/runs`, `espetero-work` | necesita SMB montado |
| F3 | Espetero: lectura secuencial desde HDD en el loader (§3.1) y medir carga real HDD vs SSD | pendiente |
| F4 | gputurn: `vaults`, cola de carga por disco, comprobación de montaje, `fetch`, `--out` | pendiente |
| F5 | Reglas de §8 en CLAUDE.md global y en `gputurn agents --install` | pendiente |
