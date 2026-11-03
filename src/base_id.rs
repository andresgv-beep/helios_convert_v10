// src/base_id.rs
// ============================================================================
// base_id: huella del modelo ORIGINAL, igual para todas sus cuantizaciones
// ============================================================================
//
// Un LoRA (.hlora, ver Hector-engine/HLORA_SPEC.md) solo vale para los pesos
// exactos con los que se entrenó. El manifiesto del .hnf cambia con la
// cuantización y las formas no distinguen un reentrenamiento del original, así
// que la huella sale del CONTENIDO del checkpoint de origen:
//
//   XXH3-128 de "hbase1\n"
//            + por cada tensor, en orden de nombre:
//                nombre \n dtype \n forma,separada,por,comas \n XXH3-128(bytes) \n
//            + "tokenizer\n" + XXH3-128(tokenizer.json) \n   (o "-" si no hay)
//
// No es una firma criptográfica: empareja adaptador y modelo, no protege de
// nadie. Se hashea tensor a tensor, así que no depende de cómo se repartieron
// los shards.

use std::path::Path;

use anyhow::Result;
use rayon::prelude::*;
use xxhash_rust::xxh3::{xxh3_128, Xxh3Default};

use crate::safetensor::SafetensorReader;

pub const PREFIX: &str = "hbase1:";

/// Huella de un checkpoint safetensors (la carpeta del modelo de texto).
pub fn compute(model_dir: &Path) -> Result<String> {
    let reader = SafetensorReader::from_folder(model_dir)?;
    let tensors: Vec<(&str, String, Vec<usize>)> = reader
        .iter_tensors()
        .map(|(name, info)| (name, info.dtype.clone(), info.shape.clone()))
        .collect();
    // El orden lo fija iter_tensors (alfabético); el hash de cada tensor es
    // independiente, así que se calcula en paralelo y se combina en ese orden.
    let hashes: Vec<u128> = tensors
        .par_iter()
        .map(|(name, _, _)| reader.read_raw(name).map(xxh3_128))
        .collect::<Result<_>>()?;
    let mut total = Xxh3Default::new();
    total.update(b"hbase1\n");
    for ((name, dtype, shape), hash) in tensors.iter().zip(&hashes) {
        let shape = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(",");
        total.update(format!("{name}\n{dtype}\n{shape}\n{hash:032x}\n").as_bytes());
    }
    let tokenizer = model_dir.join("tokenizer.json");
    let tokenizer_hash = if tokenizer.is_file() {
        format!("{:032x}", xxh3_128(&std::fs::read(&tokenizer)?))
    } else {
        "-".to_string()
    };
    total.update(format!("tokenizer\n{tokenizer_hash}\n").as_bytes());
    Ok(format!("{PREFIX}{:032x}", total.digest128()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Safetensors mínimo: cabecera JSON + datos.
    fn write_st(dir: &Path, file: &str, tensors: &[(&str, &[u8])]) {
        let mut header = serde_json::Map::new();
        let mut offset = 0usize;
        for (name, data) in tensors {
            header.insert(name.to_string(), serde_json::json!({
                "dtype": "BF16", "shape": [data.len() / 2], "data_offsets": [offset, offset + data.len()]}));
            offset += data.len();
        }
        let json = serde_json::to_vec(&header).unwrap();
        let mut f = std::fs::File::create(dir.join(file)).unwrap();
        f.write_all(&(json.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&json).unwrap();
        for (_, data) in tensors {
            f.write_all(data).unwrap();
        }
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("hbase-test-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn same_weights_same_id_regardless_of_shards() {
        let a = tmp("a");
        write_st(&a, "model.safetensors", &[("x", &[1, 2, 3, 4]), ("y", &[5, 6])]);
        let b = tmp("b");
        write_st(&b, "model-00001-of-00002.safetensors", &[("y", &[5, 6])]);
        write_st(&b, "model-00002-of-00002.safetensors", &[("x", &[1, 2, 3, 4])]);
        let id = compute(&a).unwrap();
        assert!(id.starts_with(PREFIX) && id.len() == PREFIX.len() + 32);
        assert_eq!(id, compute(&b).unwrap());
    }

    #[test]
    fn one_changed_byte_or_tokenizer_changes_the_id() {
        let a = tmp("c");
        write_st(&a, "model.safetensors", &[("x", &[1, 2, 3, 4])]);
        let base = compute(&a).unwrap();
        let b = tmp("d");
        write_st(&b, "model.safetensors", &[("x", &[1, 2, 3, 5])]);
        assert_ne!(base, compute(&b).unwrap(), "mismas formas, otro contenido");
        std::fs::write(a.join("tokenizer.json"), b"{}").unwrap();
        assert_ne!(base, compute(&a).unwrap(), "el tokenizador también cuenta");
    }
}
