//! Visión de Gemma 4 como componente HNFv10 (docs/HNFV10.md §4, §8).
//!
//! `add_vision` copia un HNFv10 de texto con sus mismos bytes (tensores, tokenizer y plantilla)
//! y le añade la torre visual del checkpoint como componente VISION opcional, que depende del
//! texto y declara su fórmula de memoria. La configuración TNIH gana la parte de visión
//! (VisionModelConfigBin y GM4V) detrás de la de texto, cuyos bytes no cambian. Así el 26B
//! refinado con GPTQ gana visión sin volver a convertir ni a refinar.
use crate::file::{open, verify_tensors, Builder};
use crate::format::{align_up, xxh3_128};
use crate::hints::binary::append_vision;
use crate::mapping::gemma4_vision::Gemma4VisionMapper;
use crate::mapping::ModelMapper;
use crate::metadata::{kind, step, Component, MemoryFormula, Metadata, Property, Step, Tensor, Value};
use crate::SafetensorReader;
use anyhow::{ensure, Context, Result};
use half::{bf16, f16};
use std::os::unix::fs::FileExt;
use std::path::Path;

/// Cada tensor de la torre empieza alineado a 256 bytes dentro del componente.
const TENSOR_ALIGN: u64 = 256;
/// Variables de las fórmulas de memoria (§8).
const IMAGES: u32 = 2;
const PATCHES: u32 = 3;

pub struct Report {
    pub tensors: usize,
    pub weight_bytes: u64,
    /// Valores que FP16 no guarda exactos (subnormales del BF16 original).
    pub inexact: u64,
    pub scratch_bytes_at_max: u64,
    pub identity: [u8; 16],
}

fn source_f32(raw: &[u8], dtype: &str, i: usize) -> f32 {
    let bytes = [raw[2 * i], raw[2 * i + 1]];
    if dtype == "BF16" { bf16::from_le_bytes(bytes).to_f32() } else { f16::from_le_bytes(bytes).to_f32() }
}

/// Memoria de trabajo del runner del motor (`Gemma4VisionRunner`), que reserva para el
/// máximo de parches. Por parche: subida FP32 y FP16 de los píxeles, posiciones, el
/// residual, norma, entrada del lineal, proyección, Q/K/V, salida de atención y gate/up en
/// FP16, y por cada soft token (un parche de cada pool²) el pool FP32 y FP16 y la proyección.
/// Por parche²: las puntuaciones de atención de todas las cabezas, en FP16.
fn memory(hints: &serde_json::Value) -> Result<(MemoryFormula, u64)> {
    let get = |k: &str| hints[k].as_u64().with_context(|| format!("falta {k} en los hints de visión"));
    let (hidden, heads, inter, patch) = (get("hidden_size")?, get("num_attention_heads")?, get("intermediate_size")?, get("patch_size")?);
    let (pool, projection, max_patches) = (get("pooling_kernel_size")?, get("projection_dim")?, get("max_patches")?);
    let width = 3 * patch * patch;
    let per_patch = width * 4 + width * 2 + 8 + hidden * 2 * 8 + hidden.max(inter) * 2 + inter * 2 * 2;
    let per_soft = hidden * 4 + hidden * 2 + projection * 2;
    let linear = per_patch as f64 + per_soft as f64 / (pool * pool) as f64;
    let square = (heads * 2) as f64;
    let formula = MemoryFormula { fixed: 0, terms: vec![(PATCHES, 1, linear), (PATCHES, 2, square)],
        maxima: vec![(IMAGES, 1), (PATCHES, max_patches)] };
    let at_max = (linear * max_patches as f64 + square * (max_patches * max_patches) as f64).ceil() as u64;
    Ok((formula, at_max))
}

/// La torre visual de un checkpoint de Gemma 4, lista para escribirse como componente VISION.
/// La usan `add_vision` (sobre un HNF de texto ya hecho) y la conversión densa (en el mismo paso).
pub struct Tower {
    /// Hints de visión para el TNIH (`append_vision`).
    pub hints: serde_json::Value,
    reader: SafetensorReader,
    /// (nombre canónico, nombre en el checkpoint, capa), globales primero y luego capa a capa.
    planned: Vec<(String, String, Option<usize>)>,
    /// (offset dentro del componente, bytes) de cada tensor.
    layout: Vec<(u64, u64)>,
    size: u64,
    formula: MemoryFormula,
    pub scratch_bytes_at_max: u64,
    /// `use_bidirectional_attention` del texto, si lo declara («vision» en el 26B).
    bidirectional: Option<String>,
}

/// Dónde quedó escrita la torre.
pub struct Written {
    at: u64,
    hashes: Vec<([u8; 16], u64)>,
    /// Valores que FP16 no guarda exactos (subnormales del BF16 original).
    pub inexact: u64,
}

impl Tower {
    pub fn plan(model: &Path) -> Result<Self> {
        let root: serde_json::Value = serde_json::from_slice(&std::fs::read(model.join("config.json"))?)?;
        let mapper = Gemma4VisionMapper::from_model_path(model, &root)?;
        let hints = mapper.execution_hints();
        let (formula, scratch_bytes_at_max) = memory(&hints)?;
        let reader = SafetensorReader::from_folder(model)?;
        let mut planned: Vec<(String, String, Option<usize>)> = reader.iter_tensors()
            .filter_map(|(name, _)| mapper.map_tensor(name).map(|m| (m.canonical_name, name.to_string(), m.layer_idx)))
            .collect();
        planned.sort_by(|a, b| (a.2.map_or(-1, |l| l as i64), &a.0).cmp(&(b.2.map_or(-1, |l| l as i64), &b.0)));
        ensure!(Some(planned.len()) == mapper.expected_tensor_count(),
            "{} tensores de visión, se esperaban {:?}", planned.len(), mapper.expected_tensor_count());
        let mut layout = Vec::with_capacity(planned.len());
        let mut end = 0u64;
        for (_, source_name, _) in &planned {
            let info = reader.tensor_info(source_name).context("tensor sin info")?;
            ensure!(info.dtype == "BF16" || info.dtype == "F16", "{source_name}: dtype {} no admitido", info.dtype);
            let elements: u64 = info.shape.iter().map(|&d| d as u64).product();
            let at = align_up(end, TENSOR_ALIGN);
            layout.push((at, elements * 2));
            end = at + elements * 2;
        }
        let bidirectional = root["text_config"]["use_bidirectional_attention"].as_str().map(String::from);
        Ok(Tower { hints, reader, planned, layout, size: end, formula, scratch_bytes_at_max, bidirectional })
    }

    pub fn tensors(&self) -> usize { self.planned.len() }
    pub fn weight_bytes(&self) -> u64 { self.layout.iter().map(|l| l.1).sum() }

    /// Escribe los pesos en FP16 como un payload nuevo del fichero.
    pub fn write(&self, builder: &mut Builder) -> Result<Written> {
        let mut inexact = 0u64;
        let mut hashes = Vec::with_capacity(self.planned.len());
        let (at, _) = builder.payload(self.size, 4096, |w| {
            let mut written = 0u64;
            for ((canonical, source_name, _), &(at, size)) in self.planned.iter().zip(&self.layout) {
                w.write_all(&vec![0u8; (at - written) as usize])?;
                let dtype = &self.reader.tensor_info(source_name).unwrap().dtype;
                let raw = self.reader.read_raw(source_name)?;
                let mut bytes = Vec::with_capacity(size as usize);
                // std_bias y std_scale entran en FP16 sin redondeo en el 26B: se exige.
                let exact = canonical.contains(".std_");
                for i in 0..raw.len() / 2 {
                    let x = source_f32(raw, dtype, i);
                    let y = f16::from_f32(x);
                    ensure!(x.is_finite() && y.is_finite(), "{canonical}: valor fuera de FP16");
                    if y.to_f32() != x {
                        ensure!(!exact, "{canonical}: {x} no es exacto en FP16");
                        inexact += 1;
                    }
                    bytes.extend_from_slice(&y.to_le_bytes());
                }
                hashes.push((xxh3_128(&bytes), xxhash_rust::xxh3::xxh3_64(raw)));
                w.write_all(&bytes)?;
                written = at + size;
            }
            Ok(())
        })?;
        Ok(Written { at, hashes, inexact })
    }

    /// Componente VISION (opcional, depende del texto), sus tensores, su fórmula de memoria y
    /// la propiedad `use_bidirectional_attention` en la configuración. Los compuestos los
    /// recalcula quien cierre el fichero.
    pub fn register(&self, md: &mut Metadata, written: &Written, text_id: u32, config_id: u32) -> Result<()> {
        let vision_id = md.components.iter().map(|c| c.id).max().unwrap_or(0) + 1;
        let first = md.tensors.len() as u32;
        for (i, (canonical, source_name, layer)) in self.planned.iter().enumerate() {
            // Los clamps del E2B/E4B son escalares; HNFv10 no tiene rango 0 y los guarda como [1].
            let mut shape: Vec<u64> = self.reader.tensor_info(source_name).unwrap().shape.iter().map(|&d| d as u64).collect();
            if shape.is_empty() { shape.push(1); }
            md.tensors.push(Tensor { name: canonical.clone(), component: vision_id, layer: layer.map_or(-1, |l| l as i32),
                dtype: "fp16".into(), logical_last: *shape.last().unwrap(), shape, offset: written.at + self.layout[i].0,
                size: self.layout[i].1, experts: 0, hash: written.hashes[i].0, source: source_name.clone(),
                source_hash: written.hashes[i].1 });
        }
        md.memory.push(self.formula.clone());
        md.components.push(Component { id: vision_id, kind: kind::VISION, required: false, name: "visión Gemma 4".into(),
            payload_offset: written.at, payload_size: self.size, alignment: 4096, weight_bytes: self.weight_bytes(),
            memory: Some(md.memory.len() as u32 - 1), first_tensor: first, tensor_count: self.planned.len() as u32,
            depends_on: Some(text_id), composite: [0; 16], bytes_hash: [0; 16] });
        // Cómo atiende el decoder a los soft tokens: «vision» (26B) los deja verse entre sí en las
        // capas deslizantes. TNIH no tiene campo para esto, así que va como propiedad tipada.
        if let Some(mode) = &self.bidirectional {
            ensure!(!md.properties.iter().any(|(_, p)| p.key == "use_bidirectional_attention"), "use_bidirectional_attention ya está");
            md.properties.push((config_id, Property { key: "use_bidirectional_attention".into(), value: Value::Str(mode.clone()) }));
        }
        Ok(())
    }
}

pub fn add_vision(from: &Path, model: &Path, to: &Path, commit: &str) -> Result<Report> {
    ensure!(!to.exists(), "{} ya existe", to.display());
    let source = open(from)?;
    let src = std::fs::File::open(from)?;
    let mut md = source.metadata.clone();
    ensure!(md.components.iter().all(|c| c.kind != kind::VISION), "{} ya tiene visión", from.display());
    let text_id = md.components.iter().find(|c| c.kind == kind::TEXT).context("sin componente de texto")?.id;
    let config_index = md.components.iter().position(|c| c.kind == kind::CONFIG).context("sin configuración")?;

    let tower = Tower::plan(model)?;
    let old_config = {
        let c = &md.components[config_index];
        let mut b = vec![0u8; usize::try_from(c.payload_size)?];
        src.read_exact_at(&mut b, c.payload_offset)?;
        b
    };
    let config = append_vision(&old_config, &tower.hints)?;

    // Los componentes que ya había, con sus mismos bytes; la configuración, ampliada.
    let mut builder = Builder::create(to)?;
    let mut moved = Vec::with_capacity(md.components.len());
    for (i, c) in md.components.iter().enumerate() {
        let (offset, hash) = if i == config_index {
            builder.payload(config.len() as u64, u64::from(c.alignment), |w| { w.write_all(&config)?; Ok(()) })?
        } else {
            builder.payload(c.payload_size, u64::from(c.alignment), |w| {
                let mut at = 0u64;
                let mut buf = vec![0u8; 64 << 20];
                while at < c.payload_size {
                    let n = (c.payload_size - at).min(buf.len() as u64) as usize;
                    src.read_exact_at(&mut buf[..n], c.payload_offset + at)?;
                    w.write_all(&buf[..n])?;
                    at += n as u64;
                }
                Ok(())
            })?
        };
        if i != config_index && c.tensor_count == 0 { ensure!(hash == c.bytes_hash, "{} cambió al copiarlo", c.name); }
        moved.push((offset, hash));
    }
    for (i, c) in md.components.iter_mut().enumerate() {
        let delta = moved[i].0 as i128 - c.payload_offset as i128;
        for t in md.tensors.iter_mut().filter(|t| t.component == c.id) { t.offset = (t.offset as i128 + delta) as u64; }
        c.payload_offset = moved[i].0;
        if i == config_index { c.payload_size = config.len() as u64; }
        if c.tensor_count == 0 { c.bytes_hash = moved[i].1; }
    }

    let written = tower.write(&mut builder)?;
    let config_id = md.components[config_index].id;
    tower.register(&mut md, &written, text_id, config_id)?;
    for i in 0..md.components.len() { md.components[i].composite = md.composite(&md.components[i]); }

    let identity = md.identity();
    md.provenance.push(Step { kind: step::ADD_COMPONENT, state: step::COMPLETE, tool: "hnf10 add-vision".into(),
        version: env!("CARGO_PKG_VERSION").into(), commit: commit.into(), input: source.header.identity, output: identity,
        calibration: String::new(), touched: vec![], params: vec![
            Property { key: "source_model".into(), value: Value::Str(model.display().to_string()) },
            Property { key: "base_id".into(), value: Value::Str(crate::base_id::compute(model)?) },
            Property { key: "dtype".into(), value: Value::Str("fp16".into()) },
            Property { key: "fp16_inexact_values".into(), value: Value::Int(written.inexact as i64) },
        ] });
    builder.finish(md)?;

    let opened = open(to)?;
    ensure!(opened.header.identity == identity, "identidad distinta al reabrir");
    let bad = verify_tensors(to, &opened.metadata)?;
    ensure!(bad.is_empty(), "tensores que no cuadran: {bad:?}");
    Ok(Report { tensors: tower.tensors(), weight_bytes: tower.weight_bytes(), inexact: written.inexact,
        scratch_bytes_at_max: tower.scratch_bytes_at_max, identity })
}
