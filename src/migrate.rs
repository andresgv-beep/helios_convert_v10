//! Migración de un HNFv9 a HNFv10 sin tocar los pesos (docs/HNFV10.md §9).
//!
//! Copia byte a byte el bloque de texto, el tokenizer HTF3 y la configuración binaria
//! (`TNIH`), construye el índice desde el manifiesto v9 y traslada la procedencia, incluido
//! el `refinement` de `hnf-seal`. Es el único sitio donde se interpretan nombres v9 para
//! sacar la capa. Al terminar reabre el HNFv10 y compara cada payload con el original.
use crate::file::{open, verify_tensors, Builder};
use crate::format::xxh3_128;
use crate::metadata::{kind, step, Component, Metadata, Property, Step, Tensor, Value};
use anyhow::{bail, ensure, Context, Result};
use memmap2::Mmap;
use serde_json::Value as Json;
use std::fs::File;
use std::path::Path;

const V9_MAGIC: &[u8; 8] = b"HNFv9\0\0\0";
const TEXT: usize = 0;
const TOKENIZER: usize = 9;
const HINTS_JSON: usize = 10;
const HINTS_BIN: usize = 11;
const COPY_CHUNK: usize = 64 << 20;

pub struct Report {
    pub tensors: usize,
    pub text_bytes: u64,
    pub identity: [u8; 16],
    pub notes: Vec<String>,
}

#[derive(Clone, Copy)]
struct Block { offset: u64, size: u64, checksum: u64 }

fn u32_at(b: &[u8], o: usize) -> u32 { u32::from_le_bytes(b[o..o + 4].try_into().unwrap()) }
fn u64_at(b: &[u8], o: usize) -> u64 { u64::from_le_bytes(b[o..o + 8].try_into().unwrap()) }

fn section(data: &[u8], offset: u64, size: u64) -> Result<&[u8]> {
    let start = usize::try_from(offset)?;
    data.get(start..start + usize::try_from(size)?).context("rango fuera del HNFv9")
}

pub(crate) fn text_param(key: &str, value: &Json) -> Option<Property> {
    let value = match value {
        Json::Bool(b) => Value::Bool(*b),
        Json::Number(n) if n.is_i64() => Value::Int(n.as_i64()?),
        Json::Number(n) => Value::Real(n.as_f64()?),
        Json::String(s) => Value::Str(s.clone()),
        _ => return None,
    };
    Some(Property { key: key.into(), value })
}

/// Propiedades planas de un objeto JSON (un nivel, con prefijo), sin listas ni objetos.
pub(crate) fn flat(prefix: &str, object: &Json, out: &mut Vec<Property>) {
    if let Some(map) = object.as_object() {
        for (k, v) in map {
            let key = if prefix.is_empty() { k.clone() } else { format!("{prefix}.{k}") };
            if v.is_object() { flat(&key, v, out) } else if let Some(p) = text_param(&key, v) { out.push(p) }
        }
    }
}

fn layer_of(name: &str) -> i32 {
    name.strip_prefix("text.layer").and_then(|r| r.split('.').next()).and_then(|n| n.parse().ok()).unwrap_or(-1)
}

pub fn migrate(from: &Path, to: &Path, template: Option<&Path>, commit: &str) -> Result<Report> {
    let file = File::open(from).with_context(|| format!("no abro {}", from.display()))?;
    let data = unsafe { Mmap::map(&file)? };
    ensure!(data.len() >= 576 && &data[0..8] == V9_MAGIC, "{} no es un HNFv9", from.display());
    let (manifest_offset, manifest_size, file_size) = (u64_at(&data, 32), u64_at(&data, 40), u64_at(&data, 48));
    ensure!(file_size == data.len() as u64, "el HNFv9 no mide lo que dice su cabecera");
    let blocks: Vec<Block> = (0..16).map(|i| {
        let at = 64 + 32 * i;
        Block { offset: u64_at(&data, at + 8), size: u64_at(&data, at + 16), checksum: u64_at(&data, at + 24) }
    }).collect();
    let mut notes = Vec::new();
    for (i, b) in blocks.iter().enumerate() {
        if b.size == 0 { continue; }
        ensure!([TEXT, TOKENIZER, HINTS_JSON, HINTS_BIN].contains(&i), "el bloque v9 {i} no tiene equivalente en el migrador todavía");
        ensure!(xxhash_rust::xxh3::xxh3_64(section(&data, b.offset, b.size)?) == b.checksum, "el bloque v9 {i} no cuadra con su xxh3");
    }
    let manifest_bytes = section(&data, manifest_offset, manifest_size)?;
    let manifest: Json = serde_json::from_slice(manifest_bytes).context("manifiesto v9")?;
    let hints: Json = serde_json::from_slice(section(&data, blocks[HINTS_JSON].offset, blocks[HINTS_JSON].size)?).context("hints JSON v9")?;
    let tnih = section(&data, blocks[HINTS_BIN].offset, blocks[HINTS_BIN].size)?;

    // Configuración: el binario manda, y para los campos que repite del JSON tiene que
    // decir lo mismo; lo que solo vivía en el JSON pasa a propiedad tipada.
    ensure!(tnih.len() >= 64 && u32_at(tnih, 0) == 0x48494E54, "configuración binaria sin magic");
    let text_offset = u32_at(tnih, 8) as usize;
    let text_bin = tnih.get(text_offset..text_offset + 128).context("configuración de texto fuera del bloque")?;
    let expected = crate::hints::TextModelConfigBin::from_json(&hints["text"]).to_bytes();
    let differing: Vec<usize> = (0..128).filter(|&i| expected[i] != text_bin[i]).collect();
    ensure!(differing.is_empty(), "la configuración binaria no dice lo mismo que el JSON en los bytes {differing:?}");
    let flags = u32_at(tnih, 40);
    for (bit, key) in [(0, "text_enabled"), (1, "vision_enabled"), (2, "audio_enabled")] {
        ensure!(((flags >> bit) & 1 == 1) == hints[key].as_bool().unwrap_or(false), "{key} distinto en binario y JSON");
    }
    notes.push("configuración: los campos de texto que repite el JSON coinciden byte a byte con el binario".into());
    let softcap = hints["text"]["final_logit_softcapping"].as_f64().context("falta final_logit_softcapping")?;

    // Escritura.
    let mut builder = Builder::create(to)?;
    let copy = |w: &mut dyn std::io::Write, bytes: &[u8]| -> Result<()> {
        for chunk in bytes.chunks(COPY_CHUNK) { w.write_all(chunk)?; }
        Ok(())
    };
    let text_block = section(&data, blocks[TEXT].offset, blocks[TEXT].size)?;
    let (text_payload, _) = builder.payload(blocks[TEXT].size, 4096, |w| copy(w, text_block))?;
    let tokenizer = section(&data, blocks[TOKENIZER].offset, blocks[TOKENIZER].size)?;
    let (tokenizer_payload, tokenizer_hash) = builder.payload(tokenizer.len() as u64, 4096, |w| copy(w, tokenizer))?;
    let (config_payload, config_hash) = builder.payload(tnih.len() as u64, 4096, |w| copy(w, tnih))?;
    let template_bytes = template.map(std::fs::read).transpose().context("plantilla")?;
    let template_payload = match &template_bytes {
        Some(bytes) => Some(builder.payload(bytes.len() as u64, 4096, |w| copy(w, bytes))?),
        None => None,
    };
    let shift = text_payload - blocks[TEXT].offset;

    let mut md = Metadata::default();
    let sources: std::collections::HashMap<&str, &Json> = manifest["source_tensors"].as_array().context("source_tensors")?
        .iter().filter_map(|s| Some((s["canonical"].as_str()?, s))).collect();
    let mut weight_bytes = 0;
    for t in manifest["tensors"].as_array().context("tensors")? {
        let name = t["name"].as_str().context("nombre")?;
        ensure!(t["block"] == "text_model", "{name} no es del bloque de texto");
        let (offset, size) = (t["offset"].as_u64().context("offset")?, t["size"].as_u64().context("size")?);
        ensure!(offset >= blocks[TEXT].offset && offset + size <= blocks[TEXT].offset + blocks[TEXT].size, "{name} fuera del bloque de texto");
        let shape: Vec<u64> = t["shape"].as_array().context("shape")?.iter().map(|d| d.as_u64().context("dim")).collect::<Result<_>>()?;
        let source = sources.get(name);
        let logical_last = source.and_then(|s| s["logical_shape"].as_array()?.last()?.as_u64()).unwrap_or(*shape.last().unwrap_or(&0));
        let experts = if name.contains(".moe.experts.") { u32::try_from(shape[0])? } else { 0 };
        md.tensors.push(Tensor {
            name: name.into(), component: 1, layer: layer_of(name), dtype: t["dtype"].as_str().context("dtype")?.into(),
            shape, logical_last, offset: offset + shift, size, experts,
            hash: xxh3_128(section(&data, offset, size)?),
            source: source.and_then(|s| s["source"].as_str()).unwrap_or("").into(),
            source_hash: source.and_then(|s| u64::from_str_radix(s["source_xxh3_64"].as_str()?, 16).ok()).unwrap_or(0),
        });
        weight_bytes += size;
    }
    let count = md.tensors.len() as u32;
    let component = |id, kind, required, name: &str, offset, size, first, tensors, weight, hash| Component {
        id, kind, required, name: name.into(), payload_offset: offset, payload_size: size, alignment: 4096,
        weight_bytes: weight, memory: None, first_tensor: first, tensor_count: tensors,
        depends_on: None, composite: [0; 16], bytes_hash: hash };
    md.components.push(component(1, kind::TEXT, true, "texto", text_payload, blocks[TEXT].size, 0, count, weight_bytes, [0; 16]));
    md.components.push(component(2, kind::TOKENIZER, true, "tokenizer HTF3", tokenizer_payload, tokenizer.len() as u64, count, 0, 0, tokenizer_hash));
    md.components.push(component(3, kind::CONFIG, true, "configuración TNIH", config_payload, tnih.len() as u64, count, 0, 0, config_hash));
    if let (Some((offset, hash)), Some(bytes)) = (template_payload, &template_bytes) {
        md.components.push(component(4, kind::TEMPLATE, false, "chat_template.jinja", offset, bytes.len() as u64, count, 0, 0, hash));
    }
    md.properties.push((3, Property { key: "final_logit_softcapping".into(), value: Value::Real(softcap) }));
    for i in 0..md.components.len() { md.components[i].composite = md.composite(&md.components[i]); }

    // Procedencia: conversión v9, captura, refinado y sellado si los hubo, y esta migración.
    let mut convert = Vec::new();
    for key in ["purpose", "scope", "base_id", "experimental", "production"] {
        if let Some(p) = text_param(key, &manifest[key]) { convert.push(p); }
    }
    flat("quantization", &manifest["quantization"], &mut convert);
    if let Some(p) = text_param("source_model", &manifest["source_model"]) { convert.push(p); }
    let unknown = [0u8; 16];
    let new_step = |kind, tool: &str, version: &str, calibration: &str, touched: Vec<String>, params| Step {
        kind, state: step::COMPLETE, tool: tool.into(), version: version.into(), commit: String::new(),
        input: unknown, output: unknown, calibration: calibration.into(), touched, params };
    md.provenance.push(new_step(step::CONVERT, manifest["generator"].as_str().unwrap_or("?"),
        manifest["version"].as_str().unwrap_or("?"), "", vec![], convert));
    if let Some(r) = manifest.get("refinement") {
        if let Some(imp) = r.get("importance") {
            let mut params = Vec::new();
            flat("", imp, &mut params);
            md.provenance.push(new_step(step::CAPTURE_IMPORTANCE, "hnf_refine --importance", "", imp["calibration"]["ids_fnv1a64"].as_str().unwrap_or(""), vec![], params));
        }
        let mut params = Vec::new();
        for key in ["damp", "grid", "arithmetic", "note"] { if let Some(p) = text_param(key, &r[key]) { params.push(p); } }
        if let Some(layers) = r["layers"].as_array() {
            params.push(Property { key: "layers".into(), value: Value::Str(format!("{}..{}", layers[0], layers[1])) });
        }
        let touched = r["refined_tensors"].as_array().map(|a| a.iter().filter_map(|t| t["name"].as_str().map(String::from)).collect()).unwrap_or_default();
        md.provenance.push(new_step(step::REFINE, "hnf_refine", "", r["calibration"]["ids_fnv1a64"].as_str().unwrap_or(""), touched, params));
        let mut seal = Vec::new();
        for key in ["text_block_xxh3_64", "text_block_xxh3_64_before"] { if let Some(p) = text_param(key, &r[key]) { seal.push(p); } }
        md.provenance.push(new_step(step::SEAL, "hnf-seal", "", "", vec![], seal));
    }
    // Identidad del v9 de partida: su manifiesto y los xxh3 de sus bloques.
    let mut v9 = manifest_bytes.to_vec();
    for b in &blocks { v9.extend_from_slice(&b.checksum.to_le_bytes()); }
    let identity = md.identity();
    let mut migrate = new_step(step::MIGRATE, "hnf10 migrate", env!("CARGO_PKG_VERSION"), "", vec![],
        vec![Property { key: "from".into(), value: Value::Str(from.display().to_string()) }]);
    migrate.commit = commit.into();
    migrate.input = xxh3_128(&v9);
    migrate.output = identity;
    md.provenance.push(migrate);
    builder.finish(md)?;

    // Verificación: reabrir, hashes de todos los tensores, y bytes comparados de verdad.
    let opened = open(to)?;
    ensure!(opened.header.identity == identity, "identidad distinta al reabrir");
    let bad = verify_tensors(to, &opened.metadata)?;
    ensure!(bad.is_empty(), "tensores que no cuadran tras migrar: {bad:?}");
    let out = unsafe { Mmap::map(&File::open(to)?)? };
    let pairs = [(text_payload, text_block), (tokenizer_payload, tokenizer), (config_payload, tnih)];
    for (offset, original) in pairs {
        if section(&out, offset, original.len() as u64)? != original { bail!("payload distinto del v9 en {offset}"); }
    }
    notes.push(format!("{} tensores, tokenizer y configuración idénticos byte a byte al v9", count));
    Ok(Report { tensors: count as usize, text_bytes: blocks[TEXT].size, identity, notes })
}

/// Propiedades planas de un registro JSON (para `hnf10 seal`).
pub fn flat_params(json: &Json, out: &mut Vec<Property>) { flat("", json, out) }
