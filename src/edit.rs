//! Modificar pesos de un HNFv10 sin que nadie cargue el fichero a medias (docs/HNFV10.md §2, §7).
//!
//! `begin_edit` copia el HNF y deja la copia «en curso», con un paso que dice qué tensores se van
//! a tocar. Una herramienta externa (`hnf_refine` en Héctor) escribe esos pesos. `seal` recalcula
//! sus hashes, comprueba que los demás tensores no cambiaron y cierra el paso. Si se cortó a
//! medias, `repair` cancela el paso cuando los pesos siguen intactos.
use crate::file::{open, Session};
use crate::format::{xxh3_128, State};
use crate::metadata::{step, Metadata, Property, Step, Value};
use anyhow::{bail, ensure, Context, Result};
use std::os::unix::fs::FileExt;
use std::path::Path;

fn tensor_hash(session: &Session, offset: u64, size: u64) -> Result<[u8; 16]> {
    let mut buf = vec![0u8; usize::try_from(size)?];
    session.file().read_exact_at(&mut buf, offset)?;
    Ok(xxh3_128(&buf))
}

/// Copia `from` en `to` y deja la copia «en curso» para que `tool` modifique los tensores cuyo
/// nombre contiene `pattern`. Devuelve cuántos tensores quedan marcados.
pub fn begin_edit(from: &Path, to: &Path, tool: &str, pattern: &str, kind: u32) -> Result<usize> {
    ensure!(!to.exists(), "{} ya existe", to.display());
    let source = open(from)?;
    let part = to.with_extension("part");
    std::fs::copy(from, &part).context("copiar el HNF")?;
    std::fs::rename(&part, to)?;
    let mut session = Session::begin(to)?;
    let mut md = session.current().metadata.clone();
    let touched: Vec<String> = md.tensors.iter().filter(|t| t.name.contains(pattern)).map(|t| t.name.clone()).collect();
    ensure!(!touched.is_empty(), "ningún tensor contiene «{pattern}»");
    md.provenance.push(Step { kind, state: step::IN_PROGRESS, tool: tool.into(), version: String::new(), commit: String::new(),
        input: source.header.identity, output: [0; 16], calibration: String::new(), touched: touched.clone(),
        params: vec![Property { key: "pattern".into(), value: Value::Str(pattern.into()) }] });
    session.commit(md, State::InProgress)?;
    Ok(touched.len())
}

/// Tokenizer, configuración, plantilla…: una edición de pesos no los toca, así que tienen que
/// seguir con su hash antes de cerrar.
fn check_resources(session: &Session, md: &Metadata) -> Result<()> {
    for c in md.components.iter().filter(|c| c.tensor_count == 0 && c.payload_size > 0) {
        ensure!(tensor_hash(session, c.payload_offset, c.payload_size)? == c.bytes_hash,
            "el recurso {} cambió durante la edición", c.name);
    }
    Ok(())
}

/// El paso «en curso» de un fichero, o error si no lo hay.
fn pending(md: &Metadata) -> Result<usize> {
    let i = md.provenance.len().checked_sub(1).context("sin procedencia")?;
    ensure!(md.provenance[i].state == step::IN_PROGRESS, "el último paso no está en curso");
    Ok(i)
}

/// Cierra la edición: hashes nuevos de los tensores tocados, los demás comprobados contra los
/// suyos, parámetros de la herramienta (`params`) y estado completo. Devuelve los que cambiaron.
pub fn seal(path: &Path, params: Vec<Property>, calibration: &str) -> Result<usize> {
    let mut session = Session::begin_repair(path)?;
    ensure!(session.current().header.state == State::InProgress, "{} no tiene una edición en curso", path.display());
    let mut md = session.current().metadata.clone();
    let i = pending(&md)?;
    check_resources(&session, &md)?;
    let touched: std::collections::HashSet<String> = md.provenance[i].touched.iter().cloned().collect();
    let mut changed = 0;
    for t in 0..md.tensors.len() {
        let (offset, size, name) = (md.tensors[t].offset, md.tensors[t].size, md.tensors[t].name.clone());
        let hash = tensor_hash(&session, offset, size)?;
        if touched.contains(&name) {
            if hash != md.tensors[t].hash { changed += 1; }
            md.tensors[t].hash = hash;
        } else {
            ensure!(hash == md.tensors[t].hash, "{name} cambió sin estar entre los tensores de la edición");
        }
    }
    for c in 0..md.components.len() { md.components[c].composite = md.composite(&md.components[c]); }
    let identity = md.identity();
    let s = &mut md.provenance[i];
    s.state = step::COMPLETE;
    s.output = identity;
    s.calibration = calibration.into();
    s.params.extend(params);
    s.params.push(Property { key: "changed_tensors".into(), value: Value::Int(changed as i64) });
    session.commit(md, State::Complete)?;
    Ok(changed)
}

/// Reparar una edición cortada: si los tensores tocados siguen con sus hashes de antes, el paso
/// se cierra como cancelado. Si cambiaron, no hay estado al que volver: hay que terminar la
/// edición (volver a lanzar la herramienta y sellar) o regenerar el fichero.
pub fn repair(path: &Path) -> Result<()> {
    let mut session = Session::begin_repair(path)?;
    ensure!(session.current().header.state == State::InProgress, "{} no necesita reparación", path.display());
    let mut md = session.current().metadata.clone();
    let i = pending(&md)?;
    check_resources(&session, &md)?;
    for t in &md.tensors {
        if tensor_hash(&session, t.offset, t.size)? != t.hash {
            bail!("{} ya cambió: termina la edición (vuelve a lanzar la herramienta y sella) o regenera el fichero", t.name);
        }
    }
    md.provenance[i].state = step::CANCELLED;
    session.commit(md, State::Complete)?;
    Ok(())
}
