//! hnf10: herramientas del contenedor HNFv10.
use anyhow::Result;
use clap::{Parser, Subcommand};
use helios_convert_v10::{edit, file::{open, verify_tensors}, fixtures, metadata::{step, Value}, migrate::migrate};
use std::path::PathBuf;

#[derive(Parser)]
#[command(about = "Herramientas de HNFv10")]
struct Cli { #[command(subcommand)] command: Command }

#[derive(Subcommand)]
enum Command {
    /// Pasa un HNFv9 a HNFv10 sin tocar los pesos y lo verifica byte a byte.
    Migrate {
        #[arg(long)] from: PathBuf,
        #[arg(long)] to: PathBuf,
        /// chat_template.jinja del checkpoint, guardado tal cual como componente opcional.
        #[arg(long)] template: Option<PathBuf>,
        /// Commit de helios_convert_v10 con el que se migra, para la procedencia.
        #[arg(long, default_value = "")] commit: String,
    },
    /// Añade la torre visual del checkpoint a un HNFv10 de texto sin tocar sus pesos.
    AddVision {
        #[arg(long)] from: PathBuf,
        /// Carpeta del checkpoint original (config.json, processor_config.json, safetensors).
        #[arg(long)] model: PathBuf,
        #[arg(long)] to: PathBuf,
        #[arg(long, default_value = "")] commit: String,
    },
    /// Componentes, tensores y procedencia.
    Info { file: PathBuf },
    /// Verificación profunda: el hash de cada tensor y de cada payload.
    Verify { file: PathBuf },
    /// Convierte un checkpoint Gemma 4 de safetensors a HNFv10: el MoE (26B) o uno denso (E2B,
    /// E4B), según su config.json.
    Convert {
        #[arg(long)] model: PathBuf,
        #[arg(long)] out: PathBuf,
        /// Denso: añade la torre visual en el mismo fichero, como componente opcional.
        #[arg(long)] vision: bool,
        /// Denso: MLP y tabla PLE en HQ44 (unos 5 GB el E4B); si no, todo en HQ8K («q8»).
        #[arg(long)] compacto: bool,
        /// Denso: formato de la tabla PLE aparte del perfil: hq44, hq52 o hq8k.
        #[arg(long)] ple: Option<String>,
        /// Atención, MLP compartida y embedding en HQ8K (perfil compacto); si no, HQ44.
        #[arg(long)] dense_hq8: bool,
        /// Importancia HQ44: por experto en el MoE (layerN.gate_up.f32 y layerN.down.f32), y por
        /// columna de cada mlp.down en el denso (text.layerN.mlp.down.weight.f32).
        #[arg(long)] importance: Option<PathBuf>,
        #[arg(long, default_value = "")] commit: String,
        #[arg(long, default_value_t = 10)] threads: usize,
    },
    /// Copia un HNFv10 y deja la copia «en curso» para que una herramienta modifique pesos.
    BeginEdit {
        #[arg(long)] from: PathBuf,
        #[arg(long)] to: PathBuf,
        /// Herramienta que hará la edición, para la procedencia.
        #[arg(long, default_value = "hnf_refine")] tool: String,
        /// Se tocan los tensores cuyo nombre contiene este texto.
        #[arg(long, default_value = ".moe.experts.")] touch: String,
    },
    /// Cierra una edición: hashes nuevos, comprobación del resto y parámetros del registro.
    Seal {
        file: PathBuf,
        /// record.json de la herramienta (hnf_refine lo deja en su carpeta de trabajo).
        #[arg(long)] record: Option<PathBuf>,
    },
    /// Compara dos HNFv10 tensor a tensor: iguales, misma rejilla HQ44 con otros códigos, o distintos.
    Compare { a: PathBuf, b: PathBuf },
    /// Cancela una edición cortada si los pesos siguen intactos.
    Repair { file: PathBuf },
    /// HNFv10 de prueba para el lector del motor (Hector-engine/tests/data/hnf10).
    #[command(hide = true)]
    Fixtures { dir: PathBuf },
}

fn rayon_threads(n: usize) -> Result<()> {
    rayon::ThreadPoolBuilder::new().num_threads(n).build_global()?;
    Ok(())
}

fn hex(h: &[u8; 16]) -> String { h.iter().map(|b| format!("{b:02x}")).collect() }

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Migrate { from, to, template, commit } => {
            let report = migrate(&from, &to, template.as_deref(), &commit)?;
            for note in &report.notes { println!("{note}"); }
            println!("listo: {} ({} tensores, {} bytes de texto), identidad {}", to.display(), report.tensors, report.text_bytes, hex(&report.identity));
        }
        Command::AddVision { from, model, to, commit } => {
            let r = helios_convert_v10::vision::add_vision(&from, &model, &to, &commit)?;
            println!("listo: {} ({} tensores de visión, {} bytes; {} valores no exactos en FP16; memoria de trabajo {} bytes con el máximo de parches), identidad {}",
                to.display(), r.tensors, r.weight_bytes, r.inexact, r.scratch_bytes_at_max, hex(&r.identity));
        }
        Command::Info { file } => {
            let o = open(&file)?;
            println!("HNFv10, cabecera {} secuencia {}, identidad {}", if o.slot == 0 { "A" } else { "B" }, o.header.sequence, hex(&o.header.identity));
            for c in &o.metadata.components {
                println!("  componente {} «{}» tipo {} {}: {} bytes en {}, {} tensores",
                    c.id, c.name, c.kind, if c.required { "obligatorio" } else { "opcional" }, c.payload_size, c.payload_offset, c.tensor_count);
            }
            for (c, p) in &o.metadata.properties {
                let v = match &p.value { Value::Real(x) => x.to_string(), Value::Int(x) => x.to_string(), Value::Bool(x) => x.to_string(),
                    Value::Str(x) => x.clone(), Value::Bytes(x) => format!("{} bytes", x.len()) };
                println!("  propiedad de {c}: {} = {v}", p.key);
            }
            for s in &o.metadata.provenance {
                println!("  paso {} «{}» {} {} params, {} tensores tocados", s.kind, s.tool, s.version, s.params.len(), s.touched.len());
            }
        }
        Command::Convert { model, out, vision, compacto, ple, dense_hq8, importance, commit, threads } => {
            rayon_threads(threads)?;
            let config: serde_json::Value = serde_json::from_slice(&std::fs::read(model.join("config.json"))?)?;
            if config["text_config"]["enable_moe_block"] != true {
                anyhow::ensure!(!dense_hq8, "--dense-hq8 es del MoE; el denso ya va en HQ8K");
                let tabla_ple = match ple.as_deref() {
                    None | Some("hq44") => helios_convert_v10::gemma4_dense::Ple::DelPerfil,
                    Some("hq52") => helios_convert_v10::gemma4_dense::Ple::Hq52,
                    Some("hq8k") => helios_convert_v10::gemma4_dense::Ple::Hq8,
                    Some(otro) => anyhow::bail!("--ple {otro}: solo hq44, hq52 o hq8k"),
                };
                let identity = helios_convert_v10::gemma4_dense::convert(&helios_convert_v10::gemma4_dense::Options {
                    model: &model, out: &out, vision, commit: &commit, ple: tabla_ple,
                    importancia: importance.as_deref(),
                    profile: if compacto { helios_convert_v10::gemma4_dense::Profile::Compacto } else { helios_convert_v10::gemma4_dense::Profile::Q8 } })?;
                println!("listo: {}, identidad {}", out.display(), hex(&identity));
                return Ok(());
            }
            anyhow::ensure!(!vision && !compacto && ple.is_none(), "en el MoE la visión se añade con add-vision y el perfil va con --dense-hq8");
            let dense = if dense_hq8 { helios_convert_v10::moe_experimental::DenseFormat::Hq8 } else { helios_convert_v10::moe_experimental::DenseFormat::Hq44 };
            let identity = helios_convert_v10::convert::convert(&helios_convert_v10::convert::Options {
                model: &model, out: &out, dense, importance: importance.as_deref(), commit: &commit, batch_rows: 128 })?;
            println!("listo: {}, identidad {}", out.display(), hex(&identity));
        }
        Command::BeginEdit { from, to, tool, touch } => {
            let n = edit::begin_edit(&from, &to, &tool, &touch, step::REFINE)?;
            println!("{}: {n} tensores en edición por {tool}", to.display());
        }
        Command::Seal { file, record } => {
            let (mut params, mut calibration) = (Vec::new(), String::new());
            if let Some(path) = record {
                let json: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
                anyhow::ensure!(json["complete"] == true, "el registro dice que la herramienta no terminó");
                helios_convert_v10::migrate::flat_params(&json, &mut params);
                calibration = json["calibration"]["ids_fnv1a64"].as_str().unwrap_or("").to_string();
            }
            let changed = edit::seal(&file, params, &calibration)?;
            println!("sellado: {changed} tensores cambiaron");
        }
        Command::Compare { a, b } => {
            use std::os::unix::fs::FileExt;
            let (oa, ob) = (open(&a)?, open(&b)?);
            // Los hashes guardados solo valen si los bytes los cumplen.
            for (path, opened) in [(&a, &oa), (&b, &ob)] {
                let bad = verify_tensors(path, &opened.metadata)?;
                anyhow::ensure!(bad.is_empty(), "{}: bytes que no cumplen su hash: {bad:?}", path.display());
            }
            let (fa, fb) = (std::fs::File::open(&a)?, std::fs::File::open(&b)?);
            let index: std::collections::HashMap<&str, &helios_convert_v10::metadata::Tensor> =
                ob.metadata.tensors.iter().map(|t| (t.name.as_str(), t)).collect();
            let (mut same, mut grid, mut other) = (0, 0, Vec::new());
            for t in &oa.metadata.tensors {
                let Some(u) = index.get(t.name.as_str()) else { other.push(format!("{} falta en B", t.name)); continue };
                let kind_of = |o: &helios_convert_v10::file::Opened, id: u32| o.metadata.components.iter().find(|c| c.id == id).map(|c| c.kind);
                if t.dtype != u.dtype || t.shape != u.shape || t.logical_last != u.logical_last || t.experts != u.experts
                    || t.layer != u.layer || t.size != u.size || kind_of(&oa, t.component) != kind_of(&ob, u.component) {
                    other.push(format!("{}: descriptores distintos", t.name));
                    continue;
                }
                if t.hash == u.hash { same += 1; continue; }
                if t.dtype == "hq44k_g16" {
                    let (mut x, mut y) = (vec![0u8; t.size as usize], vec![0u8; u.size as usize]);
                    fa.read_exact_at(&mut x, t.offset)?; fb.read_exact_at(&mut y, u.offset)?;
                    if x.chunks_exact(152).zip(y.chunks_exact(152)).all(|(p, q)| p[..24] == q[..24]) { grid += 1; continue; }
                }
                other.push(t.name.clone());
            }
            let names_a: std::collections::HashSet<&str> = oa.metadata.tensors.iter().map(|t| t.name.as_str()).collect();
            for u in &ob.metadata.tensors { if !names_a.contains(u.name.as_str()) { other.push(format!("{} sobra en B", u.name)); } }
            println!("{same} tensores iguales, {grid} HQ44 con la misma rejilla y otros códigos, {} distintos", other.len());
            for name in other.iter().take(20) { println!("  distinto: {name}"); }
            for c in oa.metadata.components.iter().filter(|c| c.tensor_count == 0) {
                let twin = ob.metadata.components.iter().find(|d| d.kind == c.kind);
                println!("  componente {}: {}", c.name, match twin { Some(d) if d.bytes_hash == c.bytes_hash => "igual", Some(_) => "DISTINTO", None => "falta en B" });
            }
            for d in ob.metadata.components.iter().filter(|d| !oa.metadata.components.iter().any(|c| c.kind == d.kind)) {
                println!("  componente {}: sobra en B", d.name);
            }
        }
        Command::Repair { file } => {
            edit::repair(&file)?;
            println!("reparado: la edición se cerró como cancelada");
        }
        Command::Fixtures { dir } => {
            fixtures::write(&dir)?;
            println!("fixtures en {}", dir.display());
        }
        Command::Verify { file } => {
            let o = open(&file)?;
            let bad = verify_tensors(&file, &o.metadata)?;
            if bad.is_empty() { println!("verificado: {} tensores y payloads correctos", o.metadata.tensors.len()); }
            else { anyhow::bail!("no cuadran: {bad:?}"); }
        }
    }
    Ok(())
}
