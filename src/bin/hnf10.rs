//! hnf10: herramientas del contenedor HNFv10.
use anyhow::Result;
use clap::{Parser, Subcommand};
use helios_convert_v10::{file::{open, verify_tensors}, metadata::Value, migrate::migrate};
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
    /// Componentes, tensores y procedencia.
    Info { file: PathBuf },
    /// Verificación profunda: el hash de cada tensor y de cada payload.
    Verify { file: PathBuf },
}

fn hex(h: &[u8; 16]) -> String { h.iter().map(|b| format!("{b:02x}")).collect() }

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Migrate { from, to, template, commit } => {
            let report = migrate(&from, &to, template.as_deref(), &commit)?;
            for note in &report.notes { println!("{note}"); }
            println!("listo: {} ({} tensores, {} bytes de texto), identidad {}", to.display(), report.tensors, report.text_bytes, hex(&report.identity));
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
        Command::Verify { file } => {
            let o = open(&file)?;
            let bad = verify_tensors(&file, &o.metadata)?;
            if bad.is_empty() { println!("verificado: {} tensores y payloads correctos", o.metadata.tensors.len()); }
            else { anyhow::bail!("no cuadran: {bad:?}"); }
        }
    }
    Ok(())
}
