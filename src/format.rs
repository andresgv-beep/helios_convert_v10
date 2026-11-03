//! Cabeceras A/B de HNFv10 (docs/HNFV10.md §2).
use anyhow::{bail, ensure, Result};

pub const MAGIC: &[u8; 8] = b"HNFv10\0\0";
pub const MAJOR: u16 = 10;
pub const MINOR: u16 = 0;
pub const HEADER_SIZE: usize = 512;
/// Primer byte de payload: las dos cabeceras y relleno hasta 4 KiB.
pub const PAYLOAD_START: u64 = 4096;
pub const ALIGN: u64 = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State { Complete = 1, InProgress = 2 }

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Header {
    pub sequence: u64,
    pub state: State,
    pub metadata_offset: u64,
    pub metadata_size: u64,
    pub metadata_hash: [u8; 16],
    pub file_size: u64,
    pub identity: [u8; 16],
}

pub fn align_up(value: u64, align: u64) -> u64 { value.div_ceil(align) * align }

pub fn xxh3_128(bytes: &[u8]) -> [u8; 16] { xxhash_rust::xxh3::xxh3_128(bytes).to_le_bytes() }

impl Header {
    pub fn to_bytes(&self) -> [u8; HEADER_SIZE] {
        let mut b = [0u8; HEADER_SIZE];
        b[0..8].copy_from_slice(MAGIC);
        b[8..10].copy_from_slice(&MAJOR.to_le_bytes());
        b[10..12].copy_from_slice(&MINOR.to_le_bytes());
        b[12..16].copy_from_slice(&(HEADER_SIZE as u32).to_le_bytes());
        b[16..24].copy_from_slice(&self.sequence.to_le_bytes());
        b[24..28].copy_from_slice(&(self.state as u32).to_le_bytes());
        b[32..40].copy_from_slice(&self.metadata_offset.to_le_bytes());
        b[40..48].copy_from_slice(&self.metadata_size.to_le_bytes());
        b[48..64].copy_from_slice(&self.metadata_hash);
        b[64..72].copy_from_slice(&self.file_size.to_le_bytes());
        b[72..88].copy_from_slice(&self.identity);
        let crc = crc32fast::hash(&b[..508]);
        b[508..512].copy_from_slice(&crc.to_le_bytes());
        b
    }

    /// Solo la forma: magic, versión, CRC y campos coherentes. Los metadatos se
    /// comprueban aparte, porque hacen falta los bytes del fichero.
    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        ensure!(b.len() == HEADER_SIZE, "cabecera de tamaño incorrecto");
        ensure!(&b[0..8] == MAGIC, "no es una cabecera HNFv10");
        let u16_at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        ensure!(u16_at(8) == MAJOR, "versión mayor {} no soportada", u16_at(8));
        ensure!(u32_at(12) as usize == HEADER_SIZE, "tamaño de cabecera declarado incorrecto");
        ensure!(u32_at(508) == crc32fast::hash(&b[..508]), "CRC de cabecera incorrecto");
        let state = match u32_at(24) { 1 => State::Complete, 2 => State::InProgress, s => bail!("estado {s} desconocido") };
        let header = Header {
            sequence: u64_at(16), state,
            metadata_offset: u64_at(32), metadata_size: u64_at(40),
            metadata_hash: b[48..64].try_into().unwrap(),
            file_size: u64_at(64), identity: b[72..88].try_into().unwrap(),
        };
        ensure!(header.metadata_offset >= PAYLOAD_START && header.metadata_offset % ALIGN == 0, "metadatos fuera de sitio");
        ensure!(header.metadata_offset.checked_add(header.metadata_size).map_or(false, |end| end <= header.file_size),
            "los metadatos pasan del tamaño declarado");
        Ok(header)
    }
}

/// Cabecera que manda en un fichero, según §2: la válida con mayor secuencia.
/// `metadata` devuelve los bytes de los metadatos que apunta una cabecera, o None si
/// el fichero es más corto.
pub fn choose<'a>(slots: [&'a [u8]; 2], file_len: u64, metadata: impl Fn(&Header) -> Option<Vec<u8>>)
    -> Result<(usize, Header, Vec<u8>)> {
    let mut best: Option<(usize, Header, Vec<u8>)> = None;
    for (index, slot) in slots.iter().enumerate() {
        let Ok(header) = Header::from_bytes(slot) else { continue };
        if header.file_size > file_len { continue; }
        let Some(bytes) = metadata(&header) else { continue };
        if xxh3_128(&bytes) != header.metadata_hash { continue; }
        if best.as_ref().map_or(true, |(_, b, _)| header.sequence > b.sequence) { best = Some((index, header, bytes)); }
    }
    let Some(chosen) = best else { bail!("ninguna cabecera válida: el fichero no es HNFv10 o está dañado") };
    ensure!(chosen.1.state == State::Complete,
        "la última escritura de pesos quedó en curso: el fichero no es fiable hasta repararlo (hnf10 repair)");
    Ok(chosen)
}
