//! Crear, abrir y confirmar cambios en un HNFv10 (docs/HNFV10.md §2).
use crate::format::{align_up, choose, xxh3_128, Header, State, ALIGN, HEADER_SIZE, PAYLOAD_START};
use crate::metadata::Metadata;
use anyhow::{ensure, Context, Result};
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::Path;

/// Un HNFv10 abierto: la cabecera que manda, su ranura (0 = A, 1 = B) y sus metadatos.
pub struct Opened {
    pub slot: usize,
    pub header: Header,
    pub metadata: Metadata,
}

fn read_slots(file: &File) -> Result<[Vec<u8>; 2]> {
    let mut a = vec![0u8; HEADER_SIZE];
    let mut b = vec![0u8; HEADER_SIZE];
    file.read_exact_at(&mut a, 0).context("fichero más corto que las cabeceras")?;
    file.read_exact_at(&mut b, HEADER_SIZE as u64).context("fichero más corto que las cabeceras")?;
    Ok([a, b])
}

fn metadata_bytes(file: &File, len: u64, h: &Header) -> Option<Vec<u8>> {
    if h.metadata_offset.checked_add(h.metadata_size)? > len { return None; }
    let mut buf = vec![0u8; usize::try_from(h.metadata_size).ok()?];
    file.read_exact_at(&mut buf, h.metadata_offset).ok()?;
    Some(buf)
}

/// Abre y comprueba todo lo barato: cabeceras, metadatos, estructura y compuestos.
/// No lee pesos; para eso está `verify_tensors`.
pub fn open(path: &Path) -> Result<Opened> {
    let file = File::open(path).with_context(|| format!("no abro {}", path.display()))?;
    let len = file.metadata()?.len();
    let slots = read_slots(&file)?;
    let (slot, header, bytes) = choose([&slots[0], &slots[1]], len, |h| metadata_bytes(&file, len, h))?;
    let metadata = Metadata::decode(&bytes)?;
    metadata.validate(header.metadata_offset)?;
    ensure!(metadata.identity() == header.identity, "la identidad de la cabecera no corresponde a sus metadatos");
    Ok(Opened { slot, header, metadata })
}

/// Recalcula el hash de bytes de cada tensor (o de cada payload sin tensores) y lo
/// compara con el índice. Devuelve los nombres que no cuadran.
pub fn verify_tensors(path: &Path, metadata: &Metadata) -> Result<Vec<String>> {
    let file = File::open(path)?;
    let mut bad = Vec::new();
    for t in &metadata.tensors {
        let mut buf = vec![0u8; usize::try_from(t.size)?];
        file.read_exact_at(&mut buf, t.offset)?;
        if xxh3_128(&buf) != t.hash { bad.push(t.name.clone()); }
    }
    for c in metadata.components.iter().filter(|c| c.tensor_count == 0 && c.payload_size > 0) {
        let mut buf = vec![0u8; usize::try_from(c.payload_size)?];
        file.read_exact_at(&mut buf, c.payload_offset)?;
        if xxh3_128(&buf) != c.bytes_hash { bad.push(c.name.clone()); }
    }
    Ok(bad)
}

/// Escritor de un HNFv10 nuevo: payloads en orden y, al final, metadatos y cabecera A.
pub struct Builder {
    file: File,
    end: u64,
}

impl Builder {
    pub fn create(path: &Path) -> Result<Self> {
        let mut file = OpenOptions::new().write(true).create_new(true).read(true).open(path)
            .with_context(|| format!("no creo {} (¿ya existe?)", path.display()))?;
        file.write_all(&vec![0u8; PAYLOAD_START as usize])?;
        Ok(Builder { file, end: PAYLOAD_START })
    }

    /// Reserva sitio alineado para `size` bytes y devuelve su offset; `fill` los escribe
    /// en orden. Devuelve también el xxh3-128 de lo escrito.
    pub fn payload(&mut self, size: u64, alignment: u64, mut fill: impl FnMut(&mut dyn Write) -> Result<()>) -> Result<(u64, [u8; 16])> {
        let alignment = alignment.max(ALIGN);
        let offset = align_up(self.end, alignment);
        self.file.seek(SeekFrom::Start(self.end))?;
        self.file.write_all(&vec![0u8; (offset - self.end) as usize])?;
        let mut hashing = Hashing { inner: &mut self.file, state: xxhash_rust::xxh3::Xxh3::new(), written: 0 };
        fill(&mut hashing)?;
        ensure!(hashing.written == size, "payload de {} bytes, se esperaban {size}", hashing.written);
        let hash = hashing.state.digest128().to_le_bytes();
        self.end = offset + size;
        Ok((offset, hash))
    }

    /// Escribe metadatos y la cabecera A (secuencia 1, completo). La B queda vacía.
    pub fn finish(mut self, mut metadata: Metadata) -> Result<Header> {
        for i in 0..metadata.components.len() {
            let composite = metadata.composite(&metadata.components[i]);
            metadata.components[i].composite = composite;
        }
        let offset = align_up(self.end, ALIGN);
        metadata.validate(offset)?;
        let bytes = metadata.encode();
        self.file.seek(SeekFrom::Start(self.end))?;
        self.file.write_all(&vec![0u8; (offset - self.end) as usize])?;
        self.file.write_all(&bytes)?;
        self.file.sync_all()?;
        let header = Header { sequence: 1, state: State::Complete, metadata_offset: offset, metadata_size: bytes.len() as u64,
            metadata_hash: xxh3_128(&bytes), file_size: offset + bytes.len() as u64, identity: metadata.identity() };
        self.file.write_all_at(&header.to_bytes(), 0)?;
        self.file.sync_all()?;
        Ok(header)
    }
}

struct Hashing<'a> { inner: &'a mut File, state: xxhash_rust::xxh3::Xxh3, written: u64 }
impl Write for Hashing<'_> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?; self.state.update(&buf[..n]); self.written += n as u64; Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> { self.inner.flush() }
}

/// Todas las cabeceras válidas (CRC, metadatos completos y con su hash), la más reciente
/// primero, incluidas las que dicen «en curso».
fn valid_headers(file: &File) -> Result<Vec<Opened>> {
    let len = file.metadata()?.len();
    let slots = read_slots(file)?;
    let mut valid = Vec::new();
    for (slot, bytes) in slots.iter().enumerate() {
        let Ok(header) = Header::from_bytes(bytes) else { continue };
        if header.file_size > len { continue; }
        let Some(meta) = metadata_bytes(file, len, &header) else { continue };
        if xxh3_128(&meta) != header.metadata_hash { continue; }
        let Ok(metadata) = Metadata::decode(&meta) else { continue };
        valid.push(Opened { slot, header, metadata });
    }
    valid.sort_by_key(|o| std::cmp::Reverse(o.header.sequence));
    Ok(valid)
}

/// Lee un HNFv10 aunque su última cabecera diga «en curso»: solo para reparar.
pub fn open_for_repair(path: &Path) -> Result<(Opened, Option<Opened>)> {
    let file = File::open(path)?;
    let mut it = valid_headers(&file)?.into_iter();
    let newest = it.next().context("ninguna cabecera válida")?;
    Ok((newest, it.next()))
}

/// Sesión de escritura sobre un HNFv10: cerrojo exclusivo del fichero mientras dure (una
/// modificación de pesos es «en curso», pesos y cierre, todo en la misma sesión) y cada
/// confirmación comprueba en disco que la cabecera vigente es la que la sesión espera.
/// Así una confirmación hecha con un estado antiguo no puede tapar un «en curso».
pub struct Session {
    file: File,
    current: Opened,
}

impl Session {
    /// Para sellar o refinar: el fichero tiene que estar completo.
    pub fn begin(path: &Path) -> Result<Self> {
        let session = Self::lock(path)?;
        ensure!(session.current.header.state == State::Complete,
            "la última escritura de pesos quedó en curso: el fichero no es fiable hasta repararlo (hnf10 repair)");
        Ok(session)
    }

    /// Para reparar: admite que la cabecera vigente diga «en curso».
    pub fn begin_repair(path: &Path) -> Result<Self> { Self::lock(path) }

    fn lock(path: &Path) -> Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path).with_context(|| format!("no abro {}", path.display()))?;
        file.try_lock().map_err(|_| anyhow::anyhow!("otro proceso está escribiendo en {}", path.display()))?;
        let current = valid_headers(&file)?.into_iter().next().context("ninguna cabecera válida")?;
        Ok(Session { file, current })
    }

    pub fn current(&self) -> &Opened { &self.current }

    /// El fichero, para escribir pesos entre la confirmación «en curso» y la de cierre.
    pub fn file(&self) -> &File { &self.file }

    /// Escribe los metadatos al final, `fsync`, y después la cabecera que no manda, con
    /// secuencia + 1 y el estado dado, `fsync` (§2).
    pub fn commit(&mut self, mut metadata: Metadata, state: State) -> Result<&Opened> {
        let on_disk = valid_headers(&self.file)?.into_iter().next().context("ninguna cabecera válida")?;
        ensure!(on_disk.header.sequence == self.current.header.sequence && on_disk.slot == self.current.slot,
            "la cabecera vigente ya no es la de esta sesión (secuencia {} en disco, {} esperada): confirmación rechazada",
            on_disk.header.sequence, self.current.header.sequence);
        for i in 0..metadata.components.len() {
            let composite = metadata.composite(&metadata.components[i]);
            metadata.components[i].composite = composite;
        }
        let len = self.file.metadata()?.len();
        let offset = align_up(len.max(self.current.header.file_size), ALIGN);
        metadata.validate(offset)?;
        let bytes = metadata.encode();
        let mut padded = vec![0u8; (offset - len) as usize];
        padded.extend_from_slice(&bytes);
        self.file.write_all_at(&padded, len)?;
        self.file.sync_all()?;
        let header = Header { sequence: self.current.header.sequence + 1, state, metadata_offset: offset,
            metadata_size: bytes.len() as u64, metadata_hash: xxh3_128(&bytes), file_size: offset + bytes.len() as u64,
            identity: metadata.identity() };
        let target = 1 - self.current.slot;
        self.file.write_all_at(&header.to_bytes(), (target * HEADER_SIZE) as u64)?;
        self.file.sync_all()?;
        self.current = Opened { slot: target, header, metadata };
        Ok(&self.current)
    }
}
