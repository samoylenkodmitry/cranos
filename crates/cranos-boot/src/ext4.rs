use core::ffi::c_void;

use crate::uefi::{
    ALLOCATE_ANY_PAGES, BlockIoProtocol, BootServices, EFI_LOADER_DATA, EFI_SUCCESS, SystemTable,
    print_ascii, print_u64,
};

const EXT4_SUPER_MAGIC: u16 = 0xEF53;
const EXT4_FEATURE_INCOMPAT_64BIT: u32 = 0x0080;
const EXT4_EXTENTS_FL: u32 = 0x0008_0000;
const EXT4_EXT_MAGIC: u16 = 0xF30A;
const ROOT_INO: u32 = 2;

struct BlockReader {
    bio: *mut BlockIoProtocol,
    media_id: u32,
    sector_size: usize,
    io_align: usize,
    scratch: *mut u8,
    scratch_cap: usize,
}

impl BlockReader {
    fn read_bytes(&self, byte_offset: u64, dst: &mut [u8]) -> Result<(), &'static str> {
        let ss = self.sector_size as u64;
        if ss == 0 || !ss.is_power_of_two() {
            return Err("invalid sector size");
        }
        let mut remaining = dst.len();
        let mut cur_off = byte_offset;
        let mut dst_pos = 0usize;

        while remaining > 0 {
            let lba = cur_off / ss;
            let in_sector = (cur_off % ss) as usize;
            let dst_ptr = unsafe { dst.as_mut_ptr().add(dst_pos) };
            let aligned_ok = self.io_align <= 1 || ((dst_ptr as usize) % self.io_align) == 0;

            // Fast path: aligned direct DMA into caller buffer for large reads.
            // UEFI NVMe BlockIo returns EFI_INVALID_PARAMETER if Buffer is not
            // aligned to Media->IoAlign, so unaligned stack buffers route via scratch.
            if in_sector == 0 && remaining >= self.sector_size && aligned_ok {
                let max_chunk = (remaining / self.sector_size).min(2048) * self.sector_size;
                let st = unsafe {
                    ((*self.bio).read_blocks)(
                        self.bio,
                        self.media_id,
                        lba,
                        max_chunk,
                        dst_ptr as *mut c_void,
                    )
                };
                if st != EFI_SUCCESS {
                    return Err("BlockIo direct read failed");
                }
                cur_off += max_chunk as u64;
                dst_pos += max_chunk;
                remaining -= max_chunk;
                continue;
            }

            let sectors_needed = ((in_sector + remaining + self.sector_size - 1) / self.sector_size)
                .min(self.scratch_cap / self.sector_size)
                .max(1);
            let read_len = sectors_needed * self.sector_size;
            let st = unsafe {
                ((*self.bio).read_blocks)(
                    self.bio,
                    self.media_id,
                    lba,
                    read_len,
                    self.scratch as *mut c_void,
                )
            };
            if st != EFI_SUCCESS {
                return Err("BlockIo buffered read failed");
            }
            let avail = read_len - in_sector;
            let take = remaining.min(avail);
            unsafe {
                core::ptr::copy_nonoverlapping(
                    self.scratch.add(in_sector),
                    dst.as_mut_ptr().add(dst_pos),
                    take,
                );
            }
            cur_off += take as u64;
            dst_pos += take;
            remaining -= take;
        }
        Ok(())
    }
}

struct Ext4Super {
    block_size: u64,
    inodes_per_group: u32,
    inode_size: u16,
    desc_size: u16,
    is_64bit: bool,
}

fn le16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}

fn le32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}

fn parse_super(reader: &BlockReader) -> Option<Ext4Super> {
    let mut sb = [0u8; 1024];
    reader.read_bytes(1024, &mut sb).ok()?;
    if le16(&sb, 0x38) != EXT4_SUPER_MAGIC {
        return None;
    }
    let log_bs = le32(&sb, 0x18);
    if log_bs > 6 {
        return None;
    }
    let block_size = 1024u64 << log_bs;
    let inodes_per_group = le32(&sb, 0x28);
    if inodes_per_group == 0 {
        return None;
    }
    let rev = le32(&sb, 0x4C);
    let inode_size = if rev >= 1 { le16(&sb, 0x58) } else { 128 };
    let incompat = le32(&sb, 0x60);
    let is_64bit = (incompat & EXT4_FEATURE_INCOMPAT_64BIT) != 0;
    let desc_size = if is_64bit {
        let d = le16(&sb, 0xFE);
        if d >= 64 { d } else { 64 }
    } else {
        32
    };
    Some(Ext4Super {
        block_size,
        inodes_per_group,
        inode_size,
        desc_size,
        is_64bit,
    })
}

struct Inode {
    size: u64,
    flags: u32,
    block_data: [u8; 60],
}

fn read_inode(reader: &BlockReader, sb: &Ext4Super, ino: u32) -> Result<Inode, &'static str> {
    if ino == 0 {
        return Err("inode 0");
    }
    let group = (ino - 1) / sb.inodes_per_group;
    let index = (ino - 1) % sb.inodes_per_group;

    let bgdt_block = if sb.block_size == 1024 { 2u64 } else { 1u64 };
    let desc_off = bgdt_block * sb.block_size + (group as u64) * (sb.desc_size as u64);
    let mut desc = [0u8; 64];
    reader.read_bytes(desc_off, &mut desc[..sb.desc_size.min(64) as usize])?;

    let mut inode_table_blk = le32(&desc, 8) as u64;
    if sb.is_64bit && sb.desc_size >= 64 {
        inode_table_blk |= (le32(&desc, 40) as u64) << 32;
    }

    let inode_off = inode_table_blk * sb.block_size + (index as u64) * (sb.inode_size as u64);
    let mut raw = [0u8; 256];
    let to_read = (sb.inode_size as usize).min(256);
    reader.read_bytes(inode_off, &mut raw[..to_read])?;

    let size_lo = le32(&raw, 0x04) as u64;
    let size_hi = le32(&raw, 0x6C) as u64;
    let flags = le32(&raw, 0x20);
    let mut block_data = [0u8; 60];
    block_data.copy_from_slice(&raw[0x28..0x64]);

    Ok(Inode {
        size: size_lo | (size_hi << 32),
        flags,
        block_data,
    })
}

fn read_extent_node(
    reader: &BlockReader,
    sb: &Ext4Super,
    node: &[u8],
    file_size: u64,
    dst: *mut u8,
) -> Result<(), &'static str> {
    if node.len() < 12 || le16(node, 0) != EXT4_EXT_MAGIC {
        return Err("bad extent header");
    }
    let entries = le16(node, 2) as usize;
    let depth = le16(node, 6);

    if depth == 0 {
        for i in 0..entries {
            let off = 12 + i * 12;
            if off + 12 > node.len() {
                break;
            }
            let ee_block = le32(node, off) as u64;
            let raw_len = le16(node, off + 4);
            let is_unwritten = raw_len > 32768;
            let ee_len = if is_unwritten { raw_len - 32768 } else { raw_len } as u64;
            let start_hi = le16(node, off + 6) as u64;
            let start_lo = le32(node, off + 8) as u64;
            let phys_blk = start_lo | (start_hi << 32);

            let byte_pos = ee_block * sb.block_size;
            if byte_pos >= file_size {
                continue;
            }
            let max_bytes = (ee_len * sb.block_size).min(file_size - byte_pos);
            let dst_slice = unsafe {
                core::slice::from_raw_parts_mut(dst.add(byte_pos as usize), max_bytes as usize)
            };
            if is_unwritten {
                dst_slice.fill(0);
            } else {
                reader.read_bytes(phys_blk * sb.block_size, dst_slice)?;
            }
        }
        return Ok(());
    }

    let mut child_buf = [0u8; 4096];
    let bs = (sb.block_size as usize).min(4096);
    for i in 0..entries {
        let off = 12 + i * 12;
        if off + 12 > node.len() {
            break;
        }
        let leaf_lo = le32(node, off + 4) as u64;
        let leaf_hi = le16(node, off + 8) as u64;
        let child_blk = leaf_lo | (leaf_hi << 32);
        reader.read_bytes(child_blk * sb.block_size, &mut child_buf[..bs])?;
        read_extent_node(reader, sb, &child_buf[..bs], file_size, dst)?;
    }
    Ok(())
}

fn read_inode_data(
    reader: &BlockReader,
    sb: &Ext4Super,
    inode: &Inode,
    dst: *mut u8,
) -> Result<(), &'static str> {
    if (inode.flags & EXT4_EXTENTS_FL) == 0 {
        return Err("non-extent ext4 inode unsupported");
    }
    read_extent_node(reader, sb, &inode.block_data, inode.size, dst)
}

fn find_in_dir(
    reader: &BlockReader,
    sb: &Ext4Super,
    dir_ino: u32,
    name: &[u8],
    bs: &BootServices,
) -> Result<Option<u32>, &'static str> {
    let inode = read_inode(reader, sb, dir_ino)?;
    let size = inode.size as usize;
    if size == 0 || size > 16 * 1024 * 1024 {
        return Ok(None);
    }
    let pages = (size + 4095) / 4096;
    let mut buf_addr = 0u64;
    if (bs.allocate_pages)(ALLOCATE_ANY_PAGES, EFI_LOADER_DATA, pages, &mut buf_addr) != EFI_SUCCESS {
        return Err("dir alloc failed");
    }
    unsafe {
        core::ptr::write_bytes(buf_addr as *mut u8, 0, pages * 4096);
    }
    let res = read_inode_data(reader, sb, &inode, buf_addr as *mut u8);
    let mut found = None;
    if res.is_ok() {
        let data = unsafe { core::slice::from_raw_parts(buf_addr as *const u8, size) };
        let mut pos = 0usize;
        while pos + 8 <= data.len() {
            let entry_ino = le32(data, pos);
            let rec_len = le16(data, pos + 4) as usize;
            let name_len = data[pos + 6] as usize;
            if rec_len < 8 || pos + rec_len > data.len() {
                break;
            }
            if entry_ino != 0 && pos + 8 + name_len <= data.len() {
                let entry_name = &data[pos + 8..pos + 8 + name_len];
                if entry_name == name {
                    found = Some(entry_ino);
                    break;
                }
            }
            pos += rec_len;
        }
    }
    let _ = (bs.free_pages)(buf_addr, pages);
    res?;
    Ok(found)
}

pub struct LoadedExt4File {
    pub phys_addr: u64,
    pub size: u64,
}

/// Attempts to open and load `path_parts` (e.g. `&["cranos", "model.gguf"]`) from
/// a UEFI BlockIo logical partition containing an ext4 filesystem.
/// Returns a 2 MiB-aligned physical buffer on success.
pub fn try_load_ext4_file(
    bs: &BootServices,
    st: &SystemTable,
    bio: *mut BlockIoProtocol,
    path_parts: &[&str],
    label: &str,
) -> Option<LoadedExt4File> {
    let media = unsafe { &*(*bio).media };
    if !media.media_present || !media.logical_partition || media.block_size == 0 {
        return None;
    }
    let io_align = (media.io_align as usize).max(1);
    let mut scratch_addr = 0u64;
    let scratch_pages = 16usize; // 64 KiB page-aligned scratch buffer
    if (bs.allocate_pages)(ALLOCATE_ANY_PAGES, EFI_LOADER_DATA, scratch_pages, &mut scratch_addr)
        != EFI_SUCCESS
    {
        return None;
    }
    let reader = BlockReader {
        bio,
        media_id: media.media_id,
        sector_size: media.block_size as usize,
        io_align,
        scratch: scratch_addr as *mut u8,
        scratch_cap: scratch_pages * 4096,
    };

    let result = (|| -> Option<LoadedExt4File> {
        let sb = parse_super(&reader)?;
        let mut cur_ino = ROOT_INO;
        for part in path_parts {
            cur_ino = find_in_dir(&reader, &sb, cur_ino, part.as_bytes(), bs).ok()??;
        }
        let file_inode = read_inode(&reader, &sb, cur_ino).ok()?;
        if file_inode.size == 0 {
            return None;
        }
        print_ascii(st, "[cranos-boot] Reading ");
        print_ascii(st, label);
        print_ascii(st, " (");
        print_u64(st, file_inode.size / (1024 * 1024));
        print_ascii(st, " MiB)...\n");

        // Allocate with 2 MiB alignment padding so weights start on a 2 MiB boundary.
        let extra = 2 * 1024 * 1024u64;
        let total_pages = ((file_inode.size + extra + 4095) / 4096) as usize;
        let mut raw_addr = 0u64;
        if (bs.allocate_pages)(ALLOCATE_ANY_PAGES, EFI_LOADER_DATA, total_pages, &mut raw_addr)
            != EFI_SUCCESS
        {
            return None;
        }
        let aligned = (raw_addr + extra - 1) & !(extra - 1);
        read_inode_data(&reader, &sb, &file_inode, aligned as *mut u8).ok()?;
        Some(LoadedExt4File {
            phys_addr: aligned,
            size: file_inode.size,
        })
    })();

    let _ = (bs.free_pages)(scratch_addr, scratch_pages);
    result
}
