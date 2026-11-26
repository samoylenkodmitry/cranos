use alloc::vec::Vec;

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct RsdpV1 {
    signature: [u8; 8],
    checksum: u8,
    oem_id: [u8; 6],
    revision: u8,
    rsdt_addr: u32,
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct RsdpV2 {
    v1: RsdpV1,
    length: u32,
    xsdt_addr: u64,
    extended_checksum: u8,
    reserved: [u8; 3],
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct SdtHeader {
    signature: [u8; 4],
    length: u32,
    revision: u8,
    checksum: u8,
    oem_id: [u8; 6],
    oem_table_id: [u8; 8],
    oem_revision: u32,
    creator_id: u32,
    creator_revision: u32,
}

pub struct MadtInfo {
    pub lapic_addr: u64,
    pub apic_ids: Vec<u32>,
}

fn find_sdt(rsdp_addr: u64, target_sig: &[u8; 4]) -> Option<u64> {
    if rsdp_addr == 0 {
        return None;
    }
    let rsdp1 = unsafe { core::ptr::read_unaligned(rsdp_addr as *const RsdpV1) };
    if &rsdp1.signature != b"RSD PTR " {
        return None;
    }

    if rsdp1.revision >= 2 {
        let rsdp2 = unsafe { core::ptr::read_unaligned(rsdp_addr as *const RsdpV2) };
        if rsdp2.xsdt_addr != 0 {
            let hdr = unsafe { core::ptr::read_unaligned(rsdp2.xsdt_addr as *const SdtHeader) };
            let hdr_size = core::mem::size_of::<SdtHeader>();
            if (hdr.length as usize) > hdr_size {
                let count = (hdr.length as usize - hdr_size) / 8;
                let entries_ptr = (rsdp2.xsdt_addr as usize + hdr_size) as *const u64;
                for i in 0..count {
                    let addr = unsafe { core::ptr::read_unaligned(entries_ptr.add(i)) };
                    if addr == 0 {
                        continue;
                    }
                    let child = unsafe { core::ptr::read_unaligned(addr as *const SdtHeader) };
                    if &child.signature == target_sig {
                        return Some(addr);
                    }
                }
            }
        }
    }

    if rsdp1.rsdt_addr != 0 {
        let rsdt = rsdp1.rsdt_addr as u64;
        let hdr = unsafe { core::ptr::read_unaligned(rsdt as *const SdtHeader) };
        let hdr_size = core::mem::size_of::<SdtHeader>();
        if (hdr.length as usize) > hdr_size {
            let count = (hdr.length as usize - hdr_size) / 4;
            let entries_ptr = (rsdt as usize + hdr_size) as *const u32;
            for i in 0..count {
                let addr = unsafe { core::ptr::read_unaligned(entries_ptr.add(i)) } as u64;
                if addr == 0 {
                    continue;
                }
                let child = unsafe { core::ptr::read_unaligned(addr as *const SdtHeader) };
                if &child.signature == target_sig {
                    return Some(addr);
                }
            }
        }
    }

    None
}

pub fn parse_madt(rsdp_addr: u64) -> Option<MadtInfo> {
    let madt_addr = find_sdt(rsdp_addr, b"APIC")?;
    let hdr = unsafe { core::ptr::read_unaligned(madt_addr as *const SdtHeader) };
    let total_len = hdr.length as usize;
    let base_hdr_len = core::mem::size_of::<SdtHeader>() + 8; // + lapic_addr(4) + flags(4)
    if total_len < base_hdr_len {
        return None;
    }

    let mut lapic_addr = unsafe {
        core::ptr::read_unaligned((madt_addr as usize + core::mem::size_of::<SdtHeader>()) as *const u32)
    } as u64;

    let mut apic_ids = Vec::new();
    let mut off = base_hdr_len;
    while off + 2 <= total_len {
        let ptr = (madt_addr as usize + off) as *const u8;
        let entry_type = unsafe { *ptr };
        let entry_len = unsafe { *ptr.add(1) } as usize;
        if entry_len < 2 || off + entry_len > total_len {
            break;
        }
        match entry_type {
            // Processor Local APIC
            0 if entry_len >= 8 => {
                let apic_id = unsafe { *ptr.add(3) } as u32;
                let flags = unsafe { core::ptr::read_unaligned(ptr.add(4) as *const u32) };
                // Bit 0: Processor Enabled, Bit 1: Online Capable
                if (flags & 0x3) != 0 && !apic_ids.contains(&apic_id) {
                    apic_ids.push(apic_id);
                }
            }
            // Local APIC Address Override
            5 if entry_len >= 12 => {
                lapic_addr = unsafe { core::ptr::read_unaligned(ptr.add(4) as *const u64) };
            }
            // Processor Local x2APIC
            9 if entry_len >= 16 => {
                let x2_id = unsafe { core::ptr::read_unaligned(ptr.add(4) as *const u32) };
                let flags = unsafe { core::ptr::read_unaligned(ptr.add(8) as *const u32) };
                if (flags & 0x3) != 0 && !apic_ids.contains(&x2_id) {
                    apic_ids.push(x2_id);
                }
            }
            _ => {}
        }
        off += entry_len;
    }

    Some(MadtInfo {
        lapic_addr,
        apic_ids,
    })
}
