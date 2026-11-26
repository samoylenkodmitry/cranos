use core::alloc::{GlobalAlloc, Layout};
use core::ptr::null_mut;
use core::sync::atomic::{AtomicBool, Ordering};

use cranos_boot_info::{BootInfo, MemoryKind};

const PAGE_SIZE: usize = 4096;
const SLAB_CLASSES: [usize; 9] = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096];

struct SpinLock {
    locked: AtomicBool,
}

impl SpinLock {
    const fn new() -> Self {
        Self {
            locked: AtomicBool::new(false),
        }
    }

    #[inline]
    fn lock(&self) {
        while self
            .locked
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            while self.locked.load(Ordering::Relaxed) {
                core::hint::spin_loop();
            }
        }
    }

    #[inline]
    fn unlock(&self) {
        self.locked.store(false, Ordering::Release);
    }
}

#[repr(C)]
struct FreeNode {
    next: *mut FreeNode,
}

#[repr(C)]
struct LargeBlock {
    size: usize,
    next: *mut LargeBlock,
}

struct Region {
    cur: usize,
    end: usize,
}

struct HeapInner {
    regions: [Region; 128],
    region_count: usize,
    total_bytes: usize,
    allocated_bytes: usize,
    slabs: [*mut FreeNode; SLAB_CLASSES.len()],
    large_free: *mut LargeBlock,
}

impl HeapInner {
    const fn new() -> Self {
        Self {
            regions: [const { Region { cur: 0, end: 0 } }; 128],
            region_count: 0,
            total_bytes: 0,
            allocated_bytes: 0,
            slabs: [null_mut(); SLAB_CLASSES.len()],
            large_free: null_mut(),
        }
    }

    fn add_region(&mut self, start: usize, end: usize) {
        let aligned_start = (start + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
        let aligned_end = end & !(PAGE_SIZE - 1);
        if aligned_end <= aligned_start + PAGE_SIZE || self.region_count >= self.regions.len() {
            return;
        }
        self.regions[self.region_count] = Region {
            cur: aligned_start,
            end: aligned_end,
        };
        self.region_count += 1;
        self.total_bytes += aligned_end - aligned_start;
    }

    fn bump_raw(&mut self, size: usize, align: usize) -> *mut u8 {
        self.bump_raw_limit(size, align, usize::MAX)
    }

    fn bump_raw_limit(&mut self, size: usize, align: usize, max_addr: usize) -> *mut u8 {
        let align = align.max(16);
        for i in 0..self.region_count {
            let r = &mut self.regions[i];
            let aligned = (r.cur + align - 1) & !(align - 1);
            if let Some(next) = aligned.checked_add(size)
                && next <= r.end
                && next <= max_addr
            {
                r.cur = next;
                self.allocated_bytes += size;
                return aligned as *mut u8;
            }
        }
        null_mut()
    }

    fn slab_index(size: usize, align: usize) -> Option<usize> {
        let req = size.max(align);
        for (i, &cls) in SLAB_CLASSES.iter().enumerate() {
            if req <= cls {
                return Some(i);
            }
        }
        None
    }

    unsafe fn alloc(&mut self, layout: Layout) -> *mut u8 {
        let size = layout.size().max(1);
        let align = layout.align().max(8);

        if let Some(idx) = Self::slab_index(size, align) {
            let head = self.slabs[idx];
            if !head.is_null() {
                unsafe {
                    self.slabs[idx] = (*head).next;
                }
                return head as *mut u8;
            }
            // Refill this slab class with a 64 KiB chunk (or 1 page for 4 KiB).
            let cls = SLAB_CLASSES[idx];
            let chunk_size = if cls <= 1024 { 64 * 1024 } else { 16 * 1024 };
            let chunk = self.bump_raw(chunk_size, PAGE_SIZE);
            if chunk.is_null() {
                return null_mut();
            }
            let count = chunk_size / cls;
            for n in 1..count {
                let node = unsafe { chunk.add(n * cls) as *mut FreeNode };
                unsafe {
                    (*node).next = self.slabs[idx];
                }
                self.slabs[idx] = node;
            }
            return chunk;
        }

        // Large allocation (> 4096 B): align to page size (or requested alignment)
        // and check the large free list first.
        let eff_align = align.max(PAGE_SIZE);
        let alloc_size = (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);

        let mut prev: *mut LargeBlock = null_mut();
        let mut cur = self.large_free;
        while !cur.is_null() {
            unsafe {
                let addr = cur as usize;
                let blk_size = (*cur).size;
                if (addr & (eff_align - 1)) == 0 && blk_size >= alloc_size && blk_size <= alloc_size * 2 {
                    if prev.is_null() {
                        self.large_free = (*cur).next;
                    } else {
                        (*prev).next = (*cur).next;
                    }
                    return cur as *mut u8;
                }
                prev = cur;
                cur = (*cur).next;
            }
        }

        self.bump_raw(alloc_size, eff_align)
    }

    unsafe fn dealloc(&mut self, ptr: *mut u8, layout: Layout) {
        if ptr.is_null() {
            return;
        }
        let size = layout.size().max(1);
        let align = layout.align().max(8);

        if let Some(idx) = Self::slab_index(size, align) {
            let node = ptr as *mut FreeNode;
            unsafe {
                (*node).next = self.slabs[idx];
            }
            self.slabs[idx] = node;
            return;
        }

        let alloc_size = (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
        if alloc_size >= core::mem::size_of::<LargeBlock>() {
            let blk = ptr as *mut LargeBlock;
            unsafe {
                (*blk).size = alloc_size;
                (*blk).next = self.large_free;
            }
            self.large_free = blk;
        }
    }
}

pub(crate) struct KernelAllocator {
    lock: SpinLock,
    inner: core::cell::UnsafeCell<HeapInner>,
}

unsafe impl Sync for KernelAllocator {}

#[global_allocator]
pub(crate) static ALLOCATOR: KernelAllocator = KernelAllocator {
    lock: SpinLock::new(),
    inner: core::cell::UnsafeCell::new(HeapInner::new()),
};

unsafe impl GlobalAlloc for KernelAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.lock.lock();
        let ptr = unsafe { (*self.inner.get()).alloc(layout) };
        self.lock.unlock();
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.lock.lock();
        unsafe { (*self.inner.get()).dealloc(ptr, layout) };
        self.lock.unlock();
    }
}

pub struct MemoryStats {
    pub total_bytes: usize,
    pub allocated_bytes: usize,
    pub region_count: usize,
}

pub fn init(boot_info: &BootInfo) -> MemoryStats {
    let regions = unsafe { boot_info.memory_regions() };
    ALLOCATOR.lock.lock();
    let inner = unsafe { &mut *ALLOCATOR.inner.get() };

    // Keep low 2 MiB untouched for legacy/AP trampoline/ACPI tables.
    const MIN_HEAP_ADDR: u64 = 2 * 1024 * 1024;
    for r in regions {
        if r.kind != MemoryKind::Usable {
            continue;
        }
        let start = r.start.max(MIN_HEAP_ADDR);
        let end = r.start.saturating_add(r.len);
        if end > start + 64 * 1024 {
            inner.add_region(start as usize, end as usize);
        }
    }

    // Sort regions largest-first so big contiguous allocations succeed immediately.
    for i in 0..inner.region_count {
        for j in (i + 1)..inner.region_count {
            let len_i = inner.regions[i].end - inner.regions[i].cur;
            let len_j = inner.regions[j].end - inner.regions[j].cur;
            if len_j > len_i {
                inner.regions.swap(i, j);
            }
        }
    }

    let stats = MemoryStats {
        total_bytes: inner.total_bytes,
        allocated_bytes: inner.allocated_bytes,
        region_count: inner.region_count,
    };
    ALLOCATOR.lock.unlock();
    stats
}

pub fn stats() -> MemoryStats {
    ALLOCATOR.lock.lock();
    let inner = unsafe { &*ALLOCATOR.inner.get() };
    let stats = MemoryStats {
        total_bytes: inner.total_bytes,
        allocated_bytes: inner.allocated_bytes,
        region_count: inner.region_count,
    };
    ALLOCATOR.lock.unlock();
    stats
}

/// Allocates a single zeroed 4 KiB-aligned physical page (identity-mapped).
pub fn alloc_zeroed_page_phys() -> Option<u64> {
    ALLOCATOR.lock.lock();
    let ptr = unsafe { (*ALLOCATOR.inner.get()).bump_raw(PAGE_SIZE, PAGE_SIZE) };
    ALLOCATOR.lock.unlock();
    if ptr.is_null() {
        return None;
    }
    unsafe {
        core::ptr::write_bytes(ptr, 0, PAGE_SIZE);
    }
    Some(ptr as u64)
}

/// Allocates contiguous 4 KiB-aligned zeroed physical memory (e.g. for xHCI rings / AP stacks).
/// Prefers memory below 4 GiB for maximum DMA controller compatibility, falling back to high RAM.
pub fn alloc_dma_zeroed(size: usize, align: usize) -> Option<u64> {
    let align = align.max(PAGE_SIZE);
    let size = (size + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
    ALLOCATOR.lock.lock();
    let inner = unsafe { &mut *ALLOCATOR.inner.get() };
    let mut ptr = inner.bump_raw_limit(size, align, 0x1_0000_0000usize);
    if ptr.is_null() {
        ptr = inner.bump_raw(size, align);
    }
    ALLOCATOR.lock.unlock();
    if ptr.is_null() {
        return None;
    }
    unsafe {
        core::ptr::write_bytes(ptr, 0, size);
    }
    Some(ptr as u64)
}
