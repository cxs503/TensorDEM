"""Patch torch .so files: drop the GLIBCXX_3.4.32 version requirement on
std::ios_base_library_init (provided at runtime by the LD_PRELOAD shim).

Sets the .gnu.version entry of the named undefined symbol to 1 (BASE),
turning the versioned reference into an unversioned one.  Backs up the
original first.  Idempotent: skips if the entry is already 1.
"""
import struct
import sys


def patch(path: str, sym: str) -> None:
    data = bytearray(open(path, "rb").read())
    assert data[:4] == b"\x7fELF" and data[4] == 2, "need ELF64"
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    shstr_off = shstr_size = None
    sections = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sh = struct.unpack_from("<IIQQQQIIQQ", data, off)
        sections.append(sh)
    shstr = data[sections[e_shstrndx][4]:sections[e_shstrndx][4] + sections[e_shstrndx][5]]

    def sec(name):
        for sh in sections:
            nm = shstr[sh[0]:shstr.index(b"\x00", sh[0])].decode()
            if nm == name:
                return sh
        raise RuntimeError(f"section {name} not found in {path}")

    dynsym = sec(".dynsym")
    ver = sec(".gnu.version")
    dynstr = data[sections[dynsym[6]][4]:sections[dynsym[6]][4] + sections[dynsym[6]][5]]
    # find symbol index in .dynsym
    idx = None
    for i in range(dynsym[5] // 24):
        off = dynsym[4] + i * 24
        st_name = struct.unpack_from("<I", data, off)[0]
        nm = dynstr[st_name:dynstr.index(b"\x00", st_name)].decode()
        if nm == sym:
            idx = i
            break
    if idx is None:
        print(f"  {path}: symbol not found, skip")
        return
    voff = ver[4] + idx * 2
    cur = struct.unpack_from("<H", data, voff)[0]
    if cur == 1:
        print(f"  {path}: already patched")
        return
    struct.pack_into("<H", data, voff, 1)
    open(path, "wb").write(data)
    print(f"  {path}: dynsym[{idx}] version {cur} -> 1 (BASE)")


if __name__ == "__main__":
    import shutil
    for p in sys.argv[1:]:
        bak = p + ".orig"
        if not __import__("os").path.exists(bak):
            shutil.copy2(p, bak)
        patch(p, "_ZSt21ios_base_library_initv")
