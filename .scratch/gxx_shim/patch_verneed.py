"""Unlink the GLIBCXX_3.4.32 version-requirement node from .gnu.version_r.

The glibc loader checks, for every verneed node of a loaded object, that the
named providing file defines a version node with the required name.  The
torch 2.4.0 loongarch build requires GLIBCXX_3.4.32 from libstdc++.so.6,
which the Loongnix 23 system library (6.0.30) does not define.  The symbol
itself (std::ios_base_library_init) is supplied at runtime by the LD_PRELOAD
shim, so the requirement node can be safely neutralised: zero the node
(vn_cnt=0, vn_aux=0, vn_next=0) and, if it is not the first node, relink the
previous node's vn_next around it.
"""
import struct
import sys


def patch(path: str, version_name: str) -> None:
    data = bytearray(open(path, "rb").read())
    assert data[:4] == b"\x7fELF" and data[4] == 2, "need ELF64"
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    sections = []
    for i in range(e_shnum):
        off = e_shoff + i * e_shentsize
        sections.append(struct.unpack_from("<IIQQQQIIQQ", data, off))
    shstr = data[sections[e_shstrndx][4]:sections[e_shstrndx][4] + sections[e_shstrndx][5]]

    def sec(name):
        for sh in sections:
            nm = shstr[sh[0]:shstr.index(b"\x00", sh[0])].decode()
            if nm == name:
                return sh
        raise RuntimeError(f"section {name} not found in {path}")

    dynstr_sh = sections[sec(".dynsym")[6]]
    dynstr = data[dynstr_sh[4]:dynstr_sh[4] + dynstr_sh[5]]
    ver_r = sec(".gnu.version_r")
    base = ver_r[4]
    size = ver_r[5]

    def cstr(off):
        end = dynstr.index(b"\x00", off)
        return dynstr[off:end].decode()

    # walk verneed chain
    prev = None
    off = 0
    found = None
    while off < size:
        vn_version, vn_file, vn_cnt, vn_aux, vn_next = struct.unpack_from(
            "<HHIII", data, base + off)
        # walk verfiles of this node
        aux = vn_aux
        for _ in range(vn_cnt):
            vf_name, vf_hash, vf_flags, vf_next, vf_aux = struct.unpack_from(
                "<HHHII", data, base + aux)
            if cstr(vf_name) == version_name:
                found = (off, prev)
                break
            aux = vf_next
        if found:
            break
        prev = off
        off = vn_next
    if found is None:
        print(f"  {path}: no {version_name} verneed node, skip")
        return
    node_off, prev_off = found
    if prev_off is None:
        # first node: zero it in place (chain starts here)
        struct.pack_into("<HHIII", data, base + node_off, 0, 0, 0, 0, 0)
    else:
        # relink previous node around this one
        _, _, _, _, vn_next = struct.unpack_from("<HHIII", data, base + node_off)
        struct.pack_into("<I", data, base + prev_off + 12, vn_next)
        struct.pack_into("<HHIII", data, base + node_off, 0, 0, 0, 0, 0)
    open(path, "wb").write(data)
    print(f"  {path}: removed {version_name} verneed node at {node_off}")


if __name__ == "__main__":
    import os
    import shutil
    for p in sys.argv[1:]:
        bak = p + ".orig"
        if not os.path.exists(bak):
            shutil.copy2(p, bak)
        patch(p, "GLIBCXX_3.4.32")
