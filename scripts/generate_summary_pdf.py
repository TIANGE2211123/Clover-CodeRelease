from pathlib import Path

WIDTH, HEIGHT = 612, 792  # Letter
LEFT = 50
TOP = 760

sections = [
    ("CLOVER App Summary", 16),
    ("", 11),
    ("What it is", 12),
    ("- CLOVER is a GPU-native exact k-nearest-neighbor (kNN) research codebase accompanying the paper", 10),
    ("  'CLOVER: A GPU-native, Spatio-graph-based Approach to Exact kNN'.", 10),
    ("- It benchmarks CUDA-based kNN strategies (linear-scan + spatio-graph variants) via", 10),
    ("  one executable: linear-scans.", 10),
    ("", 10),
    ("Who it's for", 12),
    ("- Primary persona: GPU systems/ML researchers and performance engineers evaluating", 10),
    ("  exact kNN implementations and CUDA-kernel optimizations.", 10),
    ("", 10),
    ("What it does", 12),
    ("- Builds a CUDA/C++ executable (linear-scans) from src/*.cu and include/*.cuh.", 10),
    ("- Supports CLI-selected algorithms: Bitonic, Warpwise, Hubs variants, FAISS variants,", 10),
    ("  and Treelogy KD-tree.", 10),
    ("- Runs synthetic experiments with fixed-seed random 3D data.", 10),
    ("- Runs mesh experiments by scanning ../meshes and loading .txt point files.", 10),
    ("- Copies points/queries to GPU, dispatches selected kernels, returns neighbors+distances.", 10),
    ("- Prints timing tuples in release mode and can print outputs for inspection.", 10),
    ("- Optionally enables FAISS-backed paths via build flag USE_FAISS/LINK_FAISS.", 10),
    ("", 10),
    ("How it works (architecture)", 12),
    ("- Build layer: CMake compiles CUDA sources, links cuBLAS, conditionally links", 10),
    ("  FAISS/OpenMP.", 10),
    ("- Orchestration: linear-scans.cu parses algorithm index, prepares data, and loops", 10),
    ("  benchmark sizes.", 10),
    ("- Data flow: host arrays/vectors -> cudaMemcpy to device -> algorithm kernel ->", 10),
    ("  host vectors -> stdout.", 10),
    ("- Algorithm layer: include/ provides bitonic, warpwise, hubs, FAISS wrappers,", 10),
    ("  and Treelogy integration.", 10),
    ("- External services/components: Not found in repo.", 10),
    ("", 10),
    ("How to run (minimal)", 12),
    ("1) Install CUDA+GCC (README cites CUDA 12.6, GCC 13.3 for local Linux).", 10),
    ("2) mkdir build && cd build", 10),
    ("3) cmake -DCMAKE_BUILD_TYPE=Release ..", 10),
    ("4) make", 10),
    ("5) ./linear-scans 2  (or another algorithm index)", 10),
]

def esc(s: str) -> str:
    return s.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')

content = ["BT"]
y = TOP
for text, size in sections:
    content.append(f"/F1 {size} Tf")
    content.append(f"1 0 0 1 {LEFT} {y} Tm")
    content.append(f"({esc(text)}) Tj")
    y -= 15 if size >= 12 else 13
content.append("ET")
stream = "\n".join(content).encode("latin-1", "replace")

objs = []
objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
objs.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {WIDTH} {HEIGHT}] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".encode())
objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
objs.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")

pdf = bytearray(b"%PDF-1.4\n")
offsets = [0]
for i, obj in enumerate(objs, 1):
    offsets.append(len(pdf))
    pdf.extend(f"{i} 0 obj\n".encode())
    pdf.extend(obj)
    pdf.extend(b"\nendobj\n")

xref_pos = len(pdf)
pdf.extend(f"xref\n0 {len(objs)+1}\n".encode())
pdf.extend(b"0000000000 65535 f \n")
for off in offsets[1:]:
    pdf.extend(f"{off:010d} 00000 n \n".encode())
pdf.extend(f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode())

out = Path("docs/clover_app_summary.pdf")
out.write_bytes(pdf)
print(out)
